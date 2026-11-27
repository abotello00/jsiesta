"""Electron repulsion integrals of the hybrid-functional stage (Stage 4 of ``docs/design/HYBRID_FUNCTIONALS_PLAN.md``).

External reference: PySCF (libcint). The comparison does not assume PySCF's ordering, sign or normalisation of the
real harmonics: both sets of shell functions are evaluated on random points and the linear map between them is
fitted (and required to be exact), then applied to the integrals.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from jax_siesta.core.ffi import load_ffi_library
from jax_siesta.symmetry.projection import _real_solid_harmonics_unit

DPTR = ctypes.POINTER(ctypes.c_double)
IPTR = ctypes.POINTER(ctypes.c_int32)


def _eri(shells: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]], c_full: float, c_sr: float, omega: float) -> np.ndarray:
    """``(ab|cd)`` of four shells ``(l, exponents, coefficients, centre)`` from ``jsiesta_eri_quartet``."""
    lib = load_ffi_library()
    fn = lib.jsiesta_eri_quartet
    fn.argtypes = [IPTR, IPTR, DPTR, DPTR, DPTR, ctypes.c_double, ctypes.c_double, ctypes.c_double, DPTR]
    fn.restype = ctypes.c_int32
    ls = np.array([s[0] for s in shells], dtype=np.int32)
    npr = np.array([len(s[1]) for s in shells], dtype=np.int32)
    exps = np.ascontiguousarray(np.concatenate([s[1] for s in shells]), dtype=np.float64)
    coefs = np.ascontiguousarray(np.concatenate([s[2] for s in shells]), dtype=np.float64)
    cen = np.ascontiguousarray(np.stack([s[3] for s in shells]), dtype=np.float64)
    out = np.zeros(tuple(2 * int(ll) + 1 for ll in ls))
    rc = fn(ls.ctypes.data_as(IPTR), npr.ctypes.data_as(IPTR), exps.ctypes.data_as(DPTR), coefs.ctypes.data_as(DPTR),
            cen.ctypes.data_as(DPTR), c_full, c_sr, omega, out.ctypes.data_as(DPTR))
    assert rc == 0
    return out


def _shell_values(shell: tuple[int, np.ndarray, np.ndarray, np.ndarray], pts: np.ndarray) -> np.ndarray:
    """jsiesta shell functions ``C_lm(r - A) sum_i c_i exp(-a_i |r - A|^2)`` on points, shape ``(n_pts, 2l+1)``."""
    ll, exps, coefs, centre = shell
    d = pts - centre[None, :]
    r2 = np.sum(d * d, axis=1)
    radial = np.sum(coefs[None, :] * np.exp(-exps[None, :] * r2[:, None]), axis=1)
    c_lm = _real_solid_harmonics_unit(d, ll)[:, ll * ll : (ll + 1) * (ll + 1)]
    return c_lm * radial[:, None]


def _random_shell(rng: np.random.Generator, ll: int, spread: float) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    n_prim = int(rng.integers(1, 4))
    return ll, rng.uniform(0.25, 2.5, n_prim), rng.uniform(0.3, 1.0, n_prim) * rng.choice([-1.0, 1.0], n_prim), rng.uniform(-spread, spread, 3)


def _pyscf_reference(shells, c_full: float, c_sr: float, omega: float) -> np.ndarray:
    """The same integrals from PySCF, transformed to jsiesta's shell functions."""
    from pyscf import gto

    mol = gto.Mole()
    mol.unit = "Bohr"
    mol.atom = [[f"H{i}", tuple(s[3])] for i, s in enumerate(shells)]
    # PySCF takes coefficients of normalised primitives; jsiesta's multiply the bare exp(-a r^2).
    mol.basis = {
        f"H{i}": [[s[0]] + [[float(a), float(c / gto.gto_norm(s[0], a))] for a, c in zip(s[1], s[2])]]
        for i, s in enumerate(shells)
    }
    mol.spin = 0
    mol.verbose = 0
    mol.build()
    sl = (0, 1, 1, 2, 2, 3, 3, 4)
    ref = c_full * mol.intor("int2e_sph", shls_slice=sl) if c_full != 0.0 else 0.0
    if c_sr != 0.0:
        with mol.with_range_coulomb(-omega):  # negative omega: short-range kernel erfc(omega r)/r
            ref = ref + c_sr * mol.intor("int2e_sph", shls_slice=sl)
    # Linear map between the two sets of shell functions, fitted on points and required to be exact.
    pts = np.random.default_rng(7).uniform(-2.5, 2.5, (400, 3))
    ao = mol.eval_gto("GTOval_sph", pts)
    maps, off = [], 0
    for s in shells:
        n = 2 * s[0] + 1
        mine = _shell_values(s, pts)
        u, *_ = np.linalg.lstsq(ao[:, off : off + n], mine, rcond=None)  # mine = ao @ u
        assert np.max(np.abs(ao[:, off : off + n] @ u - mine)) <= 1e-10 * np.max(np.abs(mine))
        maps.append(u)
        off += n
    return np.einsum("pqrs,pa,qb,rc,sd->abcd", ref, *maps, optimize=True)


CASES = [
    (0, 0, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 0, 1, 0), (1, 1, 1, 1), (2, 0, 0, 0), (2, 1, 0, 1), (2, 2, 0, 0),
    (2, 1, 2, 1), (2, 2, 1, 1), (2, 2, 2, 2), (3, 0, 0, 0), (3, 1, 2, 0), (3, 2, 1, 1), (3, 3, 2, 2), (3, 3, 3, 3),
    (0, 2, 3, 1), (4, 1, 0, 0), (4, 3, 2, 1),
]


@pytest.mark.parametrize("kernel", [(1.0, 0.0, 0.0), (0.0, 1.0, 0.11), (0.3, 0.7, 0.4)], ids=["coulomb", "hse", "mixed"])
def test_eri_against_pyscf(kernel: tuple[float, float, float]) -> None:
    """All shell types up to ``l = 3`` (and two with ``l = 4``), three geometries each, against libcint.

    Geometries: four separate centres; all four shells on one centre; two distant pairs (Boys argument above 30,
    the asymptotic branch). Agreement to 1e-10 of the largest element (of 1e-3 when all elements are smaller).
    """
    pytest.importorskip("pyscf")
    rng = np.random.default_rng(2026)
    worst = 0.0
    for ls in CASES:
        for geometry in ("spread", "one_centre", "distant"):
            shells = [_random_shell(rng, ll, 1.2) for ll in ls]
            if geometry == "one_centre":
                shells = [(s[0], s[1], s[2], shells[0][3]) for s in shells]
            elif geometry == "distant":
                shift = np.array([9.0, 0.0, 0.0])
                shells = [shells[0], shells[1], (*shells[2][:3], shells[2][3] + shift), (*shells[3][:3], shells[3][3] + shift)]
            mine = _eri(shells, *kernel)
            ref = _pyscf_reference(shells, *kernel)
            # Integrals that vanish by symmetry (odd shells on one centre) are compared on an absolute scale.
            err = np.max(np.abs(mine - ref)) / max(np.max(np.abs(ref)), 1e-3)
            worst = max(worst, err)
            assert err <= 1e-10, f"l={ls} geometry={geometry} kernel={kernel}: relative error {err:.2e}"
    assert worst <= 1e-10


def test_eri_identities() -> None:
    """Permutation symmetry, positivity of ``(ab|ab)`` and translation invariance (no external reference)."""
    rng = np.random.default_rng(5)
    kernel = (0.0, 1.0, 0.11)
    for ls in [(1, 2, 0, 1), (2, 2, 1, 3), (3, 1, 2, 2)]:
        a, b, c, d = (_random_shell(rng, ll, 1.5) for ll in ls)
        abcd = _eri([a, b, c, d], *kernel)
        scale = np.max(np.abs(abcd))
        assert np.max(np.abs(abcd - _eri([b, a, c, d], *kernel).transpose(1, 0, 2, 3))) <= 1e-12 * scale
        assert np.max(np.abs(abcd - _eri([a, b, d, c], *kernel).transpose(0, 1, 3, 2))) <= 1e-12 * scale
        assert np.max(np.abs(abcd - _eri([c, d, a, b], *kernel).transpose(2, 3, 0, 1))) <= 1e-12 * scale
        shift = np.array([3.1, -2.2, 0.7])
        moved = [(s[0], s[1], s[2], s[3] + shift) for s in (a, b, c, d)]
        assert np.max(np.abs(abcd - _eri(moved, *kernel))) <= 1e-11 * scale
        abab = _eri([a, b, a, b], *kernel)
        n = abab.shape[0] * abab.shape[1]
        assert np.linalg.eigvalsh(abab.reshape(n, n)).min() >= -1e-12 * np.max(np.abs(abab))


def test_derivative_integrals_against_finite_differences() -> None:
    """Derivatives with respect to the first centre and to a pair translation equal differences of the integrals.

    The integrals themselves are validated against PySCF above, so their fourth-order central differences are the
    reference; agreement to 1e-8 of the largest element for shells up to ``l = 3`` and two kernels.
    """
    lib = load_ffi_library()
    fn = lib.jsiesta_eri_quartet_mode
    fn.argtypes = [IPTR, IPTR, DPTR, DPTR, DPTR, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int32, DPTR]
    fn.restype = ctypes.c_int32

    def derivative(shells, kernel, mode: int) -> np.ndarray:  # type: ignore[no-untyped-def]
        ls = np.array([s[0] for s in shells], dtype=np.int32)
        npr = np.array([len(s[1]) for s in shells], dtype=np.int32)
        exps = np.ascontiguousarray(np.concatenate([s[1] for s in shells]))
        coefs = np.ascontiguousarray(np.concatenate([s[2] for s in shells]))
        cen = np.ascontiguousarray(np.stack([s[3] for s in shells]))
        out = np.zeros(tuple(2 * int(ll) + 1 for ll in ls))
        assert fn(ls.ctypes.data_as(IPTR), npr.ctypes.data_as(IPTR), exps.ctypes.data_as(DPTR), coefs.ctypes.data_as(DPTR),
                  cen.ctypes.data_as(DPTR), *kernel, mode, out.ctypes.data_as(DPTR)) == 0
        return out

    rng = np.random.default_rng(4)
    step = 1e-4
    for kernel in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.11)):
        for ls in ((1, 0, 0, 0), (0, 1, 1, 0), (2, 1, 0, 2), (3, 2, 2, 1), (0, 3, 1, 2)):
            shells = [_random_shell(rng, ll, 1.2) for ll in ls]
            assert np.array_equal(derivative(shells, kernel, 0), _eri(shells, *kernel))
            for d in range(3):
                for mode, moved in ((1 + d, (0,)), (4 + d, (0, 1))):

                    def at(delta: float, moved=moved, d=d) -> np.ndarray:  # type: ignore[no-untyped-def]
                        out = []
                        for i, s in enumerate(shells):
                            c = s[3].copy()
                            if i in moved:
                                c[d] += delta
                            out.append((s[0], s[1], s[2], c))
                        return _eri(out, *kernel)

                    fd = (8 * (at(step) - at(-step)) - (at(2 * step) - at(-2 * step))) / (12 * step)
                    an = derivative(shells, kernel, mode)
                    assert np.max(np.abs(an - fd)) <= 1e-8 * max(np.max(np.abs(fd)), 1e-3)

