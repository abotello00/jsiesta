r"""Exact evaluation of SIESTA's neutral-atom pair energy ``Ena`` from the tables of a SIESTA ``.ion.xml`` file.

Theory:
    SIESTA's ``Ena`` (``Src/naefs.F`` plus ``Src/dnaefs.F``) is the sum of two two-centre terms over all ordered atom
    pairs, *including* the self term :math:`R = 0` of the first one:

    .. math::
        E_{na} = \frac{1}{16\pi}\sum_{I,J,\mathbf T} \int \nabla V_I \cdot \nabla V_J
                 \;-\; \frac12 \sum_{I \ne J,\mathbf T} \Delta_{IJ}(R),
        \qquad
        \Delta_{IJ}(R) = Z_I Z_J\left[\frac{2}{\pi}\int_0^\infty \tilde f_I(k)\tilde f_J(k) j_0(kR)\,dk - \frac1R\right]

    in Rydberg units (the first term is evaluated from the screened neutral-atom potentials :math:`V_I`, ``vna``; the
    second from the normalised local pseudo-charges ``chlocal`` with form factor :math:`\tilde f`). SIESTA evaluates
    both from tables whose numerical resolution is fixed (``matel_params``: q cut-off 100 inverse Bohr, 1024 points;
    ``electrostatic.f``: a charge-normalisation tolerance ``cherr = 0.05`` followed by renormalisation), so its printed
    value deviates smoothly from the exact value of its own formula (logs/DECISIONS.md D-025).

Algorithm:
    Reads ``vna`` and ``chlocal`` from the ion files, forms their radial Fourier transforms on a fine k grid
    (to 200 inverse Bohr) and sums the two terms over neighbour images.

Invariants:
    - Agrees with SIESTA's printed ``Ena`` to about 0.02 % in absolute value (Si and H2O tests).
    - Only the pair part: the on-mesh ``DEna`` and ``Eions`` are not included.

SIESTA Mapping:
    ``naefs`` (``matel('T', vna, vna)`` / 16 pi) and ``dnaefs`` (``psover``, the ``elec_corr`` tables of ``electrostatic.f``).
"""

from __future__ import annotations

import itertools
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from jax_siesta.core.types import Structure

_RY_TO_HA = 0.5
_K = np.linspace(0.0, 200.0, 20001)


def _j0(x: np.ndarray) -> np.ndarray:
    return np.where(x > 1e-12, np.sin(x) / np.maximum(x, 1e-12), 1.0)


def _radfunc(root: ET.Element, tag: str) -> tuple[np.ndarray, np.ndarray, float]:
    rf = root.find(f"{tag}/radfunc")
    if rf is None:
        raise ValueError(f"ion file has no <{tag}> table")
    cutoff = float(rf.findtext("cutoff") or 0.0)
    data = np.array((rf.findtext("data") or "").split(), dtype=float).reshape(-1, 2)
    r, f = data[:, 0], data[:, 1]
    if r[0] > 1e-12:
        r, f = np.concatenate([[0.0], r]), np.concatenate([[f[0]], f])
    return r, f, cutoff


def _fourier(r: np.ndarray, f: np.ndarray) -> np.ndarray:
    kr = np.outer(_K, r)
    return 4.0 * np.pi * np.trapezoid(f[None, :] * r[None, :] ** 2 * _j0(kr), r, axis=1)


def read_species_tables(ion_xml: Path) -> dict[str, float | np.ndarray]:
    """Read ``vna`` and ``chlocal`` of one species and return their Fourier-space data.

    Theory:
        ``vna`` (Ry) enters the gradient term through :math:`\\tilde V(k)`; ``chlocal`` is renormalised to the valence
        charge :math:`Z` exactly as SIESTA does and gives the form factor :math:`\\tilde f(k)` with :math:`\\tilde f(0)=1`.

    Algorithm:
        Parses the ion file, Fourier transforms both radial tables on the module's k grid.

    Parameters:
        ion_xml: Path to ``<Species>.ion.xml`` written by SIESTA (``WriteIonPlotFiles true``).

    Returns:
        Dictionary with ``z``, ``vt`` (k-space ``vna``, Ry), ``f`` (form factor), ``rc_vna`` and ``rc_chlocal`` (Bohr).

    Invariants:
        ``f[0] == 1`` up to the quadrature error of the table.

    SIESTA Mapping:
        ``atm_types`` ``vna`` and ``chlocal`` radial functions.
    """
    root = ET.parse(ion_xml).getroot()
    z = float(root.findtext("valence") or 0.0)
    rv, v, cv = _radfunc(root, "vna")
    rc, c, cc = _radfunc(root, "chlocal")
    rho = -c if float(np.mean(c)) < 0 else c
    rho = rho * z / (4.0 * np.pi * np.trapezoid(rho * rc**2, rc))
    return {"z": z, "vt": _fourier(rv, v), "f": _fourier(rc, rho) / z, "rc_vna": cv, "rc_chlocal": cc}


def siesta_pair_energy_ha(structure: Structure, ion_dir: Path) -> tuple[float, float]:
    """Exact value of SIESTA's ``Ena`` formula for ``structure`` in Hartree, split as ``(naefs, dnaefs)``.

    Theory:
        See the module docstring. ``naefs`` includes the geometry-independent self term of every atom.

    Algorithm:
        Loops over ordered atom pairs and all periodic images that can lie within the cut-off sum of the table radii
        (the image range follows the cell, so compact cells are handled) and evaluates the
        k-space integrals with the trapezoid rule on a 0.01 inverse Bohr grid.

    Parameters:
        structure: Structure (cell columns, positions in Bohr).
        ion_dir: Directory with the SIESTA ``<Species>.ion.xml`` files.

    Returns:
        ``(naefs_ha, dnaefs_ha)``; their sum is SIESTA's ``Ena`` (pair part) in Hartree.

    Invariants:
        The sum is invariant under permutation of atoms and under lattice translations of the structure.

    SIESTA Mapping:
        ``naefs`` and ``dnaefs`` (``Src/naefs.F``, ``Src/dnaefs.F``).
    """
    labels = [structure.species_labels[i] for i in structure.species_indices]
    tables = {lbl: read_species_tables(ion_dir / f"{lbl}.ion.xml") for lbl in set(labels)}
    pos = np.asarray(structure.positions)
    cell = np.asarray(structure.cell)
    naefs = 0.0
    dnaefs = 0.0
    inv_cell_rows = np.linalg.inv(cell)
    r_max = 2.0 * max(max(float(t["rc_vna"]), float(t["rc_chlocal"]) + 0.1) for t in tables.values())
    n_img = [int(np.ceil(r_max * float(np.linalg.norm(inv_cell_rows[ax])))) + 1 for ax in range(3)]
    shifts = list(itertools.product(*(range(-n, n + 1) for n in n_img)))
    for i, j in itertools.product(range(len(pos)), repeat=2):
        a, b = tables[labels[i]], tables[labels[j]]
        for shift in shifts:
            dist = float(np.linalg.norm(pos[j] + cell @ np.array(shift) - pos[i]))
            x = _j0(_K * dist)
            if dist <= float(a["rc_vna"]) + float(b["rc_vna"]):
                naefs += (
                    np.trapezoid(_K**4 * a["vt"] * b["vt"] * x, _K) / (2.0 * np.pi**2) / (16.0 * np.pi)
                )  # Ry
            if dist > 1e-7 and dist <= float(a["rc_chlocal"]) + float(b["rc_chlocal"]) + 0.2:
                corr_ha = float(a["z"]) * float(b["z"]) * (
                    (2.0 / np.pi) * np.trapezoid(a["f"] * b["f"] * x, _K) - 1.0 / dist
                )
                dnaefs -= 0.5 * corr_ha  # Ha; SIESTA adds point-charge minus smeared-charge energy
    return naefs * _RY_TO_HA, dnaefs
