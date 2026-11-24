# %% [markdown]
# # 8. Under the hood: the stages of a calculation
#
# `Calculation.run` is a short function that chains a few stages. This tutorial calls the stages one by one,
# looks at the objects they exchange, and uses that access for things the high-level interface does not offer.
# It is the bridge from the user guide to the API reference.
#
# **You will learn** the data model (structure, basis, sparsity pattern, sparse matrices), the internal units,
# how to assemble H(k) yourself, and how to restart from a density matrix.
#
# Run time: under a minute.

# %%
from dataclasses import replace
from pathlib import Path

import numpy as np
import scipy.linalg

import jax_siesta as js
from jax_siesta.app import build_species
from jax_siesta.integrals.two_center import TwoCenterEvaluator
from jax_siesta.io import read_dm, write_dm
from jax_siesta.observables import StaticObservableEvaluator
from jax_siesta.scf import build_monkhorst_pack_mesh, run_scf
from jax_siesta.species import PSMLBasisLoader, build_sparse_pattern

PSEUDOS = Path("../../examples/pseudos").resolve()
WORK = Path("work/08")
WORK.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Stage 0: structure and settings
#
# Internally everything is in Hartree atomic units: lengths in Bohr, energies in Hartree. The cell matrix has
# the lattice vectors as **columns**. `make_structure` converts from the usual rows in Ångström.

# %%
a = 5.47
structure = js.make_structure(
    a * np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]),
    ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]], fractional=True,
)
print("cell (Bohr), columns are lattice vectors:\n", np.round(np.asarray(structure.cell), 4))
print("species of each atom:", structure.species_indices, "labels:", structure.species_labels)

config = js.SCFConfig(xc_functional="PBE", mesh_cutoff_ry=150.0)  # all settings in one immutable object
print(config)

# %% [markdown]
# ## Stage 1: species and basis
#
# One `SpeciesData` per species (pseudopotential projectors and radial orbitals), then a `BasisData` that
# lists every orbital of the structure: which atom it sits on, its angular momentum, its radius.

# %%
species = {"Si": build_species("Si", PSEUDOS / "Si.psml", "DZP", xc="PBE")[0]}
basis = PSMLBasisLoader().build_basis(structure, species)
print("orbitals:", basis.n_orb, " projectors:", basis.n_kb, " valence electrons:", basis.total_valence_electrons)
print("atom of each orbital:", basis.orb_atom)
print("l of each orbital:   ", basis.orb_l)

# %% [markdown]
# ## Stage 2: the sparsity pattern
#
# Two orbitals interact only if their spheres overlap (or if both overlap the same pseudopotential
# projector). The `SparsePattern` lists these pairs as *edges*: orbital `src` in the home cell, orbital `dst`
# in the cell displaced by the lattice vector `cell_shifts[edge_cell_idx]`. **Every sparse matrix in jsiesta is
# an array over these edges**, with shape `(n_spin, n_edges)`.

# %%
pattern = build_sparse_pattern(structure, basis)
print("edges:", pattern.n_edges, " out of", basis.n_orb**2, "pairs per cell x", pattern.n_super_cells, "cells")
k = 12345
shift = np.asarray(pattern.cell_shifts)[int(pattern.edge_cell_idx[k])]
print(f"edge {k}: orbital {int(pattern.src_orb[k])} <-> orbital {int(pattern.dst_unit_orb[k])} in cell {shift}")

# %% [markdown]
# ## Stage 3: two-centre integrals
#
# Overlap, kinetic energy, the non-local pseudopotential and the neutral-atom terms depend only on pairs of
# atoms. They are computed once, by C++ kernels, with their derivatives for the forces.

# %%
two_center = TwoCenterEvaluator().evaluate(structure, basis, pattern, config)
overlap = np.asarray(two_center.overlap)
src, dst = np.asarray(pattern.src_orb), np.asarray(pattern.dst_unit_orb)
shifts = np.asarray(pattern.cell_shifts)[np.asarray(pattern.edge_cell_idx)]
onsite = (src == dst) & np.all(shifts == 0, axis=1)
print("overlap of each orbital with itself:", np.round(overlap[onsite][:6], 6))

# %% [markdown]
# The orbitals are normalised, so these should be 1. They are 1 to about one part in 10⁴: the radial functions
# are generated on a logarithmic grid and then tabulated on a uniform one, which is where the difference comes
# from. The generalised eigenproblem uses the overlap as computed, so this is consistent within a calculation.

# %% [markdown]
# ## Stage 4: the self-consistent cycle
#
# `run_scf` needs the k-points and returns the converged `SCFState`: Hamiltonian, density matrix,
# energy-density matrix, Fermi level. The `callback` receives one record per iteration.

# %%
kmesh = build_monkhorst_pack_mesh((4, 4, 4))
state, mesh_shape = run_scf(
    structure, basis, pattern, two_center, kmesh, config,
    callback=lambda rec: print(f"  iteration {int(rec['iteration'])}: dDmax = {rec['d_max']:.2e}"),
)
print("outcome:", state.outcome, " mesh:", mesh_shape)
print("Hamiltonian:", state.h_sparse.shape, " density matrix:", state.dm_out.shape)

# %% [markdown]
# A first use of the raw matrices: the number of electrons is the trace of the density matrix with the
# overlap, summed over all edges.

# %%
dm = np.asarray(state.dm_out)[0]
print("Tr[D S] =", float(np.sum(dm * overlap)), "(8 valence electrons)")

# %% [markdown]
# And the Mulliken population of each atom: the same sum restricted to the orbitals of that atom.

# %%
orb_atom = np.asarray(basis.orb_atom)
populations = np.bincount(orb_atom[src], weights=dm * overlap, minlength=structure.n_atoms)
print("Mulliken populations:", np.round(populations, 6))

# %% [markdown]
# ## Stage 5: energy, forces and stress
#
# The evaluator recomputes the energy at the output density matrix and differentiates it. The result is an
# `Observables` object in atomic units; `Results` of the high-level interface is this object plus unit
# conversions.

# %%
obs = StaticObservableEvaluator().evaluate(
    structure, basis, pattern, two_center, kmesh, state, config, mesh_shape=mesh_shape
)
print("total energy:", float(obs.total_energy_ha), "Ha =", float(obs.total_energy_ha) * js.HARTREE_TO_EV, "eV")
print("stress (Ha/Bohr^3):\n", np.asarray(obs.stress_ha_per_bohr3))

high_level = js.Calculation(structure, species, config, k_grid=(4, 4, 4)).run()
print("difference to Calculation.run (eV):", high_level.energy_ev - float(obs.total_energy_ha) * js.HARTREE_TO_EV)

# %% [markdown]
# ## Building H(k) by hand
#
# The Bloch Hamiltonian at a k-point is the sum over edges of the matrix element times a phase,
# $H_{\mu\nu}(\mathbf k) = \sum_{\mathbf T} H_{\mu\nu}(\mathbf T)\, e^{2\pi i\, \mathbf k \cdot \mathbf T}$,
# with $\mathbf k$ in units of the reciprocal lattice vectors and $\mathbf T$ the integer cell shift of the edge.
# Solving the generalised eigenproblem with SciPy reproduces the band energies of jsiesta.

# %%
k_point = np.array([0.5, 0.0, 0.5])  # the X point
phase = np.exp(2j * np.pi * (shifts @ k_point))
h_edges = np.asarray(state.h_sparse)[0]
h_k = np.zeros((basis.n_orb, basis.n_orb), dtype=complex)
s_k = np.zeros_like(h_k)
np.add.at(h_k, (src, dst), h_edges * phase)
np.add.at(s_k, (src, dst), overlap * phase)
print("H(k) Hermitian:", np.allclose(h_k, h_k.conj().T), "  S(k) positive definite:", np.linalg.eigvalsh(s_k).min() > 0)

mine = scipy.linalg.eigh(h_k, s_k, eigvals_only=True) * js.HARTREE_TO_EV
theirs = high_level.eigenvalues_ev([k_point])[0, 0]
print("lowest eigenvalues at X (eV):", np.round(mine[:5], 4))
print("largest difference to Results.eigenvalues_ev:", np.abs(mine - theirs).max())

# %% [markdown]
# With the dense matrices in hand you can go further than the interface does: project bands on orbitals,
# compute optical matrix elements, or export a tight-binding model.
#
# ## Restarting from a density matrix
#
# `write_dm` and `read_dm` use SIESTA's `.DM` format. Restarting from the converged density matrix, the
# cycle needs only the minimum number of iterations.

# %%
write_dm(WORK / "si.DM", pattern, state.dm_out, config.nspin)
dm0 = read_dm(WORK / "si.DM", pattern, config.nspin)
history = []
restarted, _ = run_scf(structure, basis, pattern, two_center, kmesh, config, initial_dm=dm0, callback=history.append)
print("iterations after restart:", len(history), " first dDmax:", f"{history[0]['d_max']:.1e}")

# %% [markdown]
# The same density matrix is a good starting point for a slightly different calculation, for example a tighter
# electronic temperature:

# %%
cold = replace(config, electronic_temp_ha=config.electronic_temp_ha / 10.0)
history = []
run_scf(structure, basis, pattern, two_center, kmesh, cold, initial_dm=dm0, callback=history.append)
print("iterations at one tenth of the temperature, from the stored density matrix:", len(history))

# %% [markdown]
# ## Where to go from here
#
# Each stage has its section in the API reference, with an introduction and the documentation of every
# function: *Species*, *Two-centre integrals*, *Real-space grid*, *Exchange-correlation*, *Symmetry*,
# *Self-consistent field*, *Energies, forces, stress and bands*, *Input and output files*. The *Theory*
# section gives the equations and the conventions (units, spin channels, signs).
#
# ## Exercises
#
# 1. **Mulliken charges of water.** Run the water molecule of tutorial 5 and compute the Mulliken population
#    of each atom from `results.scf_state.dm_out`, `results.overlap` and `results.pattern`. How many electrons
#    does the oxygen gain? *(Solution in the Solutions notebook.)*
# 2. Compute the band-structure energy $\sum_{\mu\nu} D_{\mu\nu} H_{\mu\nu}$ as a sum over edges and compare it
#    with the sum of the occupied eigenvalues over the k-points of the SCF mesh (use
#    `high_level.eigenvalues_ev()` and `kmesh.weights`).
# 3. Symmetry: build the calculation with `build_native_symmetry` (API reference, *Symmetry*) instead of
#    `build_sparse_pattern`, and compare the number of edges of the symmetry-closed pattern and the number of
#    k-points.
# 4. Collinear spin: for the O₂ molecule of tutorial 6 the matrices have two spin channels. Compute the
#    magnetic moment of each atom from the difference of the Mulliken populations of the two channels and
#    compare with `results.atomic_spin_moments`.
