# %% [markdown]
# # 6. Spin: the oxygen molecule
#
# The ground state of O₂ is a spin triplet: two unpaired electrons. A calculation without spin polarisation
# cannot describe it. This tutorial runs O₂ without spin, with collinear spin and with non-collinear spin, and
# shows the role of the starting magnetic moments.
#
# **You will learn** to run spin-polarised calculations from an input file and from Python, to read spin
# moments, and to check that a magnetic solution is the one you intended.
#
# Run time: about one minute.

# %%
from pathlib import Path

import numpy as np

import jax_siesta as js

PSEUDOS = Path("../../examples/pseudos").resolve()
WORK = Path("work/06")
WORK.mkdir(parents=True, exist_ok=True)

BOX, BOND = 7.0, 1.21  # Ang


def o2(moments):
    z = BOX / 2.0
    pos = [[z, z, z - BOND / 2.0], [z, z, z + BOND / 2.0]]
    return js.make_structure(np.eye(3) * BOX, ["O", "O"], pos, initial_moments=moments)


settings = dict(basis="DZP", xc_functional="PBE", mesh_cutoff_ry=150.0)

# %% [markdown]
# ## Without spin polarisation
#
# `nspin=1` forces every orbital to hold a spin-up and a spin-down electron. The two electrons that should be
# unpaired then share the two degenerate π* orbitals with half occupations.

# %%
unpolarised = js.Calculation.from_structure(o2(None), PSEUDOS, nspin=1, **settings).run()
print("energy (eV):", round(unpolarised.energy_ev, 4))

# %% [markdown]
# ## Collinear spin
#
# With `nspin=2` (the keyword `Spin polarized` in an input file) spin-up and spin-down electrons have separate
# densities. The self-consistent cycle needs a starting point that breaks the symmetry between them: the
# initial moments, one unpaired electron per atom here.

# %%
triplet_calc = js.Calculation.from_structure(o2([1.0, 1.0]), PSEUDOS, nspin=2, **settings)
triplet = triplet_calc.run()
print("energy (eV):       ", round(triplet.energy_ev, 4))
print("total spin moment: ", np.round(triplet.spin_moment, 4), "Bohr magnetons")
print("moment per atom:   ", np.round(triplet.atomic_spin_moments[:, 2], 4))
print("gain from spin polarisation (eV):", round(unpolarised.energy_ev - triplet.energy_ev, 3))

# %% [markdown]
# The total moment is 2 μ_B, one per atom, and the energy is 1.2 eV lower than without spin: the triplet is
# the ground state.
#
# ## The starting moments matter
#
# The cycle converges to a solution close to where it starts. Starting from zero moments, a spin-polarised
# calculation stays unpolarised, because nothing breaks the symmetry between the two spins. Starting with
# opposite moments on the two atoms also ends unpolarised here: the molecule has no antiferromagnetic solution,
# and the cycle falls back to the symmetric one, 1.2 eV above the ground state.

# %%
for label, moments in (("zero moments", [0.0, 0.0]), ("antiparallel moments", [1.0, -1.0])):
    r = triplet_calc.with_structure(o2(moments)).run()
    print(f"{label:22s} E − E(triplet) = {r.energy_ev - triplet.energy_ev:7.4f} eV   "
          f"total moment {r.spin_moment[2]:6.3f}   per atom {np.round(r.atomic_spin_moments[:, 2], 3)}")

# %% [markdown]
# Always check the final moments, and when in doubt try several starting configurations and compare energies.
#
# ## The same calculation with an input file
#
# The keywords are those of SIESTA: `Spin polarized` and a `DM.InitSpin` block with one line per atom.

# %%
(WORK / "o2.fdf").write_text(f"""\
SystemLabel  o2
%block ChemicalSpeciesLabel
  1  8  O
%endblock ChemicalSpeciesLabel
LatticeConstant  {BOX} Ang
%block LatticeVectors
  1.0  0.0  0.0
  0.0  1.0  0.0
  0.0  0.0  1.0
%endblock LatticeVectors
AtomicCoordinatesFormat  Ang
%block AtomicCoordinatesAndAtomicSpecies
  {BOX / 2}  {BOX / 2}  {BOX / 2 - BOND / 2}  1
  {BOX / 2}  {BOX / 2}  {BOX / 2 + BOND / 2}  1
%endblock AtomicCoordinatesAndAtomicSpecies
Spin  polarized
%block DM.InitSpin
  1  +1.0
  2  +1.0
%endblock DM.InitSpin
PAO.BasisSize  DZP
XC.Functional  GGA
XC.Authors     PBE
MeshCutoff     150 Ry
""")
from_file = js.Calculation.from_fdf(WORK / "o2.fdf", pseudo_dir=PSEUDOS).run()
print("energy difference to the Python set-up (eV):", from_file.energy_ev - triplet.energy_ev)

# %% [markdown]
# ## Non-collinear spin
#
# With `nspin=4` (`Spin non-collinear`) the magnetisation is a vector field that may point in different
# directions in different places. Without spin-orbit coupling the energy cannot depend on the *global*
# direction of the moments, which gives a simple test: start the moments along x instead of z.

# %%
along_x = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
noncollinear = js.Calculation.from_structure(o2(along_x), PSEUDOS, nspin=4, **settings).run()
print("total spin moment (x, y, z):", np.round(noncollinear.spin_moment, 4))
print("E(non-collinear, x) − E(collinear, z) (eV):", noncollinear.energy_ev - triplet.energy_ev)

# %% [markdown]
# The moment stays along x and the energy equals the collinear one, as it must. Spin-orbit coupling
# (`nspin=8`, `Spin spin-orbit`) ties the moments to the lattice; it needs fully relativistic pseudopotentials,
# which are not among the files shipped with the examples.
#
# ## Exercises
#
# 1. **Binding energy.** The binding energy of O₂ is $2E(\mathrm{O}) - E(\mathrm{O_2})$. Compute the oxygen
#    atom in the same box with two unpaired electrons. PBE is known to overbind O₂ (about 6.2 eV against 5.2 eV
#    in experiment). What do you get? *(Solution in the Solutions notebook.)*
# 2. Stretch the triplet molecule: compute the energy for bond lengths from 1.1 to 1.4 Å and find the
#    equilibrium bond length (experiment: 1.208 Å).
# 3. Start the triplet calculation with moments of 0.1 and of 3.0 per atom. Does the result depend on the
#    size of the starting moments, or only on their presence and relative sign?
# 4. Run the triplet with `electronic_temp_ha` ten times larger. What happens to the moment and to the
#    difference between `energy_ev` and `free_energy_ev`?
