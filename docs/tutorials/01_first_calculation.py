# %% [markdown]
# # 1. A first calculation: bulk silicon
#
# This tutorial runs the same calculation twice: first the way SIESTA is used, with an input file given to a
# command, then from Python. Along the way it explains what the output means.
#
# **You will learn** how to write a minimal input, run `jsiesta`, read the output and the files it writes, and
# get the same numbers as Python objects.
#
# **Requirements:** jsiesta installed and its library built (see *Getting started*). The pseudopotentials come
# from `examples/pseudos` of the repository. Run time: under a minute.

# %%
from pathlib import Path
import json
import subprocess
import sys

import numpy as np

import jax_siesta as js

PSEUDOS = Path("../../examples/pseudos").resolve()  # PseudoDojo PBE pseudopotentials shipped with the repository
WORK = Path("work/01")
WORK.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## The input file
#
# A SIESTA input file (FDF format) is a list of `Label value` lines and `%block` sections. This one describes
# silicon in the diamond structure: two atoms in a face-centred cubic cell.

# %%
fdf = """\
SystemName   Bulk silicon
SystemLabel  si                      # prefix of the output files

%block ChemicalSpeciesLabel
  1  14  Si                          # species index, atomic number, label -> pseudopotential Si.psml
%endblock ChemicalSpeciesLabel

LatticeConstant  5.47 Ang
%block LatticeVectors                # rows, in units of the lattice constant
  0.0  0.5  0.5
  0.5  0.0  0.5
  0.5  0.5  0.0
%endblock LatticeVectors

AtomicCoordinatesFormat  Fractional
%block AtomicCoordinatesAndAtomicSpecies
  0.00  0.00  0.00  1
  0.25  0.25  0.25  1
%endblock AtomicCoordinatesAndAtomicSpecies

PAO.BasisSize  DZP                   # double-zeta plus polarisation
XC.Functional  GGA
XC.Authors     PBE
MeshCutoff     150 Ry                # fineness of the real-space mesh

%block kgrid.MonkhorstPack           # 4 x 4 x 4 k-points
  4  0  0  0.0
  0  4  0  0.0
  0  0  4  0.0
%endblock kgrid.MonkhorstPack
"""
(WORK / "si.fdf").write_text(fdf)

# %% [markdown]
# ## Running the command
#
# In a terminal you would type
#
# ```bash
# jsiesta si.fdf --pseudo-dir ../../examples/pseudos > si.out
# ```
#
# Here we start the same command from Python so that the tutorial is self-contained.

# %%
run = subprocess.run(
    [sys.executable, "-m", "jax_siesta", "si.fdf", "--pseudo-dir", str(PSEUDOS)],
    cwd=WORK, capture_output=True, text=True,
)
(WORK / "si.out").write_text(run.stdout)
print(run.stdout)
print("exit status:", run.returncode, run.stderr)

# %% [markdown]
# Reading the output from the top:
#
# * **Species and basis.** For silicon the DZP basis has two radial functions for the 3s shell, two for 3p and
#   one d polarisation function: 2 + 2x3 + 5 = 13 orbitals per atom. Each has a cut-off radius beyond which it
#   is exactly zero. That is what makes the Hamiltonian sparse.
# * **System.** 26 orbitals, 8 valence electrons, 64 k-points, and the mesh chosen for the requested cutoff.
# * **SCF table.** Each line is one iteration. `dDmax` and `dHmax` are the largest changes of the density
#   matrix and of the Hamiltonian; the cycle stops when both are below their tolerances.
# * **Final energy.** `Total` is the Kohn-Sham energy. `FreeEng` includes the electronic entropy term and is
#   the quantity the forces are derivatives of; for an insulator they coincide.
# * **Forces and stress.** By symmetry the forces vanish here. The stress is small because 5.47 Ang is close to
#   the equilibrium lattice constant of this functional and basis.
#
# The exit status is 0 for a converged run.

# %% [markdown]
# ## The files
#
# All files are named after `SystemLabel`.

# %%
for f in sorted(WORK.iterdir()):
    print(f"{f.name:12s} {f.stat().st_size:8d} bytes")

# %% [markdown]
# `si.XV` (geometry) and `si.DM` (density matrix) are restart files in SIESTA's formats, `si.FA` has the forces,
# `si.EIG` the eigenvalues, and `si.json` everything in machine-readable form:

# %%
summary = json.loads((WORK / "si.json").read_text())
print(summary["energies"]["total"], summary["units"]["energy"])
print(summary["n_scf_iterations"], "iterations; pressure", round(summary["pressure"], 3), summary["units"]["pressure"])

# %% [markdown]
# ## The same calculation from Python
#
# `Calculation.from_fdf` reads the input file and builds the basis; `run` does the work and returns a `Results`
# object. Passing `log=print` would print the same report as the command.

# %%
calc = js.Calculation.from_fdf(WORK / "si.fdf", pseudo_dir=PSEUDOS)
results = calc.run()

print("converged:        ", results.converged, "in", results.n_iterations, "iterations")
print("total energy (eV):", results.energy_ev)
print("Fermi level (eV): ", results.fermi_level_ev)
print("pressure (GPa):   ", results.pressure_gpa)
print("largest force (eV/Ang):", np.abs(results.forces_ev_ang).max())

# %% [markdown]
# The energy is the one the command printed. The individual terms:

# %%
for name, value in results.energies_ev.items():
    print(f"{name:14s} {value:16.6f} eV")

# %% [markdown]
# ## Changing a setting
#
# `Calculation` objects are immutable; `with_config` returns a modified copy. Using the crystal symmetry gives
# the same energy with 8 k-points instead of 64:

# %%
sym = calc.with_config(use_symmetry=True).run()
print("k-points:", results.kmesh.n_k, "->", sym.kmesh.n_k)
print("energy difference (eV):", sym.energy_ev - results.energy_ev)
print("time for the SCF (s):", round(results.timings["scf"], 2), "->", round(sym.timings["scf"], 2))

# %% [markdown]
# ## Checking an input without running it
#
# `jsiesta input.fdf --check` reads the input, builds the basis and stops. It also shows the most useful
# warning of the program: keywords that were **not used**. Below, `MeshCutof` is misspelt and `MD.Steps` asks
# for a relaxation that jsiesta does not do.

# %%
(WORK / "typo.fdf").write_text(fdf + "MeshCutof 300 Ry\nMD.TypeOfRun CG\nMD.Steps 50\n")
check = subprocess.run(
    [sys.executable, "-m", "jax_siesta", "typo.fdf", "--pseudo-dir", str(PSEUDOS), "--check"],
    cwd=WORK, capture_output=True, text=True,
)
print("\n".join(line for line in check.stdout.splitlines() if "WARNING" in line or "check" in line))

# %% [markdown]
# ## Exercises
#
# 1. Change `PAO.BasisSize` to `SZ` and run again. How many orbitals are there now, and how do the total energy
#    and the pressure change? (The total energy must go *up*: a smaller basis is a more constrained
#    variational space.)
# 2. Displace the second atom to `0.25 0.25 0.27` and look at the forces. In which direction do they point, and
#    is their sum zero?
# 3. Add `JSiesta.UseSymmetry T` to the input with the displaced atom. How many k-points are used now, and why
#    is it more than for the perfect crystal?
# 4. Run with `MeshCutoff 60 Ry`. Which results change most? Tutorial 2 looks at this systematically.
#
# Worked solutions to selected exercises are in the *Solutions* notebook.
