# Quantum ESPRESSO rVV10 reference (carbon bilayer)

Frozen reference for `tests/test_dispersion.py::test_rvv10_matches_quantum_espresso`.

- Code: PWscf v7.5 (develop), `input_dft='rvv10'`, `C.UPF` (norm-conserving, no NLCC) from `q-e/pseudo/`.
- Inputs: `scf.in` (SCF, `conv_thr=1e-9`), `pp.in` (`pp.x`, `plot_num=0`).
- `c_bilayer_rho.npz`: converged valence density on the 18×18×100 FFT grid (e/Bohr³, read from the
  full-precision `filplot`, array index order `[i1, i2, i3]`) and the lattice as columns (Bohr).
- Reference: last `Non-local correlation energy` printed by `pw.x` = **0.070718182102919 Ry**
  (QE's E_nl includes the β∫n term).

To regenerate: run `pw.x -in scf.in`, then `pp.x -in pp.in`, and convert `rho.pp`.
