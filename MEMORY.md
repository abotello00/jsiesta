# jsiesta Live Project Memory

- **Repository:** `jsiesta`, branch `main`, version 0.1.0; consolidated release of the development repository
  (`jsiesta_v0.1`, D-001 to D-044) in one commit (D-045). Authorship: `AUTHORS_CONTRIBUTORS.md`.
- **Current phase:** v0.1 complete: Phases 0–3 of `docs/design/IMPLEMENTATION_PLAN.md`, the review remediation and
  the GPU performance work (D-031 to D-044). Open items are listed under *Open issues*.
- **Last updated:** 2026-11-24 (D-046 to D-048 fixed costs, D-049 and D-050 basis optimisation, D-051 SIESTA restart files, D-052 projection table and test memory, D-053 user interface and documentation)
- **Status rule:** every "validated" entry below names the test that checks it against an *external* reference.
  Self-consistency tests alone do not count. Run `make check` (ruff, mypy, Doxygen + Sphinx -W, KG rebuild, pytest
  with JUnit XML in `reports/`); `make test-refs` makes missing SIESTA references fatal.

## User interface (D-053)

- `jsiesta in.fdf [--pseudo-dir DIR] [--check]` (also `python -m jax_siesta`, or the input on standard input) runs a
  SIESTA input with `<Label>.psml` files; Python: `jax_siesta.Calculation.from_fdf(...).run()` -> `Results` (eV, Ang).
  Code in `jax_siesta/app`; keywords in `docs/user_guide/fdf_reference.rst`; unused labels are reported.
- The basis is generated from the PSML file by default (Tier B generator); `User.Basis T` reads SIESTA's `.ion.xml`.
  Hydrogen needs `PAO.SplitNormH 0.5`.
- Documentation site: `docs/` (Getting started, User guide, Tutorials, API by section, Theory, Development).
  Tutorials: `docs/tutorials/*.py` + executed `*.ipynb` (`make tutorials`, about 12 minutes on the laptop).
  Examples and PseudoDojo PSML files (H, C, O, Si): `examples/`.

## Environment

- `.venv` (Python 3.11) from `requirements-dev.txt`: jax/jaxlib 0.4.38, numpy 2.2.6, scipy 1.14.1, spglib 2.8.0,
  mcp 1.27, sphinx 8.2.3, ruff 0.15.7, mypy 2.4.1, dftd3 1.7.0 (optional extra `[dispersion]`), myst-nb 1.4.0,
  jupytext 1.19.7, matplotlib 3.11.2 (documentation notebooks). The package is installed in editable mode (`jsiesta` script).
- `make ffi` builds `cpp/build/libjsiesta_ffi` with static Libxc from `/opt/etsf` (`JSIESTA_USE_LIBXC=ON`).
- External references: keys in `tests/external_refs.toml` (empty; set them in `tests/external_refs.local.toml` or `JSIESTA_<KEY>`):
  standard SIESTA 5.4.2, `siesta_spglib_full_impl`, `pr_mgga` SIESTA, SIESTA `Tests/Pseudos`, `pr_mgga` Tests, the SIESTA documentation index,
  scf_params Co pilot. Quantum ESPRESSO 7.5 was used once to freeze `tests/fixtures/qe_rvv10`.

## Validation status (external references)

- **Si bulk components, forces, stress (std + spglib SIESTA):** validated — `test_siesta_bulk_and_symmetry_parity` (matched meshes)
- **Absolute total / free energy:** validated — `test_siesta_total_and_free_energy_parity` (Si, 3 geometries; ≤ 2 meV/cell)
- **Co hcp collinear (nspin = 2, NLCC, semicore):** validated — `test_siesta_co_hcp_collinear_symmetry_parity` (components ≤ 2e-2 eV, F ≤ 1e-4 eV/Å, σ within 3e-3 eV/Å³; the 2.8 % σ_zz gap is the neutral-atom pair term: 57 % SIESTA's table error, 43 % a 0.075 % definitional difference, D-025)
- **Neutral-atom pair energy:** vs the exact value of SIESTA's own formula — `tests/test_na_pair_exact.py` (D-025): PseudoDojo H2O/Si 0.05–0.07 % (SIESTA's printed `Ena` 0.3–2 % off, smooth table error); pilot Co −0.075 % (jsiesta, definitional) and +0.097 % (SIESTA), opposite directions
- **Analytic stress terms = derivatives of their energies (fixed D, E):** validated — `tests/test_stress_components.py` (Si nspin 1; Co nspin 2 GGA and LDA with NLCC; ≤ 1e-9 Ha/Bohr³)
- **meta-GGA / NC / SOC vs pr_mgga:** validated (unchanged tolerances) — `test_pr_mgga_si_bulk_noncol_and_soc_parity`
- **DFT-D3(BJ):** validated — `tests/test_dispersion.py` (simple-dftd3 identity, FD, SIESTA energy/forces/stress)
- **rVV10:** validated — `tests/test_dispersion.py` (QE E_nl to 3.5e-11 Ry, analytic kernel, FD stress)
- **Forces/stress = derivatives of the energy:** validated — FD tests in `test_phase1_scf_and_forces.py`, `test_mgga_soc.py`
- **GPU kernels:** validated against the CPU path (not against SIESTA) — `tests/test_phase3_cuda.py`, 24 of 24 on an A100 with Libxc (job 29610921): orbital–grid kernels and their adjoints on the cached plan (D-036 to D-038), device LDA/GGA XC matching Libxc to 1e-14 (D-039), plans resident, streamed, and streamed without pinning (D-042 to D-044)
- **Tier B PAO generator:** partially — SZ/DZ/DZP radii and trends vs SIESTA (`test_basis_enthalpy.py`); absolute energies not compared
- **Basis-enthalpy optimiser:** checked against SIESTA (D-040: same `PAO.Basis` in both codes, Si DZ; line minima within 0.05 Bohr, enthalpy curvatures within 3 %, 28 meV constant offset); pressure unit fixed (was 13.6x too small)
- **Basis-optimisation speed (D-049, D-050):** SiC DZP, 3 structures, per evaluation: 1.32 s on one A100, SIESTA 23.8 s
  (same `PAO.Basis`); symmetry set-up and radial tables reused across evaluations and structures
- **Polarisation orbitals (D-050):** SIESTA convention (parent's cut-off and confinement, SIESTA's source term, `P`
  export); overlap with SIESTA's 1.00000 — `test_exported_polarised_basis_matches_siesta`
- **Broyden / ADIIS mixers:** validated against model problems and Co hcp endpoint/trajectory test — `tests/test_mixers.py`, `test_si_bulk_and_co_hcp_spin_modes_and_mixer_endpoint_agreement` (D-022); no external reference exists; ADIIS is slower than Tikhonov-Pulay on all benchmark systems; GR-Pulay not implemented
- **Harris energy:** validated against an independent recomputation, not against SIESTA — `tests/test_harris.py` (D-023); exact under DM mixing, SIESTA convention (non-variational) under Hamiltonian mixing
- **Phase 0 matrix parity (H2O, graphene, triclinic Si, C bilayer, Co AFM) with PseudoDojo SR PBE:** validated with per-case tolerances — `test_phase0_matrix_parity_against_siesta` (energies ≤ 5 meV except triclinic Si 52 meV; forces ≤ 3e-4 eV/Å except H2O 0.06 and triclinic Si 0.1; the H2O and triclinic-Si force differences are SIESTA's table error in its neutral-atom pair term, D-025)
- **SIESTA mesh criterion:** validated — `test_siesta_mesh_selection_matches_siesta` (15 cells, `criterion="siesta"`, D-020); the default since D-027 (`"jsiesta"` is the legacy option)
- **Fully relativistic PSML below nspin 8:** validated — `test_fully_relativistic_psml_below_nspin8_matches_siesta` (H2O, triclinic Si; same projector set as SIESTA, D-028)
- **Frozen spin axis (non-collinear GGA derivative):** bounded — `test_noncollinear_xc_derivative_with_frozen_spin_axis` (LDA exact, PBE <= 7e-5 relative on a spin spiral; D-026)
- **SIESTA restart files (`.XV`, `.DM`, `.nc`; D-051):** partially — formats byte-identical to SIESTA-written files (`test_siesta_written_dm_and_xv_are_reproduced_byte_for_byte`, needs a SIESTA source tree); spin conventions against a transcription of `diag3k`, round trips and warm starts are self-consistency tests (`tests/test_siesta_files.py`); the exchange with a running SIESTA (`test_siesta_reads_jsiesta_files_and_jsiesta_reads_siesta_files`) has not been run yet
- **User interface (`jax_siesta.app`, D-053):** self-consistency and identity tests only — `tests/test_app_interface.py`
  (eigenvalues vs dense SciPy solution 1e-8 eV, electron count, force vs finite difference 1.1e-5 eV/Ang at 200 Ry,
  file round trips, restart, exit codes). Not validated against SIESTA: `test_same_fdf_input_in_siesta_and_jsiesta`
  (needs `siesta_bin`) has never run; `.FA`/`.EIG`/`.bands` layouts follow the SIESTA sources but were not compared
  with SIESTA-written files
- **KG verification status:** from JUnit results since D-019 (`make kg-results`); `resume_session_context` separates verified (external reference passed), self-consistent only, failing, skipped and never run

## Upstream issues found (reported here, not fixed upstream)

- SIESTA 5.4.2 `grimme_dispersion_m.F90`: `DFTD3.UseXCDefaults T` overwrites functional parameters with generic
  defaults; D3 stress added with the wrong sign; bundled s-dftd3 1.1.0 gradient ~0.3 % off its energy (D-014).
- `siesta_spglib_full_impl`: forces of symmetric structures are not symmetrised (1e-3 eV/Å residual in hcp Co).

## Performance (A100 vs SIESTA, bulk Si, PBE; `docs/design/known_bottlenecks.md` Section 9)

- SCF iteration 2.4–7.7x faster than SIESTA on 18 ranks and 1.5–3.0x faster than on 192 ranks (8–512 atoms).
- Forces + stress 0.31–6.5 s (8–512 atoms): faster than SIESTA on 18 ranks from 64 atoms up, 2–4x slower than 192.
- r2SCAN forces + stress at 512 atoms 14.8 s with the Hessian plan streamed from host memory (D-043, D-044).
- Set-up and compilation dominate small cells: 7.7 s + 19 s at 64 atoms against 0.5 s for SIESTA's set-up (A100,
  before D-046/D-047). After them (job 29629429, 6 SCF iterations + forces): Si 64 PBE 39.5 s -> 17.3 s (9.7 s with
  a warm cache), a second geometry 19.7 s -> 6.7 s; Si 512 PBE 182 s -> 98 s, second geometry 139 s -> 65 s.

## Open issues

- **Basis optimisation:** SCF (0.48 s) and host two-centre integrals (0.42 s) are most of a GPU evaluation; a new
  neighbour set costs a recompilation; large symmetric cells spend tens of seconds in the projection table.
- **Set-up and compilation cost:** reduced by D-046 (no recompilation per geometry, far fewer first-call
  compilations, opt-in persistent cache), D-047 (threaded host builds) and D-048 (one grid plan per geometry, no
  serial zero fill), measured on the A100: a second Si 512 geometry costs 52.6 s. The sparse pattern is still NumPy
  and the largest item (17 s), then the host two-centre set-up and force terms (11 s); 6 SCF iterations of real work
  are 12 s.
- **Host-side work left in the forces:** two-centre derivatives (0.32 s at 64 atoms) are now the largest stage.
- **Meta-GGA XC on the host:** Libxc meta-GGA still runs on the host, now threaded (D-047).
- **Plan limits:** `int32` value offsets cap a grid plan at 2^31 orbital values (about 2800 Si atoms); the host plan
  must fit in host memory.
- **SIESTA re-measurement:** SIESTA timings come from shared nodes with `DirectPhi .false.`; an exclusive-node run
  with ELPA is still open.
- **Restart-file exchange with SIESTA not measured:** `test_siesta_reads_jsiesta_files_and_jsiesta_reads_siesta_files`
  needs `siesta_bin` and `pseudodojo_sr_pbe_dir`; it has never run. `nspin` 4 and 8 file conventions are checked only
  against a transcription of `diag3k`.
- **A100 reference replay:** the SIESTA-parity tests have not been replayed on the GPU against a reference cache
  (`make refs-cache`).

- **Native-basis runs vs SIESTA not measured (D-053):** the same `.fdf` + PSML in both codes
  (`test_same_fdf_input_in_siesta_and_jsiesta`), and the SIESTA-format text files against SIESTA's own.
- **Mesh noise with ONCV pseudopotentials (D-053):** forces move by 0.02 eV/Ang between 150 and 200 Ry for displaced
  Si; SIESTA's remedy (grid-cell sampling) is not implemented.
- **Automatic polarisation shell:** the generator polarises the highest-`l` valence shell; SIESTA's choice for
  elements with d or semicore valence states may differ (not compared).
- **Generated orbitals are normalised to 1 + 1e-4** after tabulation on the uniform grid (D-053).

## Next steps

1. Cut set-up and compilation cost (two-centre tables on the device, persistent compilation cache).
2. Meta-GGA XC on the device; `vmat` without atomics.
3. **Not done by choice:** adopting SIESTA's neutral-atom pair formula (D-025); GR-Pulay; a per-iteration Harris
   comparison with SIESTA (needs SIESTA's diagonal atomic initial density matrix).
