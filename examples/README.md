# Examples

Ready-to-run SIESTA-style inputs. From this directory:

```bash
cd si_bulk && jsiesta si.fdf --pseudo-dir ../pseudos > si.out
```

| Directory | System | Shows |
|---|---|---|
| `si_bulk/` | Bulk silicon, PBE, DZP, 4x4x4 k-points | A periodic solid, a band structure (`BandLines`) |
| `h2o/` | Water molecule in a box | A molecule at the Gamma point, forces |
| `o2_spin/` | Oxygen molecule, triplet | `Spin polarized` and `DM.InitSpin` |
| `pseudos/` | PSML pseudopotentials for H, C, O, Si | See `pseudos/README.md` for their origin and licence |

Each run takes a few seconds and writes `<label>.XV`, `.DM`, `.FA`, `.EIG`, `.json` (and `.bands` for silicon)
in the directory where it is started. The tutorials in `docs/tutorials` build on these systems.
