Quick start
===========

The repository ships ready-to-run inputs in ``examples/`` and four PseudoDojo pseudopotentials (H, C, O, Si) in
``examples/pseudos``.

The SIESTA way
--------------

A calculation needs an ``.fdf`` input file and one ``<Label>.psml`` pseudopotential per species.
``examples/si_bulk/si.fdf`` is bulk silicon:

.. literalinclude:: ../../examples/si_bulk/si.fdf
   :language: text

Run it:

.. code-block:: bash

   cd examples/si_bulk
   jsiesta si.fdf --pseudo-dir ../pseudos > si.out

``jsiesta < si.fdf > si.out`` works too, as with SIESTA, if ``Si.psml`` is in the same directory or in
``SIESTA_PS_PATH``. The output lists the basis, the SCF iterations and the results:

.. code-block:: text

           iscf     Eharris(eV)        E_KS(eV)     FreeEng(eV)     dDmax    dHmax(eV)    Ef(eV)
      scf:    1     -229.909946     -229.909344     -229.909344  0.017536    0.040663  -3.678421
      ...
      scf:    5     -229.909441     -229.909441     -229.909441  0.000001    0.000023  -3.701912

   SCF cycle converged after 5 iterations

   jsiesta: Final energy (eV):
   jsiesta:         Total =        -229.909441
   ...
   jsiesta: Atomic forces (eV/Ang):
   jsiesta: Stress tensor (static) (eV/Ang**3):
   jsiesta: Pressure (static):       2.816424 kBar  = 0.281642 GPa

and the directory now holds ``si.XV``, ``si.DM`` (restart), ``si.FA`` (forces), ``si.EIG`` (eigenvalues),
``si.bands`` (band structure, because the input has a ``BandLines`` block) and ``si.json`` (all results, machine
readable). Details: :doc:`../user_guide/running`.

The Python way
--------------

The same calculation from a script or notebook:

.. code-block:: python

   import jax_siesta as js

   calc = js.Calculation.from_fdf("examples/si_bulk/si.fdf", pseudo_dir="examples/pseudos")
   results = calc.run(log=print)          # log=print shows the SIESTA-style output

   print(results.energy_ev)               # total energy, eV
   print(results.forces_ev_ang)           # (n_atoms, 3), eV/Ang
   print(results.stress_ev_ang3)          # (3, 3), eV/Ang^3
   print(results.pressure_gpa)

   bands = results.band_structure([("L", (0.5, 0.5, 0.5)), ("Γ", (0, 0, 0)), ("X", (0.5, 0, 0.5))])
   print(bands.band_gap_ev)
   bands.plot()

No input file is needed either:

.. code-block:: python

   import numpy as np
   import jax_siesta as js

   a = 5.47
   silicon = js.make_structure(
       cell=a * np.array([[0, .5, .5], [.5, 0, .5], [.5, .5, 0]]),
       symbols=["Si", "Si"],
       positions=[[0, 0, 0], [.25, .25, .25]],
       fractional=True,
   )
   calc = js.Calculation.from_structure(
       silicon, pseudo_dir="examples/pseudos", basis="DZP", k_grid=(4, 4, 4),
       xc_functional="PBE", mesh_cutoff_ry=150, use_symmetry=True,
   )
   results = calc.run()

Variations reuse the calculation: ``calc.with_config(mesh_cutoff_ry=250).run()`` or
``calc.with_structure(other).run()``. Details: :doc:`../user_guide/python_interface`.

Next steps
----------

* :doc:`../tutorials/index`: worked examples with exercises.
* :doc:`../user_guide/fdf_reference`: which input keywords are understood.
* :doc:`../user_guide/siesta_differences`: what differs from SIESTA.
