User interface
==============

The entry points most users need. :class:`~jax_siesta.app.calculation.Calculation` gathers the inputs of a
single-point calculation (from an ``.fdf`` file or from Python objects), its ``run`` method executes the stages
documented in the other sections, and :class:`~jax_siesta.app.calculation.Results` exposes the outcome in eV and
Angstrom. The same package contains the ``jsiesta`` command, the text report, the writers of SIESTA-format
output files, and the helpers that find pseudopotentials and build basis sets.

``Calculation``, ``Results``, ``BandStructure`` and ``make_structure`` are also importable from the top-level
package: ``import jax_siesta as js; js.Calculation``.

.. code-block:: python

   import jax_siesta as js

   calc = js.Calculation.from_fdf("si.fdf")
   results = calc.run(log=print)

Usage is described in :doc:`../user_guide/python_interface` and :doc:`../user_guide/running`.

.. include:: ../_generated/api/interface.inc
