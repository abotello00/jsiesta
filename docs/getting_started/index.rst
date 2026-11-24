Getting started
===============

This section takes you from a fresh clone to a first result. jsiesta has two parts: a Python package
(``jax_siesta``) and a compiled library of kernels (``libjsiesta_ffi``) that the package loads at run time. Both
are built from the repository; there is no binary distribution yet.

1. :doc:`installation`: create the environment, build the library, check that it works.
2. :doc:`quickstart`: run bulk silicon with the ``jsiesta`` command and from Python.

After that, the :doc:`../tutorials/index` cover typical calculations step by step, and the
:doc:`../user_guide/index` is the reference for inputs and outputs.

.. toctree::
   :maxdepth: 1

   installation
   quickstart
