Installation
============

Requirements
------------

.. list-table::
   :widths: 25 75

   * - Python
     - 3.11 or newer
   * - C++ compiler and CMake
     - C++17 (GCC or Clang), CMake 3.20 or newer
   * - Libxc
     - Versions 5.1 to 7.x (tested with 6.2.2, 7.0.0 and 7.1.2); needed for meta-GGA and hybrid functionals and
       used for all functionals when present.
   * - `uv <https://docs.astral.sh/uv/>`_
     - Used by ``make venv`` to create the environment (plain ``pip`` works too, see below)
   * - Optional
     - CUDA 12 for the GPU build; ``netCDF4`` for ``.nc`` files; ``dftd3`` for DFT-D3; ``matplotlib`` for plots

Steps
-----

.. code-block:: bash

   git clone https://github.com/abotello00/jsiesta.git
   cd jsiesta

   make venv                              # .venv with the pinned dependencies, and jsiesta itself (editable)
   LIBXC_ROOT=/path/to/libxc make ffi     # builds cpp/build/libjsiesta_ffi.so

   source .venv/bin/activate
   jsiesta --version

``make venv`` installs the package in editable mode, which puts the ``jsiesta`` command in ``.venv/bin``.
Without ``uv``:

.. code-block:: bash

   python3.11 -m venv .venv && source .venv/bin/activate
   pip install -r requirements-dev.txt
   pip install -e .

Without Libxc, configure with ``cmake -S cpp -B cpp/build -DJSIESTA_USE_LIBXC=OFF``: the built-in LDA and GGA
functionals work, meta-GGAs do not.

Check the installation
----------------------

.. code-block:: bash

   cd examples/si_bulk
   jsiesta si.fdf --pseudo-dir ../pseudos

This should print an SCF table that converges in a few iterations and end with ``Job completed`` after a few
seconds. The first run of a new system compiles its functions; later runs of the same shapes in one Python
process reuse them.

GPU build
---------

.. code-block:: bash

   VIRTUAL_ENV=$PWD/.venv cmake -S cpp -B cpp/build -DJSIESTA_USE_LIBXC=ON -DJSIESTA_ENABLE_CUDA=ON
   cmake --build cpp/build -j

The GPU build runs the same algorithms as the CPU build. It also needs a CUDA-enabled ``jaxlib``. See
:doc:`../user_guide/performance`.

Environment variables
---------------------

.. list-table::
   :widths: 32 68

   * - ``JSIESTA_FFI_LIB``
     - Path of ``libjsiesta_ffi.so`` if it is not in ``cpp/build`` of the source tree
   * - ``SIESTA_PS_PATH``
     - Directories searched for ``<Label>.psml`` files, as in SIESTA
   * - ``JSIESTA_COMPILATION_CACHE``
     - Directory for JAX's persistent compilation cache; cuts the start-up time of repeated runs
   * - ``OMP_NUM_THREADS``
     - Threads of the C++ kernels

Troubleshooting the installation: :doc:`../user_guide/troubleshooting`.
