C++ and CUDA kernels
====================

The compute-intensive inner loops are C++ (and CUDA) functions in ``cpp/``, compiled into ``libjsiesta_ffi`` and
called from JAX through the XLA foreign-function interface. Python never loops over orbital pairs or mesh
points; it prepares flat arrays and calls a kernel. Each kernel that enters the energy has a matching adjoint
kernel, which is what the automatic differentiation of forces and stress calls.

This page is generated from the Doxygen comments of the headers and sources. The Python wrappers of these
kernels are in :doc:`integrals`, :doc:`grid` and :doc:`xc`.

.. include:: ../_generated/api/cpp.inc
