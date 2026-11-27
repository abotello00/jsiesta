Input keywords (FDF)
====================

jsiesta reads SIESTA's FDF format: labels are case-insensitive and ``.``, ``_`` and ``-`` in them are ignored;
values may carry units; ``#``, ``!`` and ``;`` start comments; ``%block Label`` ... ``%endblock`` encloses
multi-line data; ``%include file`` and ``%block Label < file`` are followed; the first definition of a label
wins.

A label that is not in the tables below is reported in the output as *not used*. Nothing is ignored silently
except the print-control options listed at the end.

Structure
---------

.. list-table::
   :header-rows: 1
   :widths: 34 16 50

   * - Keyword
     - Default
     - Notes
   * - ``SystemLabel``
     - ``siesta``
     - Prefix of the output files
   * - ``LatticeConstant``
     - 1 Bohr
     - Units: ``Bohr``, ``Ang``, ``nm``
   * - ``%block LatticeVectors``
     - identity
     - Rows are the lattice vectors, in units of ``LatticeConstant``
   * - ``%block LatticeParameters``
     -
     - ``a b c alpha beta gamma`` (used when ``LatticeVectors`` is absent)
   * - ``%block ChemicalSpeciesLabel``
     - required
     - ``index Z Label``; the pseudopotential is ``Label.psml``. Ghost atoms (negative Z) are not supported.
   * - ``AtomicCoordinatesFormat``
     - ``Bohr``
     - ``Bohr``, ``Ang``, ``Fractional`` / ``ScaledByLatticeVectors``, ``ScaledCartesian`` and the
       ``NotScaledCartesian...`` names
   * - ``%block AtomicCoordinatesAndAtomicSpecies``
     - required
     - ``x y z species_index``
   * - ``NetCharge``
     - 0
     - In units of \|e\|; positive removes electrons

Basis set
---------

.. list-table::
   :header-rows: 1
   :widths: 34 16 50

   * - Keyword
     - Default
     - Notes
   * - ``PAO.BasisSize``
     - ``DZP``
     - ``SZ``, ``DZ``, ``TZ``, ``QZ``, with ``P`` or ``2P`` (``SZP``, ``DZP``, ``TZP``, ``TZ2P``...);
       ``MINIMAL`` and ``STANDARD``
   * - ``PAO.EnergyShift``
     - 0.02 Ry
     - Sets the cut-off radii of the first zetas
   * - ``PAO.SplitNorm``
     - 0.15
     - Sets the radii of the inner zetas
   * - ``PAO.SplitNormH``
     - ``PAO.SplitNorm``
     - Split norm of hydrogen species; 0.5 is recommended (see :doc:`pseudos_and_basis`)
   * - ``%block PAO.Basis``
     -
     - Explicit shells; see :doc:`pseudos_and_basis` for the supported options
   * - ``User.Basis``
     - ``F``
     - Read the basis from ``Label.ion.xml`` written by SIESTA instead of generating it

Hamiltonian
-----------

.. list-table::
   :header-rows: 1
   :widths: 34 16 50

   * - Keyword
     - Default
     - Notes
   * - ``XC.Functional`` / ``XC.Authors``
     - ``LDA`` / ``CA``
     - ``LDA``: ``CA`` (= ``PZ``), ``PW92``. ``GGA``: ``PBE``, ``PBEsol``. Also accepted as ``XC.Authors``:
       ``R2SCAN``, ``SCAN``, ``TPSS``, and ``LIBXC-<x>+<c>`` with Libxc numbers. Hybrid functionals:
       ``XC.Authors HSE06``, ``PBE0``, ``B3LYP``, or a Libxc hybrid id. ``PBE0`` and ``B3LYP`` (full-range
       exchange) need, in a crystal, a k-point mesh at least as fine as the range of the orbitals (the error
       message gives the divisions) and an insulator. Hybrids work for every ``Spin`` mode (energies,
       eigenvalues, forces and stress) and are much slower than a GGA.
   * - ``HFX.NumberOfGaussians``
     - 5
     - jsiesta only. Gaussians fitted to each orbital shell for the exact exchange of a hybrid functional.
   * - ``HFX.ScreeningTolerance``
     - 1e-7
     - jsiesta only. Four-centre integrals whose bound times the density matrix is below this (Hartree) are
       skipped.
   * - ``HFX.PatternScale``
     - 1
     - jsiesta only. The density matrix and the exchange matrix are kept for orbital pairs within this multiple
       of the sum of their radii (1: pairs that overlap, as in HONPAS). Larger values reduce the truncation
       error of the exact exchange at a higher cost; semilocal terms are not affected.
   * - ``Spin``
     - ``none``
     - ``none``, ``polarized``, ``non-collinear``, ``spin-orbit``. The older ``SpinPolarized``,
       ``NonCollinearSpin`` and ``SpinOrbit`` flags work too. Spin-orbit needs fully relativistic PSML files.
   * - ``%block DM.InitSpin``
     - all atoms +1
     - ``atom moment [theta phi]``; ``+`` / ``-`` for the sign only; angles in degrees for non-collinear runs
   * - ``MeshCutoff``
     - 300 Ry
     - The mesh used is the smallest one that gives at least this cutoff
   * - ``Mesh.SubDivisions``
     - 2
     -
   * - ``Mesh.Sizes``
     -
     - Explicit mesh ``n1 n2 n3``
   * - ``DFTD3``
     - ``F``
     - DFT-D3(BJ) dispersion (needs the ``dftd3`` package)
   * - ``JSiesta.Dispersion``
     - ``none``
     - ``D3`` or ``RVV10`` (jsiesta extension)

Brillouin zone and occupations
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 34 16 50

   * - Keyword
     - Default
     - Notes
   * - ``%block kgrid.MonkhorstPack``
     - Gamma
     - Diagonal grids only; the fourth column is the displacement. ``kgrid.Cutoff`` is not supported.
   * - ``JSiesta.UseSymmetry``
     - ``F``
     - Use the space group: irreducible k-points, symmetrised density matrix, forces and stress (jsiesta
       extension). ``kgrid.UseSymmetry`` is accepted as an alias.
   * - ``ElectronicTemperature``
     - 300 K
     - Units ``K``, ``eV``, ``meV``, ``Ry``...
   * - ``OccupationFunction``
     - ``FD``
     - Fermi-Dirac only

Self-consistency
----------------

.. list-table::
   :header-rows: 1
   :widths: 34 16 50

   * - Keyword
     - Default
     - Notes
   * - ``MaxSCFIterations``
     - 100
     -
   * - ``SCF.DM.Tolerance`` (``DM.Tolerance``)
     - 1e-5
     - SIESTA's default is 1e-4
   * - ``SCF.H.Tolerance``
     - 1e-5 Ha (0.27 meV)
     - SIESTA's default is 1 meV
   * - ``SCF.FreeE.Tolerance``
     - 1e-6 Ha
     - Always applied
   * - ``SCF.Mix``
     - ``Hamiltonian``
     - ``Hamiltonian`` or ``density``
   * - ``SCF.Mixer.Method``
     - Tikhonov-Pulay
     - ``pulay`` or ``broyden`` select those; the default is jsiesta's regularised Pulay with an adaptive
       controller
   * - ``SCF.Mixer.Weight`` (``DM.MixingWeight``)
     - automatic
     - Chosen from the system when absent
   * - ``SCF.Mixer.History`` (``DM.NumberPulay``)
     - automatic
     -
   * - ``SCF.MustConverge``
     - ``T``
     - Exit status 1 without convergence

Files and post-processing
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 34 16 50

   * - Keyword
     - Default
     - Notes
   * - ``DM.UseSaveDM``, ``File.DM.Init``
     - ``F``
     - Start from a ``.DM`` file
   * - ``MD.UseSaveXV``
     - ``F``
     - Geometry from the ``.XV`` file
   * - ``UseSaveData``
     - ``F``
     - Both of the above
   * - ``CDF.Save``
     - ``F``
     - Also write ``<label>.nc``
   * - ``%block BandLines``, ``BandLinesScale``
     -
     - Write ``<label>.bands``

Accepted and ignored
--------------------

``SystemName``, ``NumberOfAtoms``, ``NumberOfSpecies``, the ``Write...`` and ``Save...`` print options,
``LongOutput``, ``SolutionMethod`` (diagonalisation is the only solver), ``Diag.*``, ``NumberOfEigenStates``,
``XML.Write``, ``PAO.BasisType``, ``DM.MixSCF1``, timer and allocation reports.

Requested but not done
----------------------

``MD.TypeOfRun`` with ``MD.Steps`` > 0 (relaxation or dynamics) gives a warning and a single point of the input
geometry. To relax a structure, drive jsiesta from Python: see the tutorial on molecular geometry.
