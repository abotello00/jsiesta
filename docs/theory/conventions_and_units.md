# Conventions, Units, and Spin Representations

## TheoryNode: theory.units_and_geometry
All internal calculations in `jsiesta` use **Hartree atomic units** ($\hbar = m_e = |e| = 4\pi\varepsilon_0 = 1$):

- Length: Bohr ($a_0 = 0.529177210903\text{ \AA}$)
- Energy: Hartree ($E_h = 27.211386245988\text{ eV} = 2\text{ Ry}$)
- Force: $E_h / a_0$
- Stress: $E_h / a_0^3$
- Temperature: $k_B T_e$ in $E_h$ ($k_B = 3.1668115634556 \times 10^{-6}\ E_h/\text{K}$)

Unit-cell lattice vectors are stored as columns of $\mathbf{A} = [\mathbf{a}_1, \mathbf{a}_2, \mathbf{a}_3] \in \mathbb{R}^{3\times 3}$ so Cartesian positions satisfy $\mathbf{r} = \mathbf{A}\mathbf{f}$ and reciprocal lattice vectors satisfy $\mathbf{A}^T \mathbf{B} = 2\pi \mathbf{I}$.

## TheoryNode: theory.real_spherical_harmonics_and_spinors
Numerical Atomic Orbitals use orthonormal **real spherical harmonics** $Y_{lm}(\hat{\mathbf{r}})$ ordered in SIESTA canonical order $m \in [-l, l]$:

$$\phi_{I n l m \zeta}(\mathbf{r}) = R_{s(I) n l \zeta}(r) \, Y_{lm}(\hat{\mathbf{r}}), \qquad \int_{S^2} Y_{lm}(\hat{\mathbf{r}}) Y_{l'm'}(\hat{\mathbf{r}})\, d\Omega = \delta_{ll'}\delta_{mm'}.$$

Spin modes follow SIESTA's four `nspin` sectors:

1. `nspin = 1`: Spin-unpolarized scalar density $\rho(\mathbf{r})$.
2. `nspin = 2`: Collinear spin-polarized $(\rho_\uparrow(\mathbf{r}), \rho_\downarrow(\mathbf{r}))$.
3. `nspin = 4`: Non-collinear $2\times 2$ Hermitian density matrix with magnetization axial vector $\mathbf{m}(\mathbf{r})$ transforming as $\det(R_S) R_S \mathbf{m}$.
4. `nspin = 8`: Full spin-orbit coupling (SOC) complex $2$-spinor Hamiltonian $H^{\alpha\beta}(\mathbf{k})$ transforming under $U(R_S) \in \mathrm{SU}(2)$.
