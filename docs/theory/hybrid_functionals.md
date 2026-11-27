# Hybrid Functionals: Exact Exchange with Gaussian Fits of the Orbitals

## TheoryNode: theory.hybrid_functional
A hybrid functional adds a fraction of the exact (Hartree-Fock) exchange energy of the Kohn-Sham density matrix to
a semilocal functional. In the convention of Libxc,

$$E_{xc} = E^{\text{sl}}_{xc}[\rho] + \alpha\,E_x^{HF}\!\left[\tfrac{1}{r}\right] + \beta\,E_x^{HF}\!\left[\tfrac{\operatorname{erfc}(\omega r)}{r}\right],$$

where $E^{\text{sl}}_{xc}$ is the semilocal remainder, evaluated on the mesh like any GGA. PBE0 has $\alpha = 1/4$;
HSE06 has $\alpha = 0$, $\beta = 1/4$ and $\omega = 0.11$ Bohr$^{-1}$, so that only the short-range part of the
exchange is exact. The three numbers are read from Libxc (`exact_exchange_spec`).

## TheoryNode: theory.exact_exchange_matrix
With the density matrix $P^s$ of spin channel $s$ on the sparse pattern (orbital $\mu$ in the home cell, $\lambda$ in
cell $\mathbf G$) and the kernel $w(r) = \alpha/r + \beta\operatorname{erfc}(\omega r)/r$,

$$K^{s,\mathbf G}_{\mu\lambda} = \sum_{\nu\sigma}\sum_{\mathbf N\mathbf H} P^{s,\mathbf H-\mathbf N}_{\nu\sigma}\, (\mu\mathbf 0\,\nu\mathbf N\,|\,\lambda\mathbf G\,\sigma\mathbf H)_w ,\qquad H^s_x = -K^s,\qquad E_x^{HF} = -\tfrac12\sum_s \operatorname{Tr}[P^s K^s] .$$

For `nspin = 1` the stored density matrix is the sum over spin: $H_x = -\tfrac12 K[P]$ and
$E_x = -\tfrac14\operatorname{Tr}[P K[P]]$. In every case $E_x = \tfrac12\operatorname{Tr}[P H_x]$ and
$\partial E_x/\partial P = H_x$, which is what makes the Kohn-Sham and Harris energies agree at self-consistency.
$K$ has the shape of the Hamiltonian, so k-points need nothing new: the Bloch sum is the existing one.

Two approximations are part of the method (they are those of HONPAS):

- **Truncation.** $P$ and $K$ are kept on the sparse pattern (orbital pairs that overlap or share a projector). The
  exact matrices extend further; the error is controlled by the orbital radii and is small for insulators.
- **Range of the lattice sums.** The sums run over the density matrix on the pattern, so their range is that of
  the orbitals. Measured error of the exchange energy per cell against a pattern of twice the range (single-zeta
  basis, PBE density matrix): 0.14 meV (HSE06) and 2 meV (PBE0) for Si, 1.5 and 8 meV for diamond, 0.03 and 0.2 meV for
  a chain of H$_2$ molecules.

## TheoryNode: theory.exact_exchange_periodic_full_range
With an $N_1 N_2 N_3$ mesh of k-points the density matrix is periodic in the supercell of $N_d$ cells along
direction $d$: $P^{\mathbf R} = P^{\mathbf R + N_d\mathbf a_d}$. If the translations of the sparse pattern along $d$
span $N_d$ cells or more, the lattice sum contains the same value at two distances. For the short-range kernel
this is the usual finite-size error of a screened hybrid and decays with $N_d$. For the kernel $1/r$ the repeated
values do not decay and the sum grows with the range of the pattern (measured for a hydrogen chain with one
k-point across the vacuum: $E_x$ from -4.0 to -14.5 eV when the pattern range is multiplied by four). A full-range
hybrid (PBE0, B3LYP) is therefore accepted when the translations used by the pattern span fewer than $N_d$ cells
in every direction (which includes an isolated system, where they span none) and refused otherwise. When
accepted, the sum is the real-space exchange of an insulator, truncated at the orbital range; no correction for
the long-range tail of the density matrix is applied, so metals and small-gap systems are outside its validity.

## TheoryNode: theory.exact_exchange_spinor
For `nspin` 4 and 8 the density matrix has four spin blocks, $P^{ss'}_{\mu\nu} = \sum_i f_i\, c_{i\mu s} c^*_{i\nu s'}$, and

$$H^{ss'}_x = -K[P^{ss'}],\qquad E_x = -\tfrac12\sum_{ss'}\operatorname{Re}\operatorname{Tr}\big[P^{ss'\dagger} K[P^{ss'}]\big] .$$

The integrals are real, so $K$ acts on real and imaginary parts separately; six real matrices are needed
($\operatorname{Re}$ and $\operatorname{Im}$ of $uu$, $dd$ and $ud$; the block $du$ is the Hermitian conjugate of
$ud$). Unlike a semilocal potential, the exchange matrix of the diagonal blocks has an imaginary antisymmetric
part whenever the density matrix has one (a spin current), also without spin-orbit coupling. The Hamiltonian of
`nspin = 4` is therefore complex with a hybrid, and its imaginary part is mixed with the rest. For the forces the
energy is written with the Pauli components $P_a$ ($a = 0, x, y, z$), each Hermitian in the orbital indices:
$E_x = -\tfrac14\sum_a \big(Q[\operatorname{Re}P_a] + Q[\operatorname{Im}P_a]\big)$ with the quadratic form
$Q[M] = \sum M_{\mu\lambda}M_{\nu\sigma}(\mu\nu|\lambda\sigma)_w$. The real parts are symmetric and the imaginary
parts antisymmetric, and for either symmetry the four centres of an integral contribute equally, so the force
expression below applies to each of the eight matrices with $\kappa = 1/4$.

## TheoryNode: theory.nao2gto
The four-centre integrals are analytic for Gaussians. In the exchange term only, each numerical radial function
$R(r)$ is replaced by $\tilde R(r) = r^l\sum_i c_i e^{-a_i r^2}$, fitted in the norm of the orbital with the target set
to zero beyond the cut-off radius (NAO2GTO; Shang, Li and Yang, J. Chem. Phys. 135, 034110 (2011)). The orbital is
then $C_{lm}(\mathbf r)\sum_i c_i e^{-a_i r^2}$ with the real solid harmonics $C_{lm} = r^l Y_{lm}$ of the rest of the
code. Everything else keeps the numerical orbitals. The exchange energy remains a well-defined function of the
density matrix and of the atomic positions, so the identities above hold exactly for the fitted functional; the fit
error is a basis-set error of the exchange term. A confined orbital has a kink at its radius that Gaussians follow
poorly: the residual of the fit stays at 1 to 2.5 % of the orbital norm for five to eight Gaussians (measured for
the PseudoDojo H, O and Si double-zeta polarised bases).

## TheoryNode: theory.electron_repulsion_integrals
The integrals are evaluated with the McMurchie-Davidson scheme. The product of two primitive Gaussians is expanded
in Hermite Gaussians at the product centre $\mathbf P$ with coefficients $E^{ab}_{tuv}$, and

$$(ab|cd) = \frac{2\pi^{5/2}}{pq\sqrt{p+q}}\sum_{tuv}E^{ab}_{tuv}\sum_{\tau\nu\phi}(-1)^{\tau+\nu+\phi}E^{cd}_{\tau\nu\phi}\, R_{t+\tau,u+\nu,v+\phi}(\alpha,\mathbf P-\mathbf Q),\qquad \alpha = \frac{pq}{p+q},$$

where $R_{tuv}$ follows by recursion from $R^n_{000} = (-2\alpha)^n F_n(\alpha R^2)$ and $F_n$ is the Boys function.
The kernel $\operatorname{erf}(\omega r)/r$ gives the same recursion from
$\sqrt{\alpha'/\alpha}\,(-2\alpha')^n F_n(\alpha' R^2)$ with $\alpha' = \alpha\omega^2/(\alpha+\omega^2)$, so
$\operatorname{erfc}(\omega r)/r = 1/r - \operatorname{erf}(\omega r)/r$ costs one more evaluation of the starting
values. The Cartesian form of the solid harmonics is derived at start-up from the harmonics routine used everywhere
else, which fixes the order and the signs of the $m$ components by construction.

## TheoryNode: theory.exact_exchange_forces
At a fixed density matrix, $E_x = -\kappa\sum_s\sum P^s_{\mu\lambda}P^s_{\nu\sigma}(\mu\nu|\lambda\sigma)_w$ with
$\kappa = 1/4$ (`nspin = 1`) or $1/2$ (`nspin = 2`); for `nspin` 4 and 8 see the spinor section. The sum is symmetric under the exchange of the two orbitals of
each pair and of the two pairs, so the four centres of an integral contribute equally and one derivative per
quartet is enough. With $W = \sum_s P^s_{\mu\lambda}P^s_{\nu\sigma}$,

$$\mathbf F_A = 4\kappa\sum_{\mu\in A} W\,\partial_{\mathbf R_\mu}(\mu\nu|\lambda\sigma),\qquad \frac{\partial E_x}{\partial\epsilon} = -\kappa\Big[2\sum W\,\partial_{\mathbf R_\mu}I\otimes(\mathbf R_\mu-\mathbf R_\nu) - \sum W\,\partial_{\mathbf P}I\otimes(\mathbf R_\lambda-\mathbf R_\mu)\Big],$$

where $\partial_{\mathbf P}$ is the derivative with respect to a common translation of $\mu$ and $\nu$ (translation
invariance gives $\partial_\lambda+\partial_\sigma = -\partial_{\mathbf P}$) and the stress is the strain derivative
divided by the volume. The derivative of a shell with respect to its centre follows from
$\partial_{A_x}[x_A^i e^{-a x_A^2}] = 2a\,x_A^{i+1}e^{-a x_A^2} - i\,x_A^{i-1}e^{-a x_A^2}$, and a translation of a pair
shifts the Hermite index, $\partial_{P_x}\Lambda_{tuv} = \Lambda_{t+1,u,v}$; both give expansions of one order higher
that go through the same integral routine. The part of the force that comes from the change of the overlap is in
the energy-density-matrix term, as for every other contribution to the Hamiltonian, because the eigenvalues are
those of the Hamiltonian with the exchange matrix. The Gaussian fits move rigidly with the atoms.

## TheoryNode: theory.exchange_screening
A quartet contributes at most $Q_{IJ}\,Q_{KL}\,\max|P_{JL}|$ with the Schwarz bounds
$Q_{IJ} = \max\sqrt{(ij|ij)_w}$ (the kernel is positive definite, so the Cauchy-Schwarz inequality holds for it).
Shell pairs are listed once per geometry for orbitals that overlap; quartets below the tolerance are skipped in
every build. For bulk Si with a single-zeta basis, a tolerance of 1e-6 changes the exchange energy by 0.7 meV per
cell and the matrix elements by at most 0.3 meV relative to 1e-12 (measured).

## TheoryNode: theory.exchange_integral_reuse
An integral is unchanged under the exchange of the two orbitals of a pair and of the two pairs,
$(\mu\nu|\lambda\sigma) = (\nu\mu|\lambda\sigma) = (\mu\nu|\sigma\lambda) = (\lambda\sigma|\mu\nu)$, so one value
serves up to eight contributions $K_{xy} \mathrel{+}= P_{zw}\,(xz|yw)$; this holds for any density matrix, symmetric
or not. In a crystal each of the eight is a separate term of the lattice sums, present only if its block of $K$
and its block of $P$ are on the sparse pattern; the terms that are present are found for each integral and the
integral is computed by the first of them in a fixed order. For the forces, the derivative with respect to the
second centre of a pair is the derivative with respect to a translation of the pair minus that of the first
centre, so the four orderings within the two pairs share the derivative integrals. The Hermite integrals and
their contraction with the second pair do not depend on the expansion coefficients of the first, so the three
centre derivatives and the three translation derivatives cost one pass. In the self-consistent cycle the matrix
is updated with the exchange of the change of the density matrix, $K[P_n] = K[P_{n-1}] + K[P_n - P_{n-1}]$, where
the screening acts on the change; the full matrix is rebuilt every few steps so that the neglected terms do not
accumulate.

## TheoryNode: theory.exchange_pattern_range
The truncation of $P$ and $K$ to orbital pairs that overlap is removed gradually by a pattern that contains
the pairs within $s$ times the sum of their radii (`HFX.PatternScale`). On the added pairs the overlap and the
semilocal Hamiltonian vanish, so a semilocal calculation is unchanged; the density matrix and the exchange
matrix are computed there like everywhere else. The extension is meaningful when the k-point mesh spans the range
of the pattern: otherwise the density matrix repeats within it and the added terms are periodic copies, damped
only by the kernel.

## TheoryNode: theory.exchange_distance_bound
The Schwarz bound does not decay with the distance between the two pair densities, the short-range kernel does.
If $|\phi_\mu\phi_\nu|(\mathbf r) \le w\,e^{-\gamma|\mathbf r-\mathbf c|^2}$, then for a positive kernel

$$|(\mu\nu|\lambda\sigma)_{\text{sr}}| \le q_1 q_2\,\frac{\operatorname{erf}(\sqrt{\alpha}R) - \operatorname{erf}(\sqrt{\alpha_s}R)}{R},\qquad q = w\left(\frac{\pi}{\gamma}\right)^{3/2},\quad \alpha = \frac{\gamma_1\gamma_2}{\gamma_1+\gamma_2},\quad \alpha_s = \frac{\alpha\omega^2}{\alpha+\omega^2},$$

the short-range interaction of two Gaussians at the distance $R$ of the two centres. The envelope of each shell
pair is found numerically from the Gaussian fits, including their tails beyond the cut-off radius, and a quartet
is screened with the smaller of this bound and the Schwarz product. For orbitals of radius 7 Bohr and
$\omega = 0.11$ Bohr$^{-1}$ the range of the standard pattern is not longer than that of the kernel, so the gain
is small there (7 % of the quartets for Si) and larger on an extended pattern (40 % at twice the range).

