# Confined NAOs, PSML Pseudopotentials, and Two-Center Integrals

## TheoryNode: theory.psml_and_kb
Norm-conserving pseudopotentials are loaded exclusively from **PSML (`.psml`) v1.1** XML files:

$$\hat{V}_{\text{PS}} = V_{\text{loc}}(r) + \sum_{l, j, m, p} |\chi_{l j m p}\rangle E^{\text{KB}}_{l j p} \langle \chi_{l j m p}|.$$

For scalar-relativistic calculations, the weighted spin-average channel is used; for `nspin = 8` (SOC), $j = l \pm 1/2$ channels generate the spin-orbit coupling matrix $H^{\text{SO}}$ alongside scalar nonlocal terms. Non-linear core corrections (NLCC) enter via the PSML pseudo-core charge $\rho^{\text{core}}(r)$.

## TheoryNode: theory.neutral_atom_screening
To eliminate long-range $1/r$ tail cancellations between ion-ion, Hartree, and electron-ion terms, `jsiesta` follows SIESTA's neutral-atom screening decomposition:

$$V^{\text{NA}}(r) = V_{\text{loc}}(r) + V_H[\rho^{\text{NA}}](r), \qquad \delta\rho(\mathbf{r}) = \rho(\mathbf{r}) - \sum_I \rho^{\text{NA}}_I(|\mathbf{r} - \mathbf{R}_I|),$$

where both $V^{\text{NA}}(r)$ and $\rho^{\text{NA}}(r)$ are strictly localized within the neutral-atom cutoff $r_c^{\text{NA}}$.

### Neutral-atom energy bookkeeping (absolute energy and egg-box control)
With $\rho_L = -\nabla^2 V_{\text{loc}}/4\pi$ (charge $-Z$), $n_{\text{at}}$ the neutral pseudo-atom valence density and
$V_{NA} = V_H[\rho_L + n_{\text{at}}]$, the point-ion electrostatic energy is evaluated as

$$E_{es} = \int V_{NA}\,\delta\rho \;+\; E_H[\delta\rho] \;+\; \sum_{I\neq J} W_{IJ}(R_{IJ}) \;+\; \sum_{I<J} U_{IJ}(R_{IJ})
\;+\; \sum_I \big(\tfrac12 J[\rho_{NA}] - \tfrac12 J[\rho_L]\big)_I ,$$

with $\delta\rho = \rho - \sum_I n_{\text{at},I}$ on the mesh, the exact two-centre integrals
$W_{IJ}(R) = \int V_{NA,I}\, n_{J}(\mathbf r - \mathbf R)$, the pair term $U_{IJ} = Z_IZ_J/R - J[n_I, n_J](R)$, and the one-centre
constants $\tfrac12 J[\rho] = \tfrac12\int r^2 V'(r)^2 dr$ (plus $Z^2/2r_{\max}$ for the $-Z/r$ tail of $V_{\text{loc}}$).
Only $\int V_{NA}\,\delta\rho$ is integrated on the mesh, as in SIESTA (`DEna` on the mesh, `Ena` two-centre), which removes
the egg-box error of the atomic part from energies, forces and stress; the one-centre constants make the total energy
absolute and equal to SIESTA's `Etot` (`Ena + DEna - Eions`). See `logs/DECISIONS.md` D-016/D-017.

## TheoryNode: theory.two_center_matel
Two-center overlap $S_{\mu\nu}$, kinetic $T_{\mu\nu}$, and KB overlap integrals are expanded via spherical Bessel transforms and Gaunt coefficients $G^{L}_{l_1 m_1, l_2 m_2}$:

$$I_{l_1 m_1, l_2 m_2}(\mathbf{R}) = \sum_{L=|l_1-l_2|}^{l_1+l_2} \sum_{M=-L}^{L} G^L_{l_1 m_1, l_2 m_2, M} \, Y_{LM}(\hat{\mathbf{R}}) \, F_{l_1 l_2 L}(|\mathbf{R}|).$$

Radial curves $F_{l_1 l_2 L}(R)$ are tabulated on a dense grid and evaluated with natural cubic splines in C++, yielding machine-precision analytical displacement derivatives $\nabla_{\mathbf{R}} I$.
