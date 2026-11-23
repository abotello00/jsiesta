# Fixed-(D, E) Variational Forces and Stress Tensor

## TheoryNode: theory.variational_forces_and_stress
By the Kohn–Sham variational theorem, at SCF convergence $\partial E_{\text{KS}} / \partial D = 0$ subject to the orbital orthonormality constraint $\mathrm{Tr}[D S] = N_e$. Therefore, we do **not** differentiate through the iterative eigensolver or mixer (`jax.lax.stop_gradient` on converged $D$ and $E$):

$$\mathbf{F}_I = -\left.\frac{\partial E_{\text{KS}}}{\partial \mathbf{R}_I}\right|_{D, E} = -\mathrm{Tr}\!\left[D \frac{\partial (T + V^{\text{NA}} + V^{\text{KB}})}{\partial \mathbf{R}_I}\right] + \mathrm{Tr}\!\left[E \frac{\partial S}{\partial \mathbf{R}_I}\right] - \frac{\partial E_{\text{grid}}}{\partial \mathbf{R}_I} - \frac{\partial E_{\text{disp}}}{\partial \mathbf{R}_I},$$

$$\sigma_{\alpha\beta} = \frac{1}{\Omega} \left.\frac{\partial E_{\text{KS}}}{\partial \epsilon_{\alpha\beta}}\right|_{D, E},$$

where $-\mathrm{Tr}[E\,\partial S/\partial \mathbf{R}_I]$ is the Pulay force from orbital-center motion, and $\partial E_{\text{grid}}/\partial \mathbf{R}_I$ (`dfscf`) combines the discrete orbital-grid VJP with the neutral-atom/NLCC grid derivatives, ensuring exact consistency between the discrete mesh energy and analytical forces/stresses.
