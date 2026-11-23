# Native Symmetry, Real-Space Grid, and Adaptive SCF

## TheoryNode: theory.native_symmetry
Space-group and Shubnikov Type I–IV magnetic space-group operations $g = \{R_S | \mathbf{v}_S\}$ (including anti-unitary time reversal $\mathcal{T} \cdot g$) are detected via `spglib`. Real-harmonic Wigner-$D$ matrices $D^l(S)$ are fitted via least squares over **16 Fibonacci-sphere directions** and verified orthogonal ($\|D^l (D^l)^T - I\|_\infty \le 10^{-10}$).

The sparse orbital-pair neighbor graph is closed under all $g \in G$, and the projection $\mathcal{P}_{\text{sym}}$ is precompiled into static COO/CSR gather-scatter weights acting in $O(N_{\text{edges}})$ via `segment_sum`. Grid dimensions $(N_1, N_2, N_3)$ are chosen commensurate with all fractional translations so every $S \in G$ induces an exact grid permutation $\pi_S$:

$$\rho(\mathbf{r}) \leftarrow \frac{1}{|G|} \sum_{S \in G} \rho(\pi_S^{-1}(\mathbf{r})).$$

## TheoryNode: theory.grid_poisson_xc_dispersion
On the uniform 3D periodic mesh, $\rho(\mathbf{r})$ and $V_{\text{eff}}(\mathbf{r})$ are mapped via bounded-sphere C++/CUDA XLA FFI kernels (`rhoofd`, `vmat`) satisfying the exact discrete adjoint identity $\langle V | \rho(D) \rangle = \mathrm{Tr}[D V_{\text{grid}}]$. Periodic Poisson electrostatics ($\nabla^2 V_H = -4\pi \delta\rho$), GGA gradients ($\nabla\rho$), and Sabatini rVV10 nonlocal correlation execute in `jax.numpy.fft`, paired with C++ FFI pointwise `xc_eval` (`libxc` or analytical built-in C++ LDA/PBE) and differentiable pairwise DFT-D3(BJ).

## TheoryNode: theory.adaptive_scf_and_dm_out
Cold-start mixing parameters ($w_{\text{base}}$, $N_{\text{hist}}$, $\lambda_{\text{Tikhonov}}$) are derived from Thomas–Fermi conditioning $\kappa_\varepsilon$, $d/f$-shell stiffness, vacuum fraction, and spin mode. Multiplicative Tikhonov–Pulay regularizes the Gram matrix with deterministic $1\times 1$ SVD handling and decoupled charge/spin preconditioning ($w_{\text{spin}} / w_{\text{charge}} \approx 1.4\text{--}1.6$) while preserving a stationary metric during initial Krylov construction (`iscf <= 4`) and applying trust-region backoff on residual spikes. Upon convergence at exact target $T_e$, a mandatory non-mixed post-SCF `DM_out` pass computes `final_post_scf_free_e`.
