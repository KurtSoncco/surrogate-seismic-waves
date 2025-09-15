import itertools
from functools import partial

import jax
import jax.numpy as jnp

from wave_surrogate.logging_setup import setup_logging
from wave_surrogate.pce.polynomials import _hermite_polynomial, _legendre_polynomial

logger = setup_logging()


class PCEOperatorJAX:
    """
    Implements the Polynomial Chaos Expansion for Operator Learning using JAX.
    Reference: arXiv:2508.20886v1
    """

    def __init__(self, p_order: int, q_order: int, r_dim: int, d_dim: int):
        """
        Initializes the PCE Operator with JAX.
        Args:
            p_order (int): Total polynomial degree for stochastic variables (p).
            q_order (int): Total polynomial degree for spatio-temporal variables (q).
            r_dim (int): Dimensionality of stochastic input (r).
            d_dim (int): Dimensionality of spatio-temporal domain (d+1).
        """
        self.p = p_order
        self.q = q_order
        self.r = r_dim
        self.d = d_dim
        self.C = None  # The learned coefficient matrix.

        # Generate multi-indices for polynomial bases
        self.alpha_indices = self._generate_total_degree_indices(self.r, self.p)
        self.beta_indices = self._generate_total_degree_indices(self.d, self.q)

        self.P = self.alpha_indices.shape[0]
        self.Q = self.beta_indices.shape[0]
        logger.info(f"JAX Model initialized: P={self.P}, Q={self.Q}")

    def _generate_total_degree_indices(self, dim: int, order: int) -> jnp.ndarray:
        """Generates multi-indices based on total degree truncation scheme."""
        indices = []
        for o in range(order + 1):
            for idx in itertools.product(range(o + 1), repeat=dim):
                if sum(idx) == o:
                    indices.append(idx)
        return jnp.array(indices, dtype=jnp.int32)

    @staticmethod
    @partial(jax.jit, static_argnames=["index", "basis_func"])
    def _evaluate_single_basis_for_all_points(
        points: jnp.ndarray, index: tuple[int, ...], basis_func: str = "hermite"
    ) -> jnp.ndarray:
        """
        Evaluates a single multivariate polynomial basis function at all given points.
        This function is JIT-compiled for speed.

        Args:
            points (jnp.ndarray): Input points of shape (N, dim).
            index (tuple[int, ...]): Multi-index for the polynomial basis.
            basis_func (str): Type of polynomial basis ("hermite" or "legendre").

        Returns:
            jnp.ndarray: Evaluated values of shape (N,).
        """
        # Ensures the basis function is in the correct format
        basis_functions_map = {
            "hermite": _hermite_polynomial,
            "legendre": _legendre_polynomial,
        }
        eval_1d_poly = basis_functions_map[basis_func]

        # Defines the evaluation for a single point using the specified multi-index
        def eval_multivariate_poly(point: jnp.ndarray) -> jnp.ndarray:
            # We need to convert the static tuple `index` back to a JAX array for vmap
            index_arr = jnp.array(index)
            # vmap the 1D poly over the dimensions of the point and the index
            evals_1d = jax.vmap(eval_1d_poly, in_axes=(0, 0))(index_arr, point)
            return jnp.prod(evals_1d)

        # vmap the multivariate evaluation over all input points
        return jax.vmap(eval_multivariate_poly)(points)

    @staticmethod
    def _construct_basis_matrix(
        points: jnp.ndarray, indices: jnp.ndarray, basis_func: str = "hermite"
    ):
        """
        Constructs a basis matrix by iterating through indices and calling a
        JIT-compiled helper for each one.

        Args:
            points (jnp.ndarray): Input points of shape (N, dim).
            indices (jnp.ndarray): Multi-indices of shape (M, dim).
            basis_func (str): Type of polynomial basis ("hermite" or "legendre").

        Returns:
            jnp.ndarray: Basis matrix of shape (N, M).
        """

        basis_func = basis_func.lower()
        if basis_func not in ["hermite", "legendre"]:
            raise NotImplementedError(
                f"Basis function '{basis_func}' is not implemented."
            )

        # A Python list comprehension iterates through the indices. This loop
        # runs outside of JIT.
        # Each `index` is converted to a tuple, which is a valid static argument.
        basis_vectors = [
            PCEOperatorJAX._evaluate_single_basis_for_all_points(
                points, tuple(map(int, index)), basis_func
            )
            for index in indices
        ]

        # Stack the resulting column vectors to form the final matrix.
        return jnp.stack(basis_vectors, axis=-1)

    def fit(
        self,
        S_true: jnp.ndarray,
        spatio_temporal_coords: jnp.ndarray,
        xi_samples: jnp.ndarray,
    ):
        """
        Fits the PCE model using labeled data. This method calls the
        internal JIT-compiled fitting function.

        Args:
            S_true (jnp.ndarray): True solution data of shape (n, N).
            spatio_temporal_coords (jnp.ndarray): Spatio-temporal coordinates of shape (n, d+1).
            xi_samples (jnp.ndarray): Stochastic input samples of shape (N, r).

        Returns: None
        Raises: RuntimeError: If the fitting process fails.
        """

        logger.info("Constructing basis matrices Phi and Psi...")
        Phi = self._construct_basis_matrix(
            spatio_temporal_coords, self.beta_indices, "legendre"
        )

        Psi = self._construct_basis_matrix(
            xi_samples, self.alpha_indices, "hermite"
        ).T  # Note: shape is (P, N)

        logger.info("Compiling and running the fit function...")
        self.C = self._fit_jit(S_true, Phi, Psi)
        self.C.block_until_ready()  # Wait for computation to finish
        logger.info("Fit complete. Coefficient matrix C has been learned.")

    @staticmethod
    @jax.jit
    def _fit_jit(
        S_true: jnp.ndarray,
        Phi: jnp.ndarray,
        Psi: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        JIT-compiled internal function to solve for C* from Eq. (14).
        Phi and Psi are constructed using the provided multi-indices.

        Args:
            S_true (jnp.ndarray): True solution data of shape (n, N).
            Phi (jnp.ndarray): Basis matrix for spatio-temporal coordinates of shape (n, Q).
            Psi (jnp.ndarray): Basis matrix for stochastic input samples of shape (P, N).

        Returns:
            jnp.ndarray: The learned coefficient matrix C of shape (Q, P).
        Raises: RuntimeError: If the fitting process fails.
        """

        PhiT_Phi = Phi.T @ Phi
        Psi_PsiT = Psi @ Psi.T

        Temp = Phi.T @ S_true @ Psi.T

        C_intermediate = jnp.linalg.solve(PhiT_Phi, Temp)
        C_final_T = jnp.linalg.solve(Psi_PsiT, C_intermediate.T)

        return C_final_T.T

    def fit_physics_informed(
        self,
        pde_collocation_points: jnp.ndarray,
        bc_collocation_points: jnp.ndarray,
        ic_collocation_points: jnp.ndarray,
        pde_fn,
        bc_fn,
        ic_fn,
        xi_samples: jnp.ndarray,
    ):
        """
        Fits the PCE model using physics-informed constraints.
        This method calls the internal JIT-compiled fitting function.
        """
        logger.info("Constructing basis matrix Psi...")
        Psi = self._construct_basis_matrix(xi_samples, self.alpha_indices, "hermite").T

        # PDE Residuals
        L_Phi_pde, F_pde = pde_fn(pde_collocation_points, self.beta_indices, xi_samples)
        # BC Residuals
        L_Phi_bc, F_bc = bc_fn(bc_collocation_points, self.beta_indices, xi_samples)
        # IC Residuals
        L_Phi_ic, F_ic = ic_fn(ic_collocation_points, self.beta_indices, xi_samples)

        n_pde = pde_collocation_points.shape[0]
        n_bc = bc_collocation_points.shape[0]
        n_ic = ic_collocation_points.shape[0]

        logger.info(
            f"Number of PDE points: {n_pde}, BC points: {n_bc}, IC points: {n_ic}"
        )
        logger.info(
            f"Type of number of PDE points: {type(n_pde)}, BC points: {type(n_bc)}, IC points: {type(n_ic)}"
        )

        # Do calculation
        self.C = self._fit_physics_informed_jit(
            Psi, L_Phi_pde, F_pde, L_Phi_bc, F_bc, L_Phi_ic, F_ic, n_pde, n_bc, n_ic
        )

        self.C.block_until_ready()  # Wait for computation to finish
        logger.info(
            "Physics-informed fit complete. Coefficient matrix C has been learned."
        )

    @staticmethod
    @partial(jax.jit, static_argnames=["n_pde", "n_bc", "n_ic"])
    def _fit_physics_informed_jit(
        Psi: jnp.ndarray,
        L_Phi_pde: jnp.ndarray,
        F_pde: jnp.ndarray,
        L_Phi_bc: jnp.ndarray,
        F_bc: jnp.ndarray,
        L_Phi_ic: jnp.ndarray,
        F_ic: jnp.ndarray,
        n_pde: int,
        n_bc: int,
        n_ic: int,
    ) -> jnp.ndarray:
        """
        JIT-compiled internal function to solve for C* from Eq. (19).
        Args:
            Psi (jnp.ndarray): Basis matrix for stochastic input samples of shape (P, N).
            L_Phi_pde (jnp.ndarray): PDE operator applied to basis matrix of shape (n_pde, Q).
            F_pde (jnp.ndarray): Forcing term at PDE points of shape (n_pde, N).
            L_Phi_bc (jnp.ndarray): BC operator applied to basis matrix of shape (n_bc, Q).
            F_bc (jnp.ndarray): Forcing term at BC points of shape (n_bc, N).
            L_Phi_ic (jnp.ndarray): IC operator applied to basis matrix of shape (n_ic, Q).
            F_ic (jnp.ndarray): Forcing term at IC points of shape (n_ic, N).
            n_pde (int): Number of PDE collocation points.
            n_bc (int): Number of BC collocation points.
            n_ic (int): Number of IC collocation points.

        Returns:
            jnp.ndarray: The learned coefficient matrix C of shape (Q, P).
        Raises: RuntimeError: If the fitting process fails.
        """
        # Precompute Psi @ Psi^T
        Psi_Psi_T = Psi @ Psi.T

        # Apply normalization factors from the paper's loss function
        w_pde = jnp.where(n_pde > 0, 2.0 / n_pde, 0.0)
        w_bc = jnp.where(n_bc > 0, 2.0 / n_bc, 0.0)
        w_ic = 2.0 / n_ic if n_ic > 0 else 0.0

        # PDE Residuals
        A_pde = w_pde * (L_Phi_pde.T @ L_Phi_pde)
        F_term_pde = w_pde * (L_Phi_pde.T @ F_pde @ Psi.T)

        # BC Residuals
        A_bc = w_bc * (L_Phi_bc.T @ L_Phi_bc)
        F_term_bc = w_bc * (L_Phi_bc.T @ F_bc @ Psi.T)

        # IC Residuals
        A_ic = w_ic * (L_Phi_ic.T @ L_Phi_ic)
        F_term_ic = w_ic * (L_Phi_ic.T @ F_ic @ Psi.T)

        # Combine terms
        A = A_pde + A_bc + A_ic
        F_term = F_term_pde + F_term_bc + F_term_ic

        # Solve for C
        C_intermediate = jnp.linalg.solve(A, F_term)
        C_final_T = jnp.linalg.solve(Psi_Psi_T, C_intermediate.T)
        return C_final_T.T

    def fit_physics_informed_nonlinear(
        self,
        residual_fn,
        xi_samples: jnp.ndarray,
        n_pde: int,
        n_bc: int,
        n_ic: int,
        n_iter: int = 10,
        tol: float = 1e-6,
    ):
        """
        Fits the PCE model using physics-informed constraints for nonlinear PDEs.
        This method uses the Newton-Raphson method to minimize the residuals.
        Args:
            residual_fn: A function that computes the PDE, BC, and IC residuals.
                It should have the signature:
                (R_pde, R_bc, R_ic) = residual_fn(C, Psi)
                where C is the coefficient matrix and Psi is the basis matrix.
            xi_samples (jnp.ndarray): Stochastic input samples of shape (N, r).
            n_pde (int): Number of PDE collocation points.
            n_bc (int): Number of BC collocation points.
            n_ic (int): Number of IC collocation points.
            n_iter (int): Maximum number of Newton-Raphson iterations. Default is 10.
            tol (float): Tolerance for convergence. Default is 1e-6.

        Returns: None
        Raises: RuntimeError: If the fitting process fails.
        """
        Psi = self._construct_basis_matrix(xi_samples, self.alpha_indices, "hermite").T

        # Define loss function based on Eq. (17)
        def loss_fn(C_flat):
            C = C_flat.reshape((self.Q, self.P))
            # Compute residuals at PDE, BC, and IC points
            R_pde, R_bc, R_ic = residual_fn(C, Psi)

            # Mean squared residuals
            loss_pde = jnp.where(n_pde > 0, jnp.mean(R_pde**2), 0.0)
            loss_bc = jnp.where(n_bc > 0, jnp.mean(R_bc**2), 0.0)
            loss_ic = jnp.where(n_ic > 0, jnp.mean(R_ic**2), 0.0)
            return loss_pde + loss_bc + loss_ic

        # Get JIT-compiled gradient and Hessian functions
        grad_loss = jax.jit(jax.grad(loss_fn))
        hess_loss = jax.jit(jax.hessian(loss_fn))

        # Initial guess for C
        C_flat = jnp.zeros((self.Q * self.P,))

        logger.info("Starting nonlinear physics-informed fitting...")
        for i in range(n_iter):
            g = grad_loss(C_flat)
            H = hess_loss(C_flat)

            # Solve for H * delta_C = -g using JAX's linear solver
            delta_C = jnp.linalg.solve(H, -g)
            C_flat += delta_C

            step_norm = jnp.linalg.norm(delta_C)
            logger.info(
                f"Iteration {i + 1}, Loss: {loss_fn(C_flat):.6f}, Step Norm: {step_norm:.6e}"
            )

            if step_norm < tol:
                logger.info("Convergence achieved.")
                break

        self.C = C_flat.reshape((self.Q, self.P))
        self.C.block_until_ready()  # Wait for computation to finish
        logger.info(
            "Nonlinear physics-informed fit complete. Coefficient matrix C has been learned."
        )

    def predict(
        self, spatio_temporal_coords: jnp.ndarray, xi_samples: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Predicts the solution for new inputs using the learned C matrix.

        Args:
            spatio_temporal_coords (jnp.ndarray): Spatio-temporal coordinates of shape (n, d+1).
            xi_samples (jnp.ndarray): Stochastic input samples of shape (N, r).

        Returns:
            jnp.ndarray: Predicted solution of shape (n, N).
        Raises: RuntimeError: If the model has not been fitted yet.
        """
        if self.C is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        Phi_test = self._construct_basis_matrix(
            spatio_temporal_coords, self.beta_indices, "legendre"
        )
        Psi_test = self._construct_basis_matrix(
            xi_samples, self.alpha_indices, "hermite"
        ).T  # Note: shape is (P, N)

        return self._predict_jit(self.C, Phi_test, Psi_test)

    @staticmethod
    @jax.jit
    def _predict_jit(
        C: jnp.ndarray,
        Phi_test: jnp.ndarray,
        Psi_test: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        JIT-compiled internal function for prediction using Eq. (10).
        Args:
            C (jnp.ndarray): Coefficient matrix of shape (Q, P).
            Phi_test (jnp.ndarray): Basis matrix for spatio-temporal coordinates of shape (n, Q).
            Psi_test (jnp.ndarray): Basis matrix for stochastic input samples of shape (P, N).

        Returns:
            jnp.ndarray: Predicted solution of shape (n, N).
        """

        return Phi_test @ C @ Psi_test

    def quantify_uncertainty(self, spatio_temporal_coords: jnp.ndarray) -> tuple:
        """
        Computes mean and covariance. JIT-compiled for speed.

        Args:
            spatio_temporal_coords (jnp.ndarray): Spatio-temporal coordinates of shape (n, d+1).

        Returns:
            tuple: Mean (jnp.ndarray of shape (n, 1)) and covariance (jnp.ndarray of shape (n, n)).
        """
        if self.C is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        Phi = self._construct_basis_matrix(
            spatio_temporal_coords, self.beta_indices, "legendre"
        )

        return self._uq_jit(self.C, Phi)

    @staticmethod
    @jax.jit
    def _uq_jit(
        C: jnp.ndarray,
        Phi: jnp.ndarray,
    ) -> tuple:
        """JIT-compiled internal function for UQ using Eq. (22) and (23).

        Args:
            C (jnp.ndarray): Coefficient matrix of shape (Q, P).
            Phi (jnp.ndarray): Basis matrix for spatio-temporal coordinates of shape (n, Q).

        Returns:
            tuple: Mean (jnp.ndarray of shape (n, 1)) and covariance (jnp.ndarray of shape (n, n)).
        """

        # Mean from the first column of C
        c0 = C[:, 0:1]
        mean = Phi @ c0

        # Covariance
        C_prime = C[:, 1:]
        inner_term = C_prime @ C_prime.T
        covariance = Phi @ inner_term @ Phi.T

        return mean, covariance
