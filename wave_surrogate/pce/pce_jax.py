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
        logger.info("Compiling and running the fit function...")
        self.C = self._fit_jit(
            S_true,
            spatio_temporal_coords,
            xi_samples,
            self.beta_indices,
            self.alpha_indices,
        )
        self.C.block_until_ready()  # Wait for computation to finish
        logger.info("Fit complete. Coefficient matrix C has been learned.")

    @staticmethod
    @jax.jit
    def _fit_jit(
        S_true: jnp.ndarray,
        spatio_temporal_coords: jnp.ndarray,
        xi_samples: jnp.ndarray,
        beta_indices: jnp.ndarray,
        alpha_indices: jnp.ndarray,
    ) -> jnp.ndarray:
        """JIT-compiled internal function to solve for C* from Eq. (14).
        Phi and Psi are constructed using the provided multi-indices.

        Args:
            S_true (jnp.ndarray): True solution data of shape (n, N).
            spatio_temporal_coords (jnp.ndarray): Spatio-temporal coordinates of shape (n, d+1).
            xi_samples (jnp.ndarray): Stochastic input samples of shape (N, r).
            beta_indices (jnp.ndarray): Multi-indices for spatio-temporal basis.
            alpha_indices (jnp.ndarray): Multi-indices for stochastic basis.

        Returns:
            jnp.ndarray: The learned coefficient matrix C of shape (Q, P).
        Raises: RuntimeError: If the fitting process fails.
        """

        Phi = PCEOperatorJAX._construct_basis_matrix(
            spatio_temporal_coords, beta_indices, "legendre"
        )
        Psi = PCEOperatorJAX._construct_basis_matrix(
            xi_samples, alpha_indices, "hermite"
        ).T  # Note: shape is (P, N)

        PhiT_Phi = Phi.T @ Phi
        Psi_PsiT = Psi @ Psi.T

        Temp = Phi.T @ S_true @ Psi.T

        C_intermediate = jnp.linalg.solve(PhiT_Phi, Temp)
        C_final_T = jnp.linalg.solve(Psi_PsiT, C_intermediate.T)

        return C_final_T.T

    def predict(
        self, spatio_temporal_coords: jnp.ndarray, xi_samples: jnp.ndarray
    ) -> jnp.ndarray:
        """Predicts the solution for new inputs using the learned C matrix.

        Args:
            spatio_temporal_coords (jnp.ndarray): Spatio-temporal coordinates of shape (n, d+1).
            xi_samples (jnp.ndarray): Stochastic input samples of shape (N, r).

        Returns:
            jnp.ndarray: Predicted solution of shape (n, N).
        Raises: RuntimeError: If the model has not been fitted yet.
        """
        if self.C is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        S_hat = self._predict_jit(
            self.C,
            spatio_temporal_coords,
            xi_samples,
            self.beta_indices,
            self.alpha_indices,
        )
        return S_hat

    @staticmethod
    @jax.jit
    def _predict_jit(
        C: jnp.ndarray,
        spatio_temporal_coords: jnp.ndarray,
        xi_samples: jnp.ndarray,
        beta_indices: jnp.ndarray,
        alpha_indices: jnp.ndarray,
    ) -> jnp.ndarray:
        """JIT-compiled internal function for prediction using Eq. (10).
        Args:
            C (jnp.ndarray): Coefficient matrix of shape (Q, P).
            spatio_temporal_coords (jnp.ndarray): Spatio-temporal coordinates of shape (n, d+1).
            xi_samples (jnp.ndarray): Stochastic input samples of shape (N, r).
            beta_indices (jnp.ndarray): Multi-indices for spatio-temporal basis.
            alpha_indices (jnp.ndarray): Multi-indices for stochastic basis.

        Returns:
            jnp.ndarray: Predicted solution of shape (n, N).
        """
        Phi_test = PCEOperatorJAX._construct_basis_matrix(
            spatio_temporal_coords, beta_indices, "legendre"
        )
        Psi_test = PCEOperatorJAX._construct_basis_matrix(
            xi_samples, alpha_indices, "hermite"
        ).T

        return Phi_test @ C @ Psi_test

    def quantify_uncertainty(self, spatio_temporal_coords: jnp.ndarray) -> tuple:
        """Computes mean and covariance. JIT-compiled for speed."""
        if self.C is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        return self._uq_jit(self.C, spatio_temporal_coords, self.beta_indices)

    @staticmethod
    @jax.jit
    def _uq_jit(
        C: jnp.ndarray,
        spatio_temporal_coords: jnp.ndarray,
        beta_indices: jnp.ndarray,
    ) -> tuple:
        """JIT-compiled internal function for UQ using Eq. (22) and (23).

        Args:
            C (jnp.ndarray): Coefficient matrix of shape (Q, P).
            spatio_temporal_coords (jnp.ndarray): Spatio-temporal coordinates of shape (n, d+1).
            beta_indices (jnp.ndarray): Multi-indices for spatio-temporal basis.

        Returns:
            tuple: Mean (jnp.ndarray of shape (n, 1)) and covariance (jnp.ndarray of shape (n, n)).
        """
        Phi = PCEOperatorJAX._construct_basis_matrix(
            spatio_temporal_coords, beta_indices, "legendre"
        )

        # Mean from the first column of C
        c0 = C[:, 0:1]
        mean = Phi @ c0

        # Covariance
        C_prime = C[:, 1:]
        inner_term = C_prime @ C_prime.T
        covariance = Phi @ inner_term @ Phi.T

        return mean, covariance


if __name__ == "__main__":
    # Example usage and simple test cases
    x_values = jnp.array(
        [[-1.0, 0, 1.0], [-0.5, 0, 0.5], [0.0, 0, 0.0], [0.5, 0, 0.5], [1.0, 0, 1.0]]
    )

    model = PCEOperatorJAX(p_order=2, q_order=2, r_dim=2, d_dim=1)

    model._construct_basis_matrix(
        x_values.reshape(-1, 1), jnp.array([[0], [1], [2]]), "legendre"
    )
