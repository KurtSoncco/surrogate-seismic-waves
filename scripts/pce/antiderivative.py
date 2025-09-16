from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from metrax import MAE, MSE, RMSE, RSQUARED

from wave_surrogate.logging_setup import setup_logging
from wave_surrogate.models.pce.pce_jax import PCEOperatorJAX
from wave_surrogate.models.pce.polynomials import _legendre_polynomial
from wave_surrogate.plot_utils.radar_graph import radar_graph

logger = setup_logging()
# --- 1. Settings from the Paper (e.g., Anti-derivative example) ---
p_order = 3
q_order = 10
r_dim = 6
d_dim = 1

# --- 2. Create Mock Data using JAX NumPy ---
n = 101  # Spatio-temporal points
N_train = 100  # Training samples
N_test = 1000  # Testing samples

# Use a JAX key for reproducible random numbers
key = jax.random.PRNGKey(0)
key, S_key, xi_train_key, xi_test_key = jax.random.split(key, 4)
xi_train = jax.random.normal(xi_train_key, (N_train, r_dim))
xi_test = jax.random.normal(xi_test_key, (N_test, r_dim))
x = jnp.linspace(0, 1, n).reshape(-1, 1)  # Spatio-temporal coordinates
x_scaled = 2 * x - 1  # Scale to [-1, 1]


# --- 3. Generate Stochastic Field and Ground Truth Solutions ---
# Gaussian Process
def rbf_kernel(x1, x2, sigma=1.0, length_scale=0.2):
    """Radial Basis Function (RBF) kernel."""
    return sigma**2 * jnp.exp(-jnp.sum((x1 - x2) ** 2) / (2 * length_scale**2))


# Vectorize the kernel function to compute the full covariance matrix
K = jax.vmap(lambda x1: jax.vmap(lambda x2: rbf_kernel(x1, x2))(x))(x)
# Add a small nugget for numerical stability
K += 1e-6 * jnp.eye(n)

# --- Karhunen-Loève Expansion ---
# 1. Decompose the kernel to get eigenvalues and eigenfunctions
eigenvalues, eigenvectors = jnp.linalg.eigh(K)

# eigh returns them in ascending order, so we reverse them for descending order
eigenvalues = eigenvalues[::-1]
eigenvectors = eigenvectors[:, ::-1]

# 2. Select the top `r_dim` components
lambda_k = eigenvalues[:r_dim]
phi_k = eigenvectors[:, :r_dim]

# 3. Construct u(x) from xi using the KL expansion formula
# u(x) = sum_{k=1 to r} sqrt(lambda_k) * phi_k(x) * xi_k
# Matrix form: u = xi @ (phi_k * sqrt(lambda_k))^T
kl_basis = phi_k * jnp.sqrt(lambda_k)  # Shape (n, r_dim)
u_train = xi_train @ kl_basis.T  # (N_train, r_dim) @ (r_dim, n) -> (N_train, n)
u_test = xi_test @ kl_basis.T  # (N_test, r_dim) @ (r_dim, n) -> (N_test, n)

# Generate Ground Truth Solutions
dx = x[1] - x[0]
S_train_true = jnp.cumsum(u_train, axis=1) * dx  # Anti-derivative, shape (N_train, n)
S_test_true = jnp.cumsum(u_test, axis=1) * dx  # Anti-derivative, shape (N_test, n)

S_train_true_model = S_train_true.T  # Shape (n, N_train)

# --- 4.  Data-Driven PCE ---
pce_data_driven = PCEOperatorJAX(p_order, q_order, r_dim, d_dim)
pce_data_driven.fit(S_train_true_model, x_scaled, xi_train)
S_pred_data_driven = pce_data_driven.predict(x_scaled, xi_test).T  # Shape (N_test, n)


# --- 5. Physics-Informed PCE ---

# --- FIX: First, fit a PCE model to the forcing term u(x) ---
# This gives us the coefficients of u(x) in the stochastic basis.
pce_u = PCEOperatorJAX(p_order, q_order, r_dim, d_dim)
# u_train has shape (N_train, n), we need (n, N_train) for the fit method.
pce_u.fit(u_train.T, x_scaled, xi_train)
C_u = pce_u.C  # These are the coefficients we need to match.


def pde_operator_fn(points, beta_indices, xi_samples):
    """Computes the L_Phi matrix for the ds/dx operator."""

    def evaluate_single_phi(point, index_tuple):
        return jnp.prod(jax.vmap(_legendre_polynomial)(jnp.array(index_tuple), point))

    jac_phi = jax.jacfwd(evaluate_single_phi)
    vmap_jac = jax.vmap(jax.vmap(jac_phi, in_axes=(None, 0)), in_axes=(0, None))
    L_Phi_unscaled = vmap_jac(points, beta_indices).squeeze()
    return 2.0 * L_Phi_unscaled  # Apply chain rule


def antiderivative_bc_fn(points, beta_indices, xi_samples):
    # BC: s(0) = 0
    def evaluate_single_phi(point, index_tuple):
        evals_1d = jax.vmap(_legendre_polynomial)(jnp.array(index_tuple), point)
        return jnp.prod(evals_1d)

    vmap_phi_over_indices = jax.vmap(evaluate_single_phi, in_axes=(None, 0))
    Phi_bc = jax.vmap(vmap_phi_over_indices, in_axes=(0, None))(
        points, beta_indices
    ).squeeze()

    # Ensure Phi_bc is a 2D array, even with a single point
    if Phi_bc.ndim == 1:
        Phi_bc = Phi_bc.reshape(1, -1)

    # The forcing term is zero, with shape (num_bc_points, N_train)
    F_bc = jnp.zeros((points.shape[0], xi_samples.shape[0]))
    return Phi_bc, F_bc


# Collocation points
pce_physics = PCEOperatorJAX(p_order, q_order, r_dim, d_dim)


def antiderivative_pde_fn(points, beta_indices, xi_samples):
    """
    Defines the PDE residual for dS/dx = u(x).
    Returns the operator matrix L_Phi and the forcing term F_pde.
    """
    # The operator L_Phi is the derivative of the basis functions
    L_Phi = pde_operator_fn(points, beta_indices, xi_samples)

    # The forcing term F_pde is now the coefficients of u(x), C_u.
    # The fit function expects a spatio-temporal field, so we reconstruct
    # u(x) from its coefficients.
    Phi_pde = pce_u._construct_basis_matrix(points, beta_indices, "legendre")
    Psi_pde = pce_u._construct_basis_matrix(
        xi_samples, pce_u.alpha_indices, "hermite"
    ).T
    F_pde = Phi_pde @ C_u @ Psi_pde

    return L_Phi, F_pde


pde_points = x_scaled
bc_points = jnp.array([[-1.0]])  # s(0) = 0
ic_points = jnp.empty((0, 1))  # No initial condition needed


def ic_fn(points, beta_indices, xi_samples):
    """
    Defines the initial condition residual.
    Since there is no IC, this function returns empty matrices with consistent shapes.
    """
    # --- FIX ---
    # The operator L_Phi_ic must have the same number of columns as L_Phi_pde.
    # This number is the size of the spatial basis, P, which is the
    # number of rows in the beta_indices array.
    num_spatial_basis_funcs = beta_indices.shape[0]  # This will be 11

    # Operator matrix with 0 rows (no IC points) and P columns.
    L_ic = jnp.empty((0, num_spatial_basis_funcs))

    # Forcing term with 0 rows and N_train columns.
    F_ic = jnp.empty((0, xi_samples.shape[0]))

    return L_ic, F_ic


pce_physics.fit_physics_informed(
    pde_fn=antiderivative_pde_fn,
    pde_collocation_points=pde_points,
    bc_fn=antiderivative_bc_fn,
    bc_collocation_points=bc_points,
    ic_fn=ic_fn,
    ic_collocation_points=ic_points,
    xi_samples=xi_train,
)
S_pred_physics = pce_physics.predict(x_scaled, xi_test).T  # Shape (N_test, n)


# --- 6. Evaluate and Plot Results ---
mse_data_driven = MSE.from_model_output(
    predictions=S_pred_data_driven, labels=S_test_true
)
mse_physics = MSE.from_model_output(predictions=S_pred_physics, labels=S_test_true)
mae_data_driven = MAE.from_model_output(
    predictions=S_pred_data_driven, labels=S_test_true
)
mae_physics = MAE.from_model_output(predictions=S_pred_physics, labels=S_test_true)
rmse_data_driven = RMSE.from_model_output(
    predictions=S_pred_data_driven, labels=S_test_true
)
rmse_physics = RMSE.from_model_output(predictions=S_pred_physics, labels=S_test_true)
r2_data_driven = RSQUARED.from_model_output(
    predictions=S_pred_data_driven, labels=S_test_true
)
r2_physics = RSQUARED.from_model_output(predictions=S_pred_physics, labels=S_test_true)

## Save into dictionary for radar plot
results = {
    "Data-Driven": {
        "MSE": float(mse_data_driven.compute()),
        "MAE": float(mae_data_driven.compute()),
        "RMSE": float(rmse_data_driven.compute()),
        "R2": float(r2_data_driven.compute()),
    },
    "Physics-Informed": {
        "MSE": float(mse_physics.compute()),
        "MAE": float(mae_physics.compute()),
        "RMSE": float(rmse_physics.compute()),
        "R2": float(r2_physics.compute()),
    },
}
logger.info(f"✅ Evaluation Results: {results}")

fig_path = Path(__file__).parent / "radar_plot_antiderivative.png"
radar_graph(
    results,
    title="Anti-Derivative Problem: Data-Driven vs Physics-Informed PCE",
    save_path=str(fig_path),
)

## Save a few examples for plotting (this part remains the same)
num_examples_to_save = 3
example_indices = jnp.linspace(0, N_test - 1, num_examples_to_save, dtype=jnp.int32)

x_coords = np.array(x)
true_examples = np.array(S_test_true[example_indices, :])
dd_pred_examples = np.array(S_pred_data_driven[example_indices, :])
pi_pred_examples = np.array(S_pred_physics[example_indices, :])

output_filename = Path(__file__).parent / "pce_antiderivative_results.npz"
np.savez(
    output_filename,
    x=x_coords,
    s_true=true_examples,
    s_pred_data_driven=dd_pred_examples,
    s_pred_physics=pi_pred_examples,
)
logger.info(
    f"✅ Saved {num_examples_to_save} examples for plotting to '{output_filename}'"
)
