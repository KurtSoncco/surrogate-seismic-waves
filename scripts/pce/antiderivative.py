import jax
import jax.numpy as jnp

from wave_surrogate.logging_setup import setup_logging
from wave_surrogate.pce.pce_jax import PCEOperatorJAX

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

S_train_true = jax.random.normal(S_key, (n, N_train))
xi_train = jax.random.normal(xi_train_key, (N_train, r_dim))
coords = jnp.linspace(-1, 1, n).reshape(
    -1, 1
)  # Legendre polynomials defined on [-1, 1]

print(f"Shape of coords: {coords.shape}")
print(f"Shape of S_train_true: {S_train_true.shape}")


xi_test = jax.random.normal(xi_test_key, (N_test, r_dim))

# --- 3. Initialize and Train the JAX Model ---
pce_model = PCEOperatorJAX(p_order, q_order, r_dim, d_dim)
pce_model.fit(S_train_true, coords, xi_train)
# The first run will be slower due to JIT compilation.
# Subsequent calls to .fit (if any) would be instantaneous.

# --- 4. Make Predictions ---
logger.info("\nRunning prediction...")
S_predicted = pce_model.predict(coords, xi_test)
S_predicted.block_until_ready()  # Wait for prediction to finish
logger.info(f"Predicted solutions shape: {S_predicted.shape}")

# --- 5. Perform Uncertainty Quantification ---
logger.info("\nRunning UQ...")
mean_pred, cov_pred = pce_model.quantify_uncertainty(coords)
std_dev_pred = jnp.sqrt(jnp.diag(cov_pred))
mean_pred.block_until_ready()  # Wait for UQ to finish

logger.info(f"Mean prediction shape: {mean_pred.shape}")
logger.info(f"Standard deviation prediction shape: {std_dev_pred.shape}")
