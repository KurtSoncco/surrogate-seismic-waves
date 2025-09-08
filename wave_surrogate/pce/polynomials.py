import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


@jax.jit
def hermite_polynomial(n: int, x: ArrayLike) -> jnp.ndarray:
    """Computes the nth physicist's Hermite polynomial at x using an efficient iterative method.

    This approach is significantly more performant and robust than a naive
    recursive implementation, avoiding recursion depth limits and redundant calculations.

    Args:
        n (int): The non-negative integer order of the polynomial.
        x (ArrayLike): A scalar or array of points to evaluate the polynomial at.

    Returns:
        jnp.ndarray: The values of the nth Hermite polynomial at x.

    Raises:
        ValueError: If n is not a non-negative integer.
    """
    # Ensure x is a JAX array for consistent operations
    x = jnp.asarray(x)

    # Input validation
    if not isinstance(n, int) or n < 0:
        raise ValueError("Order 'n' must be a non-negative integer.")

    # Handle the base cases H_0(x) = 1 and H_1(x) = 2x
    if n == 0:
        return jnp.ones_like(x)
    if n == 1:
        return 2 * x

    # Initialize with H_0(x) and H_1(x) for the loop
    h_prev = jnp.ones_like(x)  # This will hold H_{k-2}
    h_curr = 2 * x  # This will hold H_{k-1}

    # Iterate from k=2 up to n to compute H_n(x)
    for k in range(2, n + 1):
        # Apply the correct recurrence relation:
        # H_k(x) = 2x * H_{k-1}(x) - 2(k-1) * H_{k-2}(x)
        h_next = 2 * x * h_curr - 2 * (k - 1) * h_prev

        # Update the previous two values for the next iteration
        h_prev, h_curr = h_curr, h_next

    return h_curr


@jax.jit
def legendre_polynomial(n: int, x: ArrayLike) -> jnp.ndarray:
    """Computes the nth Legendre polynomial P_n(x) using a JIT-compiled recurrence relation.

    This pure JAX implementation is efficient, differentiable, and runs on accelerators.
    It uses Bonnet's recurrence relation:
    (k+1) * P_{k+1}(x) = (2k+1) * x * P_k(x) - k * P_{k-1}(x)

    Args:
        n (int): The non-negative integer order of the polynomial. This is
                 treated as a static argument for JIT compilation.
        x (ArrayLike): A scalar or array of points to evaluate the polynomial at.
                       The domain is typically [-1, 1].

    Returns:
        jnp.ndarray: The values of the nth Legendre polynomial.

    Raises:
        ValueError: If n is not a non-negative integer.
    """
    x = jnp.asarray(x)
    if not isinstance(n, int) or n < 0:
        raise ValueError("Order 'n' must be a non-negative integer.")

    # Base cases: P_0(x) = 1 and P_1(x) = x
    if n == 0:
        return jnp.ones_like(x)
    if n == 1:
        return x

    # Initialize the recurrence with P_0(x) and P_1(x)
    p_prev = jnp.ones_like(x)  # Represents P_{k-1}
    p_curr = x  # Represents P_k

    # Iterate from k=1 up to n-1 to find P_n(x)
    for k in range(1, n):
        # The recurrence relation calculates P_{k+1} from the previous two
        p_next = ((2 * k + 1) * x * p_curr - k * p_prev) / (k + 1)

        # Update the values for the next iteration
        p_prev, p_curr = p_curr, p_next

    return p_curr
