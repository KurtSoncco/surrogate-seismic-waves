import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


@jax.jit
def _hermite_polynomial(n: int, x: ArrayLike) -> jnp.ndarray:
    """JIT-compiled core logic for the nth physicist's Hermite polynomial."""
    x = jnp.asarray(x)

    def loop_body(k_minus_2, vals):
        h_prev, h_curr = vals
        h_next = 2 * x * h_curr - 2 * (k_minus_2 + 1) * h_prev
        return h_curr, h_next

    def h_n_calc(operand):
        init_vals = (jnp.ones_like(operand), 2 * operand)
        # Loop n-1 times to get from H_1 to H_n
        _, h_n = jax.lax.fori_loop(0, n - 1, loop_body, init_vals)
        return h_n

    branches = [
        lambda op: jnp.ones_like(op),  # n == 0
        lambda op: 2 * op,  # n == 1
        h_n_calc,  # n >= 2
    ]
    return jax.lax.switch(jnp.minimum(n, 2), branches, x)


def hermite_polynomial(n: int, x: ArrayLike) -> jnp.ndarray:
    """Computes the nth physicist's Hermite polynomial at x using a JIT-compiled iterative method.

    This function acts as a user-facing wrapper that performs input validation
    before calling the JIT-compiled core implementation.

    Args:
        n (int): The non-negative integer order of the polynomial. This is
                 treated as a static argument for JIT compilation.
        x (ArrayLike): A scalar or array of points to evaluate the polynomial at.

    Returns:
        jnp.ndarray: The values of the nth Hermite polynomial at x.

    Raises:
        ValueError: If n is not a non-negative integer.
    """
    # Input validation is done outside the JIT-compiled function.
    if not isinstance(n, int) or n < 0:
        raise ValueError("Order 'n' must be a non-negative integer.")

    # Call the efficient, compiled version of the function.
    return _hermite_polynomial(n, x)


@jax.jit
def _legendre_polynomial(n: int, x: ArrayLike) -> jnp.ndarray:
    """JIT-compiled core logic for the nth Legendre polynomial."""
    x = jnp.asarray(x)

    def loop_body(k_minus_1, vals):
        p_prev, p_curr = vals
        k = k_minus_1 + 1
        # Recurrence: (k+1)P_{k+1}(x) = (2k+1)xP_k(x) - kP_{k-1}(x)
        p_next = ((2 * k + 1) * x * p_curr - k * p_prev) / (k + 1)
        return p_curr, p_next

    def p_n_calc(operand):
        # Initial values are P_0(x) and P_1(x)
        init_vals = (jnp.ones_like(operand), operand)
        # Loop n-1 times to get from P_1 to P_n
        _, p_n = jax.lax.fori_loop(0, n - 1, loop_body, init_vals)
        return p_n

    branches = [
        lambda op: jnp.ones_like(op),  # Case n=0
        lambda op: op,  # Case n=1
        p_n_calc,  # Case n>=2
    ]
    return jax.lax.switch(jnp.minimum(n, 2), branches, x)


def legendre_polynomial(n: int, x: ArrayLike) -> jnp.ndarray:
    """Computes the nth Legendre polynomial P_n(x) using a JIT-compiled recurrence relation.

    This function acts as a user-facing wrapper that performs input validation
    before calling the JIT-compiled core implementation.

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
    if not isinstance(n, int) or n < 0:
        raise ValueError("Order 'n' must be a non-negative integer.")

    return _legendre_polynomial(n, x)
