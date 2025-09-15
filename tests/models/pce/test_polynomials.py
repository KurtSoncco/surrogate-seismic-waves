import jax.numpy as jnp
import pytest

from wave_surrogate.models.pce.polynomials import (
    hermite_polynomial,
    legendre_polynomial,
)


@pytest.mark.parametrize(
    "order, x_values, expected",
    [
        (0, [-1.0, -0.5, 0.0, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0]),
        (1, [-1.0, -0.5, 0.0, 0.5, 1.0], [-2.0, -1.0, 0.0, 1.0, 2.0]),
        (2, [-1.0, -0.5, 0.0, 0.5, 1.0], [2.0, -1.0, -2.0, -1.0, 2.0]),
        (3, [-1.0, -0.5, 0.0, 0.5, 1.0], [4.0, 5.0, 0.0, -5.0, -4.0]),
        (4, [-1.0, -0.5, 0.0, 0.5, 1.0], [-20.0, 1.0, 12.0, 1.0, -20.0]),
    ],
)
def test_hermite_polynomial(order, x_values, expected):
    """Tests the Hermite polynomial implementation."""
    x = jnp.array(x_values)
    assert jnp.allclose(hermite_polynomial(order, x), jnp.array(expected), atol=1e-6)


@pytest.mark.parametrize(
    "order, x_values, expected",
    [
        (0, [-1.0, -0.5, 0.0, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0]),
        (1, [-1.0, -0.5, 0.0, 0.5, 1.0], [-1.0, -0.5, 0.0, 0.5, 1.0]),
        (
            2,
            [-1.0, -0.5, 0.0, 0.5, 1.0],
            [1.0, -0.125, -0.5, -0.125, 1.0],
        ),
        (
            3,
            [-1.0, -0.5, 0.0, 0.5, 1.0],
            [-1.0, 0.4375, 0.0, -0.4375, 1.0],
        ),
        (
            4,
            [-1.0, -0.5, 0.0, 0.5, 1.0],
            [1.0, -0.2890625, 0.375, -0.2890625, 1.0],
        ),
    ],
)
def test_legendre_polynomial(order, x_values, expected):
    """Tests the Legendre polynomial implementation."""
    x = jnp.array(x_values)
    assert jnp.allclose(legendre_polynomial(order, x), jnp.array(expected), atol=1e-6)
