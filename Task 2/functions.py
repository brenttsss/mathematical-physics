import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# Define F, alpha, and beta (vectorized versions)
def F(k, A):
    return (A / (2 * np.sqrt(np.pi))) * np.exp(-k ** 2 / 4)


def alpha(k, m):
    return np.sqrt(k ** 2 + m ** 2)


def beta(k, F_k, m):
    return F_k * (k - m)


# Define the integrand for v(x,t)
def integrand_v(k, x, t, A, m):
    F_k = F(k, A)
    alpha_k = alpha(k, m)
    beta_k = beta(k, F_k, m)
    term1 = ((F_k * alpha_k - beta_k) / (2 * alpha_k)) * np.exp(-1j * alpha_k * t) * (-(k + alpha_k) / m)
    term2 = ((F_k * alpha_k + beta_k) / (2 * alpha_k)) * np.exp(1j * alpha_k * t) * (-(k - alpha_k) / m)
    return (term1 + term2) * np.exp(-1j * k * x)


# Define the integrand for u(x,t)
def integrand_u(k, x, t, A, m):
    F_k = F(k, A)
    alpha_k = alpha(k, m)
    beta_k = beta(k, F_k, m)
    term1 = ((F_k * alpha_k - beta_k) / (2 * alpha_k)) * np.exp(-1j * alpha_k * t)
    term2 = ((F_k * alpha_k + beta_k) / (2 * alpha_k)) * np.exp(1j * alpha_k * t)
    return (term1 + term2) * np.exp(-1j * k * x)


# Numerical integration for v(x, t) and u(x, t)
def compute_integral(integrand, x, t, A, m):
    real_part = quad(lambda k: np.real(integrand(k, x, t, A, m)), -np.inf, np.inf, limit=100)[0]
    imag_part = quad(lambda k: np.imag(integrand(k, x, t, A, m)), -np.inf, np.inf, limit=100)[0]
    return real_part + 1j * imag_part


# Modified to handle array inputs for x
def numerical_integration_v(x, t, A, m):
    results = [compute_integral(integrand_v, xi, t, A, m) for xi in x]
    return np.array(results)


def numerical_integration_u(x, t, A, m):
    results = [compute_integral(integrand_u, xi, t, A, m) for xi in x]
    return np.array(results)

def finite_difference_method(L, T, A, m, Nx=200):
    dx = 2 * L / (Nx - 1)  # Spatial step size
    dt = 0.1 * dx / np.sqrt(2)

    # Discretize space and time
    x = np.linspace(-L, L, Nx)
    t = np.arange(0, T + dt, dt)
    Nt = len(t)

    # Initial conditions
    u = np.zeros((Nt, Nx), dtype=complex)
    v = np.zeros((Nt, Nx), dtype=complex)

    u[0, :] = A * np.exp(-x ** 2)
    v[0, :] = -A * np.exp(-x ** 2)

    # Finite difference method
    for n in range(0, Nt - 1):
        for i in range(1, Nx - 1):
            # Central difference for space, forward difference for time
            u[n + 1, i] = u[n, i] + dt * (- (u[n, i + 1] - u[n, i - 1]) / (2 * dx) + 1j * m * v[n, i])
            v[n + 1, i] = v[n, i] + dt * ((v[n, i + 1] - v[n, i - 1]) / (2 * dx) + 1j * m * u[n, i])

        # Periodic boundary conditions
        u[n + 1, 0] = u[n + 1, -2]  # u(x=-L) = u(x=L)
        u[n + 1, -1] = u[n + 1, 1]
        v[n + 1, 0] = v[n + 1, -2]  # v(x=-L) = v(x=L)
        v[n + 1, -1] = v[n + 1, 1]

    return u, v

# Example usage
if __name__ == "__main__":
    x = np.linspace(-10, 10, 200)  # Example input
    t = 12
    A = 0.25
    m = 1.0

    result_v = numerical_integration_v(x, t, A, m)
    result_u = numerical_integration_u(x, t, A, m)

    # finite_u = finite_difference_method_2(10, t)[0][-1, :]
    finite_v = finite_difference_method(10, t, A, m)[1][-1, :]

    plt.plot(x, np.real(result_v))
    plt.plot(x, np.real(finite_v.T))
    plt.show()
