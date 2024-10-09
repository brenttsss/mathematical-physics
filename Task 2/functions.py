import numpy as np
from scipy.integrate import quad
from concurrent.futures import ThreadPoolExecutor, as_completed


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
