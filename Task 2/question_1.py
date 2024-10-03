import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Constants
A = 0.5
pi = np.pi
m = 1  # You can adjust m as needed

# Define F, alpha, and beta
def F(k):
    return (A / (2 * np.sqrt(pi))) * np.exp(-k**2 / 4)

def alpha(k):
    return np.sqrt(k**2 + m**2)

def beta(k):
    return F(k) * (k - m)

# Define the integrand for v(x,t)
def integrand_v(k, x, t):
    alpha_k = alpha(k)
    beta_k = beta(k)
    term1 = ((F(k) * alpha_k - beta_k) / (2 * alpha_k)) * np.exp(-1j * alpha_k * t) * (-(k + alpha_k) / m)
    term2 = ((F(k) * alpha_k + beta_k) / (2 * alpha_k)) * np.exp(1j * alpha_k * t) * (-(k - alpha_k) / m)
    return (term1 + term2) * np.exp(-1j * k * x)

# Define the integrand for u(x,t)
def integrand_u(k, x, t):
    alpha_k = alpha(k)
    beta_k = beta(k)
    term1 = ((F(k) * alpha_k - beta_k) / (2 * alpha_k)) * np.exp(-1j * alpha_k * t)
    term2 = ((F(k) * alpha_k + beta_k) / (2 * alpha_k)) * np.exp(1j * alpha_k * t)
    return (term1 + term2) * np.exp(-1j * k * x)

# Numerical integration for v(x, t) and u(x, t) for different x values
def compute_v(x, t):
    real_part = quad(lambda k: np.real(integrand_v(k, x, t)), -np.inf, np.inf)[0]
    imag_part = quad(lambda k: np.imag(integrand_v(k, x, t)), -np.inf, np.inf)[0]
    return real_part + 1j * imag_part

def compute_u(x, t):
    real_part = quad(lambda k: np.real(integrand_u(k, x, t)), -np.inf, np.inf)[0]
    imag_part = quad(lambda k: np.imag(integrand_u(k, x, t)), -np.inf, np.inf)[0]
    return real_part + 1j * imag_part

# Generate x values
x_values = np.linspace(-10, 10, 200)

# Generate time values
t_values = np.linspace(0, 5, 50)

# Create meshgrid for x and t values
x_values, t_values = np.meshgrid(x_values, t_values)

print('check')

# Compute v(x, t) and u(x, t) for each x and t value
v_values = np.array([[compute_v(x, t) for x in x_values[0]] for t in t_values[:, 0]])
u_values = np.array([[compute_u(x, t) for x in x_values[0]] for t in t_values[:, 0]])

# Plot the real part of v(x, t) in 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x_values, t_values, v_values.real, cmap='viridis')

ax.set_title('Real Part of $v(x,t)$')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$t$')
ax.set_zlabel(r'Real Part of $v(x,t)$')
plt.show()

# Plot the real part of u(x, t) in 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x_values, t_values, u_values.real, cmap='viridis')

ax.set_title('Real Part of $u(x,t)$')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$t$')
ax.set_zlabel(r'Real Part of $u(x,t)$')
plt.show()

# # Plot the real and imaginary parts of the solution for v(x, t)
# plt.figure(figsize=(10, 6))
# plt.plot(x_values, np.abs(v_values), label="Real part of v(x,t)", color='b')
# plt.title(r'Solution $v(x,t)$ of the Integral')
# plt.xlabel(r'$x$')
# plt.ylabel(r'$v(x,t)$')
# plt.legend()
# plt.grid(True)
# plt.show()
#
# # Plot the real and imaginary parts of the solution for u(x, t)
# plt.figure(figsize=(10, 6))
# plt.plot(x_values, np.abs(u_values), label="Real part of u(x,t)", color='b')
# plt.title(r'Solution $u(x,t)$ of the Integral')
# plt.xlabel(r'$x$')
# plt.ylabel(r'$u(x,t)$')
# plt.legend()
# plt.grid(True)
# plt.show()
