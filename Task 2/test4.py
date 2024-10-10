import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftfreq

# Constants
A = 0.25
m = 1
j = 1j  # Complex unit


# Define the initial conditions
def initial_conditions(x):
    u0 = A * np.exp(-x ** 2)
    v0 = -A * np.exp(-x ** 2)
    return u0, v0


# Define the solver
def solve_uv(x, t):
    # Number of points
    N = len(x)
    L = x[-1] - x[0]  # Assuming x is evenly spaced
    k = fftfreq(N, d=(x[1] - x[0])) * 2 * np.pi  # Fourier wave numbers

    # Initial conditions
    u0, v0 = initial_conditions(x)

    # Fourier transform of the initial conditions
    u0_hat = fft(u0)
    v0_hat = fft(v0)

    # Time evolution in Fourier space
    omega = np.sqrt(k ** 2 + m ** 2)  # Dispersion relation for the system

    # Solve for Fourier coefficients at time t
    u_hat_t = u0_hat * np.cos(omega * t) + v0_hat * np.sin(omega * t) * (j * m)
    v_hat_t = v0_hat * np.cos(omega * t) + u0_hat * np.sin(omega * t) * (j * m)

    # Inverse Fourier transform to get solution in real space
    u_t = ifft(u_hat_t)
    v_t = ifft(v_hat_t)

    return u_t, v_t


# Spatial grid
x = np.linspace(-10, 10, 1000)

# Time values for plotting
t_values = [0, 5, 10, 20, 30]

# Plotting the solution for different values of t
plt.figure(figsize=(10, 8))

for t in t_values:
    u_t, v_t = solve_uv(x, t)
    plt.plot(x, u_t.real, label=f'u(x, t={t})')
    plt.plot(x, v_t.real, '--', label=f'v(x, t={t})')

plt.title("Solution of the system for different times t")
plt.xlabel("x")
plt.ylabel("u(x, t) and v(x, t)")
plt.legend()
plt.grid(True)
plt.show()
