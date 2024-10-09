import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functions import *

A = 0.25
m = 1

def alpha_f(k):
    return np.sqrt(k**2 + m**2)

def beta_f(k):
    return F_f(k) * k + G_f(k) * m

def F_f(k):
    return A / (2 * np.sqrt(np.pi)) * np.exp(-k ** 2 / 4)

def G_f(k):
    return - A / (2 * np.sqrt(np.pi)) * np.exp(-k ** 2 / 4)

def g_1_f(k):
    return ((F_f(k) * alpha_f(k) - beta_f(k)) / (2 * alpha_f(k))) * (- (k + alpha_f(k)) / m)

def g_2_f(k):
    return ((F_f(k) * alpha_f(k) + beta_f(k)) / (2 * alpha_f(k))) * (-(k - alpha_f(k)) / m)

def g_3_f(k):
    return (F_f(k) * alpha_f(k) - beta_f(k)) / (2 * alpha_f(k))

def g_4_f(k):
    return (F_f(k) * alpha_f(k) + beta_f(k)) / (2 * alpha_f(k))

def h_1_f(k, x, t):
    return - alpha_f(k) - k * (x/t)

def h_2_f(k, x, t):
    return alpha_f(k) - k * (x/t)

def dd_h_1_f(k):
    return - m ** 2 / (k ** 2 + m ** 2) ** (3 / 2)

def dd_h_2_f(k):
    return m ** 2 / (k ** 2 + m ** 2) ** (3 / 2)

def k_0_f(x, t):
    return - (m*x) / np.sqrt(t**2 - x**2)

def v_f(k, x, t):

    factor_1 = g_1_f(k) * np.sqrt(2 * np.pi / (np.abs(dd_h_1_f(k)) * t))
    factor_2 = np.exp(1j * (t * h_1_f(k, x, t) - (np.pi/4)))
    term_1 = factor_1 * factor_2

    factor_3 = g_2_f(k) * np.sqrt(2 * np.pi / (np.abs(dd_h_2_f(k)) * t))
    factor_4 = np.exp(1j * (t * h_2_f(k, x, t) + (np.pi/4)))
    term_2 = factor_3 * factor_4

    return term_1 + term_2

def u_f(k, x, t):

    factor_1 = g_3_f(k) * np.sqrt(2 * np.pi / (np.abs(dd_h_1_f(k)) * t))
    factor_2 = np.exp(1j * (t * h_1_f(k, x, t) - (np.pi/4)))
    term_1 = factor_1 * factor_2

    factor_3 = g_4_f(k) * np.sqrt(2 * np.pi / (np.abs(dd_h_2_f(k)) * t))
    factor_4 = np.exp(1j * (t * h_2_f(k, x, t) + (np.pi/4)))
    term_2 = factor_3 * factor_4

    return term_1 + term_2

def compute_integral(integrand, x, t):
    if integrand == v_f:
        k_0 = k_0_f(x, t)
        return integrand(k_0, x, t)
    elif integrand == u_f:
        k_0 = k_0_f(x, t)
        return integrand(k_0, x, t)


def compute_integral(integrand, x, t):
    if integrand == v_f:
        k_0 = k_0_f(x, t)
        return integrand(k_0, x, t)
    elif integrand == u_f:
        k_0 = k_0_f(x, t)
        return integrand(k_0, x, t)


# Time values
time_values = np.arange(15, 51)  # From 15 to 50 inclusive

# Generate x values
x_values = np.linspace(-10, 10, 100)

# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Initialize empty plots for u(x,t) and v(x,t)
u_line_asymp, = ax1.plot([], [], label='Asymptotic expansion', color='blue')
u_line_numerical, = ax1.plot([], [], label='Numerical integration', color='orange')
v_line_asymp, = ax2.plot([], [], label='Asymptotic expansion', color='blue')
v_line_numerical, = ax2.plot([], [], label='Numerical integration', color='orange')

# Set up plot limits and labels
ax1.set_xlim(x_values[0], x_values[-1])
ax1.set_ylim(-A / 5 - A, A / 5 + A)
ax1.set_title(f'Real Part of $u(x,t)$')
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$\text{Re}\left\{u(x,t)\right\}$')
ax1.legend()

ax2.set_xlim(x_values[0], x_values[-1])
ax2.set_ylim(-A / 5 - A, A / 5 + A)
ax2.set_title(f'Real Part of $v(x,t)$')
ax2.set_xlabel(r'$x$')
ax2.set_ylabel(r'$\text{Re}\left\{v(x,t)\right\}$')
ax2.legend()


# Update function for each frame in the animation
def update(t):
    u_values = compute_integral(u_f, x_values, t)
    u_values_numerical = numerical_integration_u(x_values, t, A, m)
    v_values = compute_integral(v_f, x_values, t)
    v_values_numerical = numerical_integration_v(x_values, t, A, m)

    u_line_asymp.set_data(x_values, u_values.real)
    u_line_numerical.set_data(x_values, u_values_numerical.real)
    v_line_asymp.set_data(x_values, v_values.real)
    v_line_numerical.set_data(x_values, v_values_numerical.real)

    ax1.set_title(f'Real Part of $u(x,t)$, t={t}')
    ax2.set_title(f'Real Part of $v(x,t)$, t={t}')

    return u_line_asymp, u_line_numerical, v_line_asymp, v_line_numerical


# Create the animation
ani = FuncAnimation(fig, update, frames=time_values, blit=True, repeat=False)

# Save the animation as a gif using Pillow
ani.save('animation.gif', writer='pillow', fps=10)

# Show the animation
plt.show()