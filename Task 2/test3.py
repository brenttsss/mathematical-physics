import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def solve_pde(x_array, t_final, A=0.25, m=1, dx=0.1, dt=0.01):
    # Parameters
    nx = len(x_array)
    nt = int(t_final / dt)
    j = 1j  # complex unit

    # Initial conditions
    u = A * np.exp(-x_array ** 2)
    v = -A * np.exp(-x_array ** 2)

    # To store the solution at each time step for plotting
    u_solutions = np.zeros((nt, nx), dtype=complex)
    v_solutions = np.zeros((nt, nx), dtype=complex)

    # Store initial condition
    u_solutions[0, :] = u
    v_solutions[0, :] = v

    # Time-stepping loop
    for t in range(1, nt):
        # Use finite difference for the derivatives
        u_x = np.zeros(nx, dtype=complex)
        v_x = np.zeros(nx, dtype=complex)

        # Central difference in space (with no periodic boundary conditions)
        u_x[1:-1] = (u[2:] - u[:-2]) / (2 * dx)
        v_x[1:-1] = (v[2:] - v[:-2]) / (2 * dx)

        # Boundary conditions (Dirichlet: keep boundary values constant)
        u_x[0] = u_x[-1] = 0
        v_x[0] = v_x[-1] = 0

        # Update u and v using explicit Euler method
        u_new = u + dt * (-u_x + j * m * v)
        v_new = v + dt * (-v_x - j * m * u)

        # Update for the next time step
        u = u_new
        v = v_new

        # Save the solution at this time step
        u_solutions[t, :] = u
        v_solutions[t, :] = v

    return u_solutions, v_solutions, np.linspace(0, t_final, nt)


# Example usage
x_array = np.linspace(-10, 10, 200)  # Spatial grid
t_final = 5  # Final time

u_solutions, v_solutions, t_values = solve_pde(x_array, t_final)

# 3D plot for the real part of u
X, T = np.meshgrid(x_array, t_values)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface for the real part of u
ax.plot_surface(X, T, np.real(u_solutions), cmap='viridis')

ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('Real part of u')
ax.set_title('Evolution of the real part of u over time')
plt.show()

# 3D plot for the real part of v
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface for the real part of v
ax.plot_surface(X, T, np.real(v_solutions), cmap='plasma')

ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('Real part of v')
ax.set_title('Evolution of the real part of v over time')
plt.show()
