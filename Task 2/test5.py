import numpy as np
import matplotlib.pyplot as plt

def solve_pde(x_array, t_final, A=0.25, m=1, dx=0.1, dt=0.01):
    # Parameters
    nx = len(x_array)
    nt = int(t_final / dt)
    j = 1j  # complex unit

    # Initial conditions
    u = A * np.exp(-x_array**2)
    v = -A * np.exp(-x_array**2)

    # To store the solution at each time step
    u_sol = np.copy(u)
    v_sol = np.copy(v)

    # Time-stepping loop
    for t in range(nt):
        # Use finite difference for the derivatives
        u_x = np.zeros(nx, dtype=complex)
        v_x = np.zeros(nx, dtype=complex)

        # Central difference in space
        u_x[1:-1] = (u[2:] - u[:-2]) / (2 * dx)
        v_x[1:-1] = (v[2:] - v[:-2]) / (2 * dx)

        # Apply periodic boundary conditions
        u_x[0] = (u[1] - u[-1]) / (2 * dx)
        u_x[-1] = (u[0] - u[-2]) / (2 * dx)
        v_x[0] = (v[1] - v[-1]) / (2 * dx)
        v_x[-1] = (v[0] - v[-2]) / (2 * dx)

        # Update u and v using explicit Euler method
        u_new = u + dt * (-u_x + j * m * v)
        v_new = v + dt * (-v_x - j * m * u)

        # Update for the next time step
        u = u_new
        v = v_new

        # Save the solution for the last time step
        u_sol = u
        v_sol = v

    return u_sol, v_sol

# Example usage
x_array = np.linspace(-10, 10, 200)  # Spatial grid
t_final = 50  # Final time

u_sol, v_sol = solve_pde(x_array, t_final)

# Plot the solution
plt.figure(figsize=(10, 6))
plt.plot(x_array, np.real(u_sol), label='Real part of u')
plt.plot(x_array, np.imag(u_sol), label='Imaginary part of u', linestyle='--')
plt.plot(x_array, np.real(v_sol), label='Real part of v')
plt.plot(x_array, np.imag(v_sol), label='Imaginary part of v', linestyle='--')
plt.legend()
plt.xlabel('x')
plt.ylabel('Solution')
plt.title(f'Solution at t={t_final}')
plt.grid(True)
plt.show()
