import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def solve_pde_split_step(x_array, t_final, A=0.25, m=1, dx=0.1, dt=0.01):
    # Parameters
    nx = len(x_array)
    nt = int(t_final / dt)
    j = 1j  # complex unit

    # Initial conditions
    u = A * np.exp(-x_array ** 2)
    v = -A * np.exp(-x_array ** 2)

    # Wavenumbers for Fourier space
    k = np.fft.fftfreq(nx, d=dx) * 2 * np.pi  # Wavenumbers for FFT
    k = 1j * k  # Fourier space derivative factor

    # To store the solution at each time step (for visualization)
    u_history = []
    v_history = []

    # Time-stepping loop
    for t in range(nt):
        # Store solution every few time steps for visualization
        if t % 100 == 0:  # Adjust this value to store at specific intervals
            u_history.append(np.real(u))  # Store real part of u for plotting

        # Fourier transform of u and v
        u_hat = np.fft.fft(u)
        v_hat = np.fft.fft(v)

        # 1. Linear part (solve in Fourier space for half step)
        u_hat = u_hat * np.exp(-0.5 * dt * k)  # Half step for u_x
        v_hat = v_hat * np.exp(-0.5 * dt * k)  # Half step for v_x

        # Inverse Fourier transform back to physical space
        u = np.fft.ifft(u_hat)
        v = np.fft.ifft(v_hat)

        # 2. Nonlinear part (evolve u and v for full step in real space)
        u_new = u + dt * j * m * v
        v_new = v + dt * j * m * u

        u = u_new
        v = v_new

        # 3. Linear part (solve in Fourier space for another half step)
        u_hat = np.fft.fft(u)
        v_hat = np.fft.fft(v)

        u_hat = u_hat * np.exp(-0.5 * dt * k)  # Half step for u_x
        v_hat = v_hat * np.exp(-0.5 * dt * k)  # Half step for v_x

        # Inverse Fourier transform back to physical space
        u = np.fft.ifft(u_hat)
        v = np.fft.ifft(v_hat)

    # Convert u_history to a 2D array for 3D plotting
    u_history = np.array(u_history)

    return u_history


# Example usage
x_array = np.linspace(-10, 10, 200)  # Spatial grid
t_final = 5.0  # Final time
dt = 0.001  # Time step

# Solve the PDE and get the history of u over time
u_history = solve_pde_split_step(x_array, t_final, dt=dt)

# Prepare the 3D plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Time array (based on the intervals where we stored the solution)
t_array = np.linspace(0, t_final, u_history.shape[0])

# Create a meshgrid for x and t
X, T = np.meshgrid(x_array, t_array)

# Plot the surface
ax.plot_surface(X, T, u_history.imag, cmap='viridis')

# Labels and titles
ax.set_xlabel('x')
ax.set_ylabel('Time')
ax.set_zlabel('u(x,t)')
ax.set_title('3D Plot of u(x,t) over time')

plt.show()
