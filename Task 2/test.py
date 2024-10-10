import numpy as np
import matplotlib.pyplot as plt

def solve_pde_split_step(x_array, t_final, A=0.25, m=1, dx=0.1, dt=0.01):
    # Parameters
    nx = len(x_array)
    nt = int(t_final / dt)
    j = 1j  # complex unit

    # Initial conditions
    u = A * np.exp(-x_array**2)
    v = -A * np.exp(-x_array**2)

    # Wavenumbers for Fourier space
    k = np.fft.fftfreq(nx, d=dx) * 2 * np.pi  # Wavenumbers for FFT
    k = 1j * k  # Fourier space derivative factor

    # Time-stepping loop
    for t in range(nt):
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

    return u, v

# Example usage
x_array = np.linspace(-5, 5, 200)  # Spatial grid
t_final = 5  # Final time

u_sol, v_sol = solve_pde_split_step(x_array, t_final)

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
