import numpy as np
import matplotlib.pyplot as plt

Nx, Ny = 100, 100
Lx, Ly = 1.0, 1.0
dx, dy = Lx / (Nx - 1), Ly / (Ny - 1)
tolerance = 1e-6
epsilon_0 = 1.0
max_iter = 10000

V = np.zeros((Nx, Ny))
Pe = np.zeros((Nx, Ny))

x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')

circle_center = (0.5, 0.5)
circle_radius = 0.2
am_i_circle = ((X - circle_center[0])**2 + (Y - circle_center[1])**2) < circle_radius**2

charge_center = (0.8, 0.8)
charge_radius = 0.1
distance_squared = (X - charge_center[0])**2 + (Y - charge_center[1])**2
Pe = 70 * np.exp(-distance_squared / 0.02) * (distance_squared <= charge_radius**2)

B1 = np.sin(np.pi * y)

V[:, 0] = 0
V[:, -1] = 1
V[am_i_circle] = 0

denom = 2 * (1/dx**2 + 1/dy**2)
for iteration in range(max_iter):
    V2 = V.copy()

    V2[1:-1, 1:-1] = (
        (V[2:, 1:-1] + V[:-2, 1:-1]) / dx**2 +
        (V[1:-1, 2:] + V[1:-1, :-2]) / dy**2 -
        Pe[1:-1, 1:-1] / epsilon_0
    ) / denom

    V2[0, :] = V2[1, :] - B1 * dx / epsilon_0
    V2[-1, :] = V2[-2, :] + B1 * dx / epsilon_0

    V2[:, 0] = 0
    V2[:, -1] = 1
    V2[am_i_circle] = 0

    error = np.max(np.abs(V2 - V))
    V[:] = V2

    if error < tolerance:
        print(f"Converged in {iteration} iterations! 🚀")
        break

Ex, Ey = np.gradient(-V, dx, dy)
E_mag = np.sqrt(Ex**2 + Ey**2)

plt.figure(figsize=(8, 6))
plt.contourf(X.T, Y.T, V.T, 50, cmap='inferno')
plt.colorbar(label="Potential (V)")
plt.title("Voltage Potential Distribution")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

plt.figure(figsize=(8, 6))
strm = plt.streamplot(x, y, Ex.T, Ey.T, color=E_mag.T, cmap='cool', linewidth=1, density=1.5)
plt.colorbar(strm.lines, label="Electric Field Magnitude")
plt.contour(X.T, Y.T, V.T, 15, colors='white', alpha=0.3, linewidths=1)
plt.title("Electric Field Lines & Equipotentials")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
