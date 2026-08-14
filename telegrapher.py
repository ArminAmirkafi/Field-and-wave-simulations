import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ....PARAMETERS AND CONSTANTS

tolerance = 0.01
C_speed = 3e8  
epsilon = 8.85e-12
w = 2 * np.pi * 15e9
L = 1
Nx, Nz = 100, 100
dx, dz = 0.01, 0.01

a = 0.25
b = 0.21
tang = (b - a) / L

# Grid setup
Z = np.linspace(0, 100, Nx)
distance = 29 - a*100 - tang*Z
distance_idx = np.floor(distance).astype(int)

space = np.zeros((30, Nx))

#TEST CASE 1: Connecting lower plate to 1V

v1 = space.copy()
v2 = space.copy()
v1[29, :] = 1.0
v2[29, :] = 1.0

mask = np.ones((30, Nx), dtype=bool)
mask[0, :] = False     # Top boundary
mask[29, :] = False    # Bottom boundary
mask[:, 0] = False     # Left boundary
mask[:, -1] = False    # Right boundary
for j in range(1, Nx-1):
    mask[distance_idx[j], j] = False

# LAPLACE equation for capacitator_induced effect
for cycle in range(1000):
    # Calculate averages for the entire inner grid at once
    v2_new = 0.25 * (v1[2:, 1:-1] + v1[:-2, 1:-1] + v1[1:-1, 2:] + v1[1:-1, :-2])
    # Apply updates only where the mask is True
    v2[1:-1, 1:-1] = np.where(mask[1:-1, 1:-1], v2_new, v1[1:-1, 1:-1])
    
    error = np.max(np.abs(v2 - v1))
    v1[:] = v2
    if error < tolerance:
        break

# Calculate C1 and Cm
C1 = np.zeros(Nx)
Cm = np.zeros(Nx)
for i in range(Nx):
    C1[i] = -epsilon * (v1[28, i] - v1[29, i]) / dz
    Cm[i] = epsilon * (v1[distance_idx[i], i] - v1[distance_idx[i]+1, i]) / dz

#TEST CASE 2: Calculating C2

v1 = space.copy()
v2 = space.copy()
for i in range(Nx):
    v1[distance_idx[i], i] = 1.0
    v2[distance_idx[i], i] = 1.0

# Mask for second Laplace calculation
mask2 = np.ones((30, Nx), dtype=bool)
mask2[0:4, :] = False  # Skip upper rows based on original logic
mask2[29, :] = False
mask2[:, 0] = False
mask2[:, -1] = False
for j in range(1, Nx-1):
    mask2[distance_idx[j], j] = False

# Vectorized Laplace Solver
for cycle in range(100):
    v2_new = 0.25 * (v1[2:, 1:-1] + v1[:-2, 1:-1] + v1[1:-1, 2:] + v1[1:-1, :-2])
    v2[1:-1, 1:-1] = np.where(mask2[1:-1, 1:-1], v2_new, v1[1:-1, 1:-1])
    
    error = np.max(np.abs(v2 - v1))
    v1[:] = v2
    if error < tolerance:
        break

# Calculate C2 and Inductances
C2 = np.zeros(Nx)
for i in range(Nx):
    C2[i] = epsilon * (-1 * (v1[distance_idx[i], i] - v1[distance_idx[i]+1, i])) / dz

# Boundary conditions and zero-division safety
C2[0], C2[-1] = 1e-16, 1e-16
Cm[0], Cm[-1] = 1e-16, 1e-16

Lm = (1 / C_speed**2) * (1 / Cm)
L1 = (1 / C_speed**2) * (1 / C1)
L2 = (1 / C_speed**2) * (1 / C2)

# beloved ODE SOLVER: 4th-Order Adams Predictor-Corrector

# State vector Y = [V1, V2, dV1/dz, dV2/dz]
Y = np.zeros((Nx, 4))
Y[0] = [1.0, 0.0, 0.0, 0.0]  # Initial conditions at z=0

# History array to store the derivatives at each step
F = np.zeros((Nx, 4))

def get_derivatives(idx, state):
    v1_val, v2_val, u1_val, u2_val = state
    
    # Extract coefficients for the current spatial step
    c11 = -(w**2) * (C1[idx]*L1[idx] + Cm[idx]*Lm[idx])
    c12 = -(w**2) * (C1[idx]*Lm[idx] + Cm[idx]*L2[idx])
    c21 = -(w**2) * (Cm[idx]*L1[idx] + C2[idx]*Lm[idx])
    c22 = -(w**2) * (Cm[idx]*Lm[idx] + C2[idx]*L2[idx])
    
    # Calculate second derivatives (which are the first derivatives of u1, u2)
    du1_dz = c11 * v1_val + c12 * v2_val
    du2_dz = c21 * v1_val + c22 * v2_val
    
    return np.array([u1_val, u2_val, du1_dz, du2_dz])

# Calculate initial derivative
F[0] = get_derivatives(0, Y[0])

# BOOTSTRAP: Use 4th-Order Runge-Kutta (RK4) for steps 1, 2, 3
for i in range(3):
    k1 = get_derivatives(i, Y[i])
    k2 = get_derivatives(i, Y[i] + 0.5 * dz * k1)
    k3 = get_derivatives(i, Y[i] + 0.5 * dz * k2)
    # Ensure index doesn't exceed bounds, though safe here since i < 3
    idx_next = min(i+1, Nx-1) 
    k4 = get_derivatives(idx_next, Y[i] + dz * k3)
    
    Y[i+1] = Y[i] + (dz / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    F[i+1] = get_derivatives(i+1, Y[i+1])

# Adams-ODE_solver: Predictor-Corrector for Steps 4 to Nx
for i in range(3, Nx - 1):
    # Predictor: 4th-Order Adams-Bashforth
    Y_pred = Y[i] + (dz / 24.0) * (55 * F[i] - 59 * F[i-1] + 37 * F[i-2] - 9 * F[i-3])
    F_pred = get_derivatives(i+1, Y_pred)
    
    # Corrector: 4th-Order Adams-Moulton
    Y[i+1] = Y[i] + (dz / 24.0) * (9 * F_pred + 19 * F[i] - 5 * F[i-1] + F[i-2])
    F[i+1] = get_derivatives(i+1, Y[i+1])

# Unpack the voltages
V1 = Y[:, 0]
V2 = Y[:, 1]

# PLOTTING AND ANIMATION

for i in range(Nx):
    space[29 - distance_idx[i], i] = V2[i]
    space[0, i] = V1[i]

T_period = 2 * np.pi / w
duration = T_period * 3
n_frames = 30
tr = np.linspace(0, duration, n_frames)

fig, ax = plt.subplots(figsize=(8, 4))

def update(frame):
    t = tr[frame]
    ax.clear()
    data = space * np.cos(w * t)
    conobj = ax.contourf(data, levels=50, cmap='inferno')
    cycle_num = t / T_period
    ax.set_title(f"{t*1e9:.2f}ns ({cycle_num:.2f} cycles)")
    ax.set_xlabel("Z")
    ax.set_ylabel("Y")
    return conobj.collections

ani = FuncAnimation(fig, update, frames=n_frames, interval=100, blit=False)
plt.show()