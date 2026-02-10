import numpy as np
import matplotlib.pyplot as plt

# ============================
# Parameters
# ============================

T = 5.0
N = 1000
dt = T / N
t = np.linspace(0.0, T, N + 1)

r0 = 0.03

# Vasicek parameters
a_v = 0.8
b_v = 0.04
sigma_v = 0.02

# CIR parameters
a_c = 0.8
b_c = 0.04
sigma_c = 0.15

np.random.seed(0)

# ============================
# Vasicek exact simulation
# ============================

r_v = np.zeros(N + 1)
r_v[0] = r0

for i in range(N):
    m = r_v[i] * np.exp(-a_v * dt) + b_v * (1.0 - np.exp(-a_v * dt))
    s2 = sigma_v**2 / (2.0 * a_v) * (1.0 - np.exp(-2.0 * a_v * dt))
    r_v[i + 1] = m + np.sqrt(s2) * np.random.randn()

# ============================
# CIR exact simulation
# (non-central chi-square)
# ============================

r_c = np.zeros(N + 1)
r_c[0] = r0

kappa = a_c
theta = b_c
sigma = sigma_c

for i in range(N):

    c = (sigma**2 * (1.0 - np.exp(-kappa * dt))) / (4.0 * kappa)
    d = 4.0 * kappa * theta / sigma**2
    lambda_nc = (4.0 * kappa * np.exp(-kappa * dt) * r_c[i]) / \
                (sigma**2 * (1.0 - np.exp(-kappa * dt)))

    r_c[i + 1] = c * np.random.noncentral_chisquare(d, lambda_nc)

# ============================
# Numeraire computation
# ============================

# Integral of r_t by left-point rule
int_v = np.zeros(N + 1)
int_c = np.zeros(N + 1)

int_v[1:] = np.cumsum(r_v[:-1]) * dt
int_c[1:] = np.cumsum(r_c[:-1]) * dt

B_v = np.exp(int_v)
B_c = np.exp(int_c)

# ============================
# Plots
# ============================

plt.figure(figsize=(8, 4))
plt.plot(t, r_v, label="Vasicek")
plt.plot(t, r_c, label="CIR")
plt.xlabel("Time")
plt.ylabel("Short rate")
plt.title("Short rate trajectories")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(t, B_v, label="Vasicek numeraire")
plt.plot(t, B_c, label="CIR numeraire")
plt.xlabel("Time")
plt.ylabel("Numeraire")
plt.title("Numeraire comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================
# Simple numerical comparison
# ============================

print("Final short rate:")
print("  Vasicek :", r_v[-1])
print("  CIR     :", r_c[-1])

print("\nFinal numeraire:")
print("  Vasicek :", B_v[-1])
print("  CIR     :", B_c[-1])
