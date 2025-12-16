#!/usr/bin/env python3
"""
ECSIM 1D3V Particle-In-Cell code (periodic) with diagnostics for electrostatic two-stream.

Reference:
G. Lapenta, "Exactly energy conserving semi-implicit particle in cell formulation",
JCP 334 (2017) 349–366.

This version:
- keeps your ECSIM Maxwell solve (theta-scheme, GMRES)
- fixes/clarifies normalization checks
- adds correct cold electrostatic two-stream growth rate gamma(k)
- adds correct diagnostics for electrostatic two-stream: |Ex_k| and Ex-energy
- (optional) initializes Ex from Gauss/Poisson to satisfy Gauss law at t=0 if desired

Units:
- Gaussian units in the field solver/energy:
  Ampere: dE/dt = c curl B - 4π J
  Energy density: (E^2 + B^2)/(8π)

Author: adapted from your script, with fixes by ChatGPT
License: MIT
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import gmres, LinearOperator
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time

# ------------------------
# Plot params
# ------------------------
params = {
    "axes.labelsize": "large",
    "xtick.labelsize": "medium",
    "ytick.labelsize": "medium",
    "font.size": 15,
    "font.family": "sans-serif",
    "text.usetex": False,
    "mathtext.fontset": "stixsans",
}
plt.rcParams.update(params)

# ------------------------
# Theory: cold electrostatic two-stream growth rate (symmetric beams)
# ------------------------
def cold_two_stream_gamma(k, v0, omega_pe):
    """
    Cold symmetric two-stream (two equal-density beams drifting at ±v0),
    longitudinal electrostatic instability.

    Instability band: k*v0 < omega_pe

    gamma = omega_pe * sqrt( (sqrt(1+8α) - (1+2α)) / 2 )
    α = (k*v0/omega_pe)^2

    Returns amplitude growth rate gamma (for |Ex_k| ~ exp(gamma t)).
    Energy in Ex grows as exp(2 gamma t).
    """
    alpha = (k * v0 / omega_pe) ** 2
    if alpha <= 0.0 or alpha >= 1.0:
        return 0.0
    return omega_pe * np.sqrt((np.sqrt(1.0 + 8.0 * alpha) - (1.0 + 2.0 * alpha)) / 2.0)

def fit_log_slope(t, y, tmin=None, tmax=None, eps=1e-300):
    """Fit ln(y) = a + b t over [tmin,tmax]. Return b."""
    t = np.asarray(t)
    y = np.asarray(y)
    if tmin is None: tmin = t[0]
    if tmax is None: tmax = t[-1]
    mask = (t >= tmin) & (t <= tmax) & np.isfinite(y) & (y > 0)
    if np.sum(mask) < 10:
        return np.nan
    coef = np.polyfit(t[mask], np.log(y[mask] + eps), 1)
    return coef[0]

# ------------------------
# Physical constants (Gaussian units in solver)
# ------------------------
c = 1.0

# ------------------------
# Simulation parameters
# ------------------------
Nx      = 64
Lx      = 2*np.pi
dx      = Lx / Nx
dt      = 0.05
nsteps  = 1000
TOut    = 10

ngB     = Nx          # B at cell centers
ngE     = Nx + 1      # E stored with duplicated last node for plotting; solver uses first Nx

# Coordinates
xgrid_centers = np.linspace(0, Lx - dx, Nx) + dx/2
xgrid_nodes   = np.linspace(0, Lx - dx, Nx)          # nodes 0..Nx-1

print("CFL (c*dt/dx):", c*dt/dx)

# ------------------------
# Species / normalization choices
# ------------------------
moving_s = 1  # only electrons advanced

# We enforce omega_pe = w_pe in Gaussian units via q_e and qm_e and macro-density n0 = Np/Lx
w_pe   = 1.0          # desired plasma frequency
qm_e   = -1.0         # q/m (normalized)
vt_e   = 0.0          # thermal speed (cold)
vdrift = 0.2          # beam drift magnitude (beams at ±vdrift)

nppc   = 154
Np     = nppc * Nx

# Macro-density in 1D (number per unit length in code units)
n0 = Np / Lx

# Choose per-particle charge q_e so that omega_pe^2 = 4π n0 (q^2/m) = 4π n0 (q*qm)
# => q = omega_pe^2 / (4π n0 qm)
q_e = w_pe**2 / (4*np.pi * n0 * qm_e)   # negative since qm_e < 0

# Derived implied mass (for diagnostics): m = q/qm
m_e = q_e / qm_e

print("=== Normalization check (Gaussian) ===")
print(f"Nx={Nx}, Lx={Lx:.6g}, dx={dx:.6g}, Np={Np}, n0= Np/Lx = {n0:.6g}")
print(f"qm_e={qm_e:.6g}, q_e={q_e:.6g}, implied m_e=q/qm={m_e:.6g}")
omega_check = np.sqrt(4*np.pi * n0 * (q_e * qm_e))
print(f"Target omega_pe={w_pe:.6g}, check omega_pe={omega_check:.6g}")
print("======================================")

# Instability mode (Fourier index)
mode = 3
k_mode = 2*np.pi*mode / Lx
gamma_th = cold_two_stream_gamma(k_mode, abs(vdrift), w_pe)
print(f"Cold electrostatic two-stream theory: mode={mode}, k={k_mode:.6g}, gamma={gamma_th:.6g}, 2gamma={2*gamma_th:.6g}")

# Seeding (choose ONE of these typical seeds)
perturb_v = 0.0       # velocity perturbation amplitude (add dv*sin(kx))
perturb_x = 1e-3       # density/position perturbation amplitude (shift x by (perturb_x/k)*sin(kx))

# If you use a density perturbation (perturb_x != 0), it's recommended to initialize Ex from Gauss law
initialize_Ex_from_charge = True

# Background magnetic field (optional)
B0 = 0.0

# ------------------------
# Helper functions (CIC weights)
# ------------------------

def get_weights_CIC1D_Egrid_matrix(xp):
    """
    CIC weights from particles to Nx periodic nodes (no duplicated node).
    Returns sparse matrix W of shape (Np, Nx), where W[p,g] is weight of particle p on node g.
    """
    g1 = np.floor(xp/dx).astype(int) % Nx
    g2 = (g1 + 1) % Nx
    frac = xp/dx - np.floor(xp/dx)
    w1 = 1.0 - frac
    w2 = frac

    p_idx = np.arange(len(xp))
    rows = np.concatenate([p_idx, p_idx])
    cols = np.concatenate([g1, g2])
    data = np.concatenate([w1, w2])
    return sparse.csc_matrix((data, (rows, cols)), shape=(len(xp), Nx))

def get_weights_CIC1D_Bgrid_matrix(xp):
    """
    CIC weights to B-grid (cell centers), ngB = Nx, periodic.
    """
    xi = xp/dx - 0.5
    g1f = np.floor(xi)
    g1 = (g1f.astype(int)) % ngB
    g2 = (g1 + 1) % ngB
    frac = xi - g1f
    w1 = 1.0 - frac
    w2 = frac

    p_idx = np.arange(len(xp))
    rows = np.concatenate([p_idx, p_idx])
    cols = np.concatenate([g1, g2])
    data = np.concatenate([w1, w2])
    return sparse.csc_matrix((data, (rows, cols)), shape=(len(xp), ngB))

# ------------------------
# Field operators (1D periodic curls with staggering)
# ------------------------
def curl_nodes_to_center_1D_periodic(field_nodes):
    """
    field_nodes: (3, Nx) on nodes.
    returns curl on centers (3, Nx).
    In 1D (variation in x only):
      (∇×E)_y = -∂Ez/∂x
      (∇×E)_z =  ∂Ey/∂x
    """
    curl = np.zeros((3, Nx))
    curl[1, :] = (field_nodes[2, :] - np.roll(field_nodes[2, :], -1)) / dx   # -dEz/dx
    curl[2, :] = (np.roll(field_nodes[1, :], -1) - field_nodes[1, :]) / dx   #  dEy/dx
    return curl

def curl_center_to_nodes_1D_periodic(field_centers):
    """
    field_centers: (3, Nx) on centers.
    returns curl on nodes (3, Nx).
    In 1D:
      (∇×B)_y = -∂Bz/∂x
      (∇×B)_z =  ∂By/∂x
    Using center->node difference.
    """
    curl = np.zeros((3, Nx))
    curl[1, :] = (np.roll(field_centers[2, :], +1) - field_centers[2, :]) / dx  # -dBz/dx
    curl[2, :] = (field_centers[1, :] - np.roll(field_centers[1, :], +1)) / dx  #  dBy/dx
    return curl

# ------------------------
# ECSIM rotation / alpha matrix
# ------------------------
def build_alpha_matrices_vectorized(xp, beta_p, B_centers):
    """
    Build per-particle alpha matrices for ECSIM (vectorized).
    B is on centers; interpolate to particles.
    """
    Wb = get_weights_CIC1D_Bgrid_matrix(xp)            # (Np, Nx)
    Bp = (Wb @ B_centers.T)                            # (Np, 3)

    b2 = np.sum(Bp**2, axis=1)                        # (Np,)
    denom = 1.0 + (beta_p**2) * b2 / c**2             # (Np,)

    # Skew-symmetric cross-product matrices for Bp
    Bx = np.zeros((len(xp), 3, 3))
    Bx[:, 0, 1] = -Bp[:, 2]
    Bx[:, 0, 2] =  Bp[:, 1]
    Bx[:, 1, 0] =  Bp[:, 2]
    Bx[:, 1, 2] = -Bp[:, 0]
    Bx[:, 2, 0] = -Bp[:, 1]
    Bx[:, 2, 1] =  Bp[:, 0]

    outer = np.einsum("ni,nj->nij", Bp, Bp)           # (Np,3,3)
    I = np.eye(3)[None, :, :]

    alpha = (I - (beta_p/c) * Bx + (beta_p**2 / c**2) * outer) / denom[:, None, None]
    return alpha

def deposit_current_hat(xp, vp, q_s, beta_p, B_centers):
    """
    Deposit J_hat on nodes using ECSIM: J_hat = Σ q_s * (alpha_p v_p) W / dx
    """
    We = get_weights_CIC1D_Egrid_matrix(xp)          # (Np, Nx)
    alpha = build_alpha_matrices_vectorized(xp, beta_p, B_centers)  # (Np,3,3)
    v_hat = np.einsum("pij,jp->ip", alpha, vp)       # (3, Np)

    J_hat = (We.T @ v_hat.T).T * (q_s / dx)          # (3, Nx)
    return J_hat

def build_mass_matrix_blocks(xp, beta_p, q_s, B_centers):
    """
    Build diagonal and upper-diagonal 3x3 blocks of ECSIM mass matrix on nodes.

    Uses:
      (q_s * beta_p / dx) * alpha_p  as the per-particle coefficient, summed with weights.
    """
    alpha = build_alpha_matrices_vectorized(xp, beta_p, B_centers)  # (Np,3,3)
    We = get_weights_CIC1D_Egrid_matrix(xp).toarray()               # (Np, Nx) (Nx small here)
    coeff = (q_s * beta_p / dx) * alpha                              # (Np,3,3)

    We2 = We**2
    main = np.einsum("pij,pg->ijg", coeff, We2)                      # (3,3,Nx)

    We_shift = np.roll(We, -1, axis=1)
    prod = We * We_shift
    upper = np.einsum("pij,pg->ijg", coeff, prod)                    # (3,3,Nx)
    return main, upper

def mass_matrix_dot_E(main, upper, E_nodes):
    """
    Multiply stored block-tridiagonal mass matrix by E (nodes).
    main, upper shape (3,3,Nx). E shape (3,Nx). Return (3,Nx).
    """
    y = np.einsum("ijg,jg->ig", main, E_nodes)
    y += np.einsum("ijg,jg->ig", upper, np.roll(E_nodes, -1, axis=1))  # from g+1
    y += np.roll(np.einsum("ijg,jg->ig", upper, E_nodes), +1, axis=1)   # from g-1
    return y

# ------------------------
# Particle update (1D positions, 3V velocities)
# ------------------------
def update_position1D(xp, vp, dt):
    xp[:] = (xp + vp[0, :] * dt) % Lx

def update_velocity_ECSIM(xp, vp, beta_p, B_centers, E_half_nodes):
    """
    ECSIM velocity update using E at half step and B at n:
      v^{n+1} = 2 v^- - v^n, where v^- = alpha ( v^n + beta E_half(xp) )
    """
    We = get_weights_CIC1D_Egrid_matrix(xp)
    E_p = (We @ E_half_nodes.T).T                     # (3,Np)

    vm = vp + beta_p * E_p
    alpha = build_alpha_matrices_vectorized(xp, beta_p, B_centers)
    vminus = np.einsum("pij,jp->ip", alpha, vm)
    return 2.0 * vminus - vp

# ------------------------
# Optional: initialize Ex from Gauss law (Poisson) in Gaussian units
# ------------------------
def deposit_charge_density_nodes(xp, q_s):
    """
    Deposit charge density rho(x) on Nx nodes:
      rho_g = Σ_p q_s W_pg / dx
    """
    We = get_weights_CIC1D_Egrid_matrix(xp)
    ones = np.ones(len(xp))
    rho = (We.T @ ones) * (q_s / dx)     # (Nx,)
    return np.asarray(rho).ravel()

def solve_Ex_from_rho_gauss(rho_nodes):
    """
    Periodic Gauss law: dEx/dx = 4π rho.
    In Fourier:
      i k Ex_k = 4π rho_k  => Ex_k = -4π i rho_k / k, for k != 0; Ex_0 = 0.
    """
    rho0 = rho_nodes - rho_nodes.mean()
    rho_hat = np.fft.fft(rho0)
    k = 2*np.pi*np.fft.fftfreq(Nx, d=dx)

    Ex_hat = np.zeros_like(rho_hat, dtype=complex)
    mask = (k != 0.0)
    Ex_hat[mask] = -4*np.pi * 1j * rho_hat[mask] / k[mask]
    Ex = np.fft.ifft(Ex_hat).real
    return Ex

# ------------------------
# Maxwell ECSIM solve (theta-scheme) via GMRES
# ------------------------
def solve_maxwell_full(E_n_nodes, B_n_centers, J_hat_tot, main_tot, upper_tot,
                       theta=0.5, tol=1e-10, maxit=500):
    """
    Solve for E^{n+theta} (here theta=0.5 -> E_half), then update to E^{n+1}, B^{n+1}.

    Operator:
      [I + (theta c dt)^2 curl curl + 4π theta dt M] E_half = RHS
    RHS:
      E_n + theta dt c curl B_n - 4π theta dt J_hat
    """
    fac_M = 4*np.pi*theta*dt
    fac_curlcurl = (theta*c*dt)**2

    def A_matvec(e_flat):
        Eg = e_flat.reshape(3, Nx)
        curlC = curl_nodes_to_center_1D_periodic(Eg)
        curlN = curl_center_to_nodes_1D_periodic(curlC)
        y = Eg + fac_curlcurl * curlN + fac_M * mass_matrix_dot_E(main_tot, upper_tot, Eg)
        return y.ravel()

    Aop = LinearOperator((3*Nx, 3*Nx), matvec=A_matvec, dtype=np.float64)

    rhs = (E_n_nodes
           + theta*dt*c*curl_center_to_nodes_1D_periodic(B_n_centers)
           - fac_M * J_hat_tot)
    b = rhs.ravel()

    # GMRES call compatible across SciPy versions
    it_counter = {"n": 0}
    def cb(_):
        it_counter["n"] += 1

    try:
        E_half_flat, info = gmres(Aop, b, rtol=tol, atol=0.0, restart=20, maxiter=maxit,
                                  callback=cb, callback_type="legacy")
    except TypeError:
        E_half_flat, info = gmres(Aop, b, tol=tol, restart=20, maxiter=maxit, callback=cb)

    if info != 0:
        raise RuntimeError(f"GMRES failed (info={info})")

    E_half = E_half_flat.reshape(3, Nx)
    E_np1 = 2*E_half - E_n_nodes
    B_np1 = B_n_centers - c*dt*curl_nodes_to_center_1D_periodic(E_half)
    return E_half, E_np1, B_np1, it_counter["n"]

# ------------------------
# Initialization: two cold beams ±vdrift
# ------------------------
rng = np.random.default_rng(1)

# positions (you can also do a quiet start; random is fine if you want noise seeding)
#x_e = rng.random(Np) * Lx
Npairs = Np // 2
x_base = (np.arange(Npairs) + 0.5) * (Lx / Npairs)
x_e = np.concatenate([x_base, x_base])
# optional density perturbation (position shift): x <- x + (A/k) sin(kx)
if perturb_x != 0.0:
    x_e = (x_e + (perturb_x / k_mode) * np.sin(k_mode * x_e)) % Lx

# velocities: alternate ±vdrift (cold beams) + optional thermal spread
v_e = np.zeros((3, Np))
pm = np.arange(Np)
pm = 1 - 2 * np.mod(pm+1, 2)   # +1, -1, +1, -1, ...
v_e[0, :] = pm * vdrift + rng.normal(0.0, vt_e, Np)

# optional velocity perturbation: dv*sin(kx)
if perturb_v != 0.0:
    v_e[0, :] += perturb_v * np.sin(k_mode * x_e)

# fields
E = np.zeros((3, ngE))
B = np.zeros((3, ngB))
B[1, :] = B0

# initialize Ex to satisfy Gauss law if desired (especially recommended if perturb_x != 0)
if initialize_Ex_from_charge:
    rho_nodes = deposit_charge_density_nodes(x_e, q_e)
    Ex0 = solve_Ex_from_rho_gauss(rho_nodes)
    E[0, :Nx] = Ex0
    E[0, -1] = E[0, 0]

E_half = np.zeros_like(E)

beta_e = qm_e * dt / 2

# ------------------------
# Diagnostics
# ------------------------
hist_t = []
hist_EB_energy = []
hist_E_energy = []
hist_Ex_energy = []
hist_K_energy = []
hist_total = []
hist_total_err = []

hist_Exk_amp = []
hist_Exk_amp_theory = []
hist_ExE_theory = []
hist_gamma_inst = []

hist_Bzk_amp = []

# for momentum diagnostic
pX0 = m_e * np.sum(v_e[0, :])
print("Initial px0 =", pX0)

initial_total_energy = None

# Live plotting
live_plot = True
live_stride = 10

if live_plot:
    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(14, 6))
    phase_ax, field_ax, energy_ax, spec_ax = axes.flatten()
    fig.tight_layout()
    phase_stride = max(1, Np // 4000)

# ------------------------
# Main time loop
# ------------------------
start = time.perf_counter()

for it in range(nsteps):
    tnow = it * dt

    # 1) advance positions (simple)
    update_position1D(x_e, v_e, dt)

    # 2) deposit J_hat
    J_hat = deposit_current_hat(x_e, v_e, q_e, beta_e, B)

    # 3) build mass matrix
    main_e, upper_e = build_mass_matrix_blocks(x_e, beta_e, q_e, B)

    # 4) solve Maxwell for E_half, then update E,B
    Eold_nodes = E[:, :Nx].copy()
    Bold = B.copy()

    E_half_nodes, E_np1_nodes, B, gmres_iters = solve_maxwell_full(
        Eold_nodes, Bold, J_hat, main_e, upper_e, theta=0.5, tol=1e-10, maxit=500
    )

    # store back with duplicate periodic node for plotting convenience
    E_half[:, :Nx] = E_half_nodes
    E[:, :Nx] = E_np1_nodes
    E_half[:, -1] = E_half[:, 0]
    E[:, -1] = E[:, 0]

    # 5) final velocity update
    v_np1 = update_velocity_ECSIM(x_e, v_e, beta_e, Bold, E_half_nodes)
    v_e[:, :] = v_np1

    # ----------------
    # Diagnostics
    # ----------------
    # energies (Gaussian): field energy density (E^2+B^2)/(8π)
    E_energy = dx * np.sum(E[:, :Nx]**2) / (8*np.pi)
    B_energy = dx * np.sum(B[:, :]**2) / (8*np.pi)
    EB_energy = E_energy + B_energy

    # electrostatic two-stream should be in Ex primarily
    Ex_energy = dx * np.sum(E[0, :Nx]**2) / (8*np.pi)

    # kinetic energy: sum 1/2 m v^2
    K_energy = 0.5 * m_e * np.sum(v_e**2)

    total = EB_energy + K_energy
    if initial_total_energy is None:
        initial_total_energy = total
    total_err = abs(total - initial_total_energy) / max(abs(initial_total_energy), 1e-300)

    # mode amplitude of Ex at chosen mode
    Ex_fft = np.fft.fft(E[0, :Nx])
    Exk_amp = np.abs(Ex_fft[mode]) / Nx

    # theory curves (only meaningful during linear phase, for electrostatic mode)
    if len(hist_t) == 0:
        Exk0 = max(Exk_amp, 1e-300)
        ExE0 = max(Ex_energy, 1e-300)
    else:
        Exk0 = max(hist_Exk_amp[0], 1e-300)
        ExE0 = max(hist_Ex_energy[0], 1e-300)

    Exk_theory = Exk0 * np.exp(gamma_th * tnow)
    ExE_theory = ExE0 * np.exp(2*gamma_th * tnow)

    # instantaneous growth estimate from |Ex_k|
    if len(hist_t) >= 1:
        dt_loc = tnow - hist_t[-1]
        eps = 1e-300
        gamma_inst = (np.log(Exk_amp + eps) - np.log(hist_Exk_amp[-1] + eps)) / max(dt_loc, 1e-300)
    else:
        gamma_inst = np.nan

    # Bz mode amplitude (kept as additional info; not the electrostatic benchmark)
    Bz_fft = np.fft.fft(B[2, :])
    Bzk_amp = np.abs(Bz_fft[mode]) / Nx

    hist_t.append(tnow)
    hist_EB_energy.append(EB_energy)
    hist_E_energy.append(E_energy)
    hist_Ex_energy.append(Ex_energy)
    hist_K_energy.append(K_energy)
    hist_total.append(total)
    hist_total_err.append(total_err)

    hist_Exk_amp.append(Exk_amp)
    hist_Exk_amp_theory.append(Exk_theory)
    hist_ExE_theory.append(ExE_theory)
    hist_gamma_inst.append(gamma_inst)

    hist_Bzk_amp.append(Bzk_amp)

    # ----------------
    # Live plots / output
    # ----------------
    if (it % TOut) == 0:
        # Fit gamma over a window (heuristic): later you can tune this
        gamma_fit = fit_log_slope(np.array(hist_t), np.array(hist_Exk_amp),
                                  tmin=0.2*(nsteps*dt), tmax=0.6*(nsteps*dt))
        print(f"[it={it:4d}] t={tnow:.3g} GMRES iters={gmres_iters:3d} |Ex_k|={Exk_amp:.3e} "
              f"gamma_inst={gamma_inst:.3g} gamma_fit~{gamma_fit:.3g} gamma_th={gamma_th:.3g}")

    if live_plot and (it % live_stride) == 0:
        phase_ax.cla()
        phase_ax.scatter(x_e[0::2][::phase_stride], v_e[0, 0::2][::phase_stride], s=0.6, marker=".", color="blue", label="+beam")
        phase_ax.scatter(x_e[1::2][::phase_stride], v_e[0, 1::2][::phase_stride], s=0.6, marker=".", color="red",  label="-beam")
        phase_ax.set_xlim(0, Lx)
        phase_ax.set_ylim(-0.6, 0.6)
        phase_ax.set_xlabel("x")
        phase_ax.set_ylabel("v_x")
        phase_ax.legend(loc=1, frameon=False)

        field_ax.cla()
        field_ax.plot(xgrid_nodes, E[0, :Nx], linewidth=2, label="Ex")
        field_ax.set_xlim(0, Lx)
        field_ax.set_xlabel("x")
        field_ax.set_ylabel("E_x")
        field_ax.legend(loc=1)

        energy_ax.cla()
        energy_ax.set_xlim(0, nsteps*dt)
        energy_ax.set_yscale("log")
        energy_ax.set_xlabel("t")
        energy_ax.set_ylabel("Energy")
        energy_ax.plot(hist_t, hist_Ex_energy, label="Ex energy", linewidth=2)
        energy_ax.plot(hist_t, hist_ExE_theory, label="Ex energy theory (2γ)", linestyle="--", linewidth=2)
        energy_ax.plot(hist_t, hist_K_energy, label="Kinetic", linewidth=1.8)
        energy_ax.plot(hist_t, hist_EB_energy, label="Field (E+B)", linewidth=1.8)
        energy_ax.plot(hist_t, hist_total, label="Total", linestyle="--", linewidth=1.8)
        energy_ax.plot(hist_t, hist_total_err, label="|ΔE|/E0", color="k", linestyle=":", linewidth=2)
        energy_ax.legend(loc=4)

        spec_ax.cla()
        spec_ax.set_xlim(0, nsteps*dt)
        spec_ax.set_yscale("log")
        spec_ax.set_xlabel("t")
        spec_ax.set_ylabel("Mode amplitudes")
        spec_ax.plot(hist_t, np.array(hist_Exk_amp) + 1e-300, label="|Ex_k|", linewidth=2)
        spec_ax.plot(hist_t, np.array(hist_Exk_amp_theory) + 1e-300, label="|Ex_k| theory (γ)", linestyle="--", linewidth=2)
        spec_ax.plot(hist_t, np.array(hist_Bzk_amp) + 1e-300, label="|Bz_k|", linewidth=1.5)
        spec_ax.legend(loc=4)

        fig.canvas.draw_idle()
        plt.pause(0.01)

end = time.perf_counter()
print(f"Execution time: {end - start:.6f} s")

if live_plot:
    plt.ioff()

# ------------------------
# Final summary plots
# ------------------------
tarr = np.array(hist_t)
Exk = np.array(hist_Exk_amp)
gamma_fit = fit_log_slope(tarr, Exk, tmin=0.2*(nsteps*dt), tmax=0.6*(nsteps*dt))

print("\n=== Summary ===")
print(f"mode={mode}, k={k_mode:.6g}, v0={abs(vdrift):.6g}, omega_pe={w_pe:.6g}")
print(f"cold theory gamma_th = {gamma_th:.6g}")
print(f"fit from ln|Ex_k|    = {gamma_fit:.6g}")
print("================\n")

plt.figure()
plt.semilogy(tarr, Exk + 1e-300, label="|Ex_k|")
plt.semilogy(tarr, np.array(hist_Exk_amp_theory) + 1e-300, "--", label="|Ex_k| theory")
plt.xlabel("t")
plt.ylabel("|Ex_k|")
plt.title("Electrostatic two-stream: mode amplitude")
plt.grid(True)
plt.legend()

plt.figure()
plt.semilogy(tarr, np.array(hist_Ex_energy) + 1e-300, label="Ex energy")
plt.semilogy(tarr, np.array(hist_ExE_theory) + 1e-300, "--", label="Ex energy theory (2γ)")
plt.xlabel("t")
plt.ylabel("Energy")
plt.title("Electrostatic two-stream: Ex energy")
plt.grid(True)
plt.legend()

plt.show()
