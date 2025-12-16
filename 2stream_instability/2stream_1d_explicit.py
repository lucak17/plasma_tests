#!/usr/bin/env python3
"""
1D-1V electrostatic PIC simulation of the two-stream instability.

Key features:
- Periodic domain
- Two counter-streaming electron beams + immobile ion background (neutrality)
- CIC (Cloud-In-Cell) deposition and gather (linear weighting)
- Explicit leapfrog pusher for particles
- Field solve via spectral Poisson solver (periodic), then E = -∂φ/∂x
- Diagnostics:
  * Field energy, kinetic energy, total energy
  * Momentum
  * Net charge
  * Discrete Gauss law residual (L2 norm)
  * Fourier mode amplitude |E_k| and measured growth rate from log-slope
  * Phase space snapshots

Units:
- Normalized units are used by default: epsilon0 = 1, m_e = 1, |q_e| = 1, n0 = 1
  => omega_p = sqrt(n0*q^2/(m*epsilon0)) = 1
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# -----------------------------
# Theory: cold two-stream growth rate
# -----------------------------
def cold_two_stream_gamma(k, v0, omega_p=1.0):
    """
    Cold symmetric two-beam growth rate for equal-density beams drifting at ±v0.

    Instability if k*v0 < omega_p. Growth rate:
        gamma = omega_p * sqrt( (sqrt(1+8α) - (1+2α))/2 )
    where α = (k*v0/omega_p)^2.
    """
    alpha = (k * v0 / omega_p) ** 2
    if alpha <= 0.0 or alpha >= 1.0:
        return 0.0
    return omega_p * np.sqrt((np.sqrt(1.0 + 8.0 * alpha) - (1.0 + 2.0 * alpha)) / 2.0)


# -----------------------------
# PIC utilities: CIC deposit and gather
# -----------------------------
def cic_deposit_charge_density(x, q, weight, Nx, L):
    """
    Deposit charge density rho(x) onto Nx grid points using CIC.

    Grid points are at x_i = i*dx, i=0..Nx-1 (periodic).
    For each particle:
      - find left grid index i = floor(x/dx)
      - fractional distance f = x/dx - i in [0,1)
      - deposit to i and i+1 with weights (1-f) and f

    Returns:
      rho : array shape (Nx,), charge density (charge per unit length)
    """
    dx = L / Nx
    xi = x / dx
    i0 = np.floor(xi).astype(int) % Nx
    f = xi - np.floor(xi)

    i1 = (i0 + 1) % Nx

    # Each macro-particle represents "weight" physical particles (in normalized density units).
    # Its charge is q * weight. To convert deposited charge to charge density, divide by dx.
    charge_density_factor = (q * weight) / dx

    rho = np.zeros(Nx, dtype=float)
    np.add.at(rho, i0, charge_density_factor * (1.0 - f))
    np.add.at(rho, i1, charge_density_factor * f)
    return rho


def cic_gather(x, Fgrid, Nx, L):
    """
    Gather a grid field Fgrid (defined at Nx grid points) to particle positions x using CIC.
    """
    dx = L / Nx
    xi = x / dx
    i0 = np.floor(xi).astype(int) % Nx
    f = xi - np.floor(xi)
    i1 = (i0 + 1) % Nx
    return (1.0 - f) * Fgrid[i0] + f * Fgrid[i1]


# -----------------------------
# Field solve: Poisson equation (spectral, periodic) then E = -∂φ/∂x
# -----------------------------
def solve_poisson_periodic(rho, L, epsilon0=1.0):
    """
    Solve Poisson: d^2 phi/dx^2 = -rho/epsilon0 with periodic BC via FFT,
    then compute E = -dphi/dx.

    Returns:
      phi   : (Nx,) potential with zero-mean reference
      E     : (Nx,) electric field at grid points
      rho_c : (Nx,) cell-centered rho used for Gauss residual checks
    """
    Nx = rho.size
    dx = L / Nx

    # Enforce neutrality (zero k=0) for periodic Poisson solve
    rho0 = rho - rho.mean()
    rho_hat = np.fft.rfft(rho0)
    k = 2.0 * np.pi * np.fft.rfftfreq(Nx, d=dx)

    phi_hat = np.zeros_like(rho_hat, dtype=complex)
    mask = k != 0.0
    phi_hat[mask] = rho_hat[mask] / (epsilon0 * k[mask] ** 2)
    # k=0 stays zero -> zero-mean potential reference

    phi = np.fft.irfft(phi_hat, n=Nx)

    # E = -dphi/dx in Fourier space
    E_hat = -1j * k * phi_hat
    E = np.fft.irfft(E_hat, n=Nx)

    # Midpoint charge density between grid points (second-order accurate)
    rho_c = 0.5 * (rho0 + np.roll(rho0, -1))
    return phi, E, rho_c


def gauss_law_residual(E, rho_c, L, epsilon0=1.0):
    """
    Compute discrete Gauss's law residual on intervals:
        (E_{i+1}-E_i)/dx - rho_{i+1/2}/epsilon0
    """
    Nx = E.size
    dx = L / Nx
    dEdx = (np.roll(E, -1) - E) / dx
    res = dEdx - (rho_c / epsilon0)
    l2 = np.sqrt(np.mean(res**2))
    return l2, res


# -----------------------------
# Main simulation
# -----------------------------
def run_two_stream_pic(
    Nx=256,
    particles_per_cell=200,
    L=2.0 * np.pi,
    dt=0.05,
    nsteps=2000,
    v0=0.7,
    vth=0.0,
    mode_m=1,
    perturb_amp=1e-3,
    seed=1,
    diagnostic_stride=1,
    snapshot_stride=200,
    live_plot=False,
    live_stride=None,
):
    """
    Run a two-stream PIC simulation and return diagnostics.
    Set live_plot=True to watch diagnostics update during the run.
    """
    rng = np.random.default_rng(seed)

    # Normalized constants
    epsilon0 = 1.0
    me = 1.0
    qe = -1.0
    n0 = 1.0  # background ion density (and desired mean electron number density)
    omega_p = np.sqrt(n0 * (qe**2) / (me * epsilon0))  # with q^2 = 1 -> omega_p = 1
    dx = L / Nx

    # Derived quantities
    Np = Nx * particles_per_cell
    weight = n0 * L / Np  # each macro-electron represents this many "real" electrons in 1D

    k = 2.0 * np.pi * mode_m / L
    gamma_th = cold_two_stream_gamma(k, v0, omega_p=omega_p)
    if live_stride is None:
        live_stride = diagnostic_stride
    live_stride = max(1, live_stride)

    print("=== Two-stream PIC parameters ===")
    print(f"Nx={Nx}, dx={dx:.4g}, L={L:.4g}")
    print(f"Np={Np} ({particles_per_cell} per cell)")
    print(f"dt={dt:.4g}, nsteps={nsteps}, total time={dt*nsteps:.4g}")
    print(f"v0={v0:.4g}, vth={vth:.4g}")
    print(f"mode m={mode_m}, k={k:.4g}")
    print(f"Cold-theory gamma_th={gamma_th:.6g} (in omega_p units; omega_p={omega_p:.3g})")
    print("=================================")

    # --- Initialize particles ---
    # positions uniform
    #x = rng.random(Np) * L
    Npairs = Np // 2
    x_base = (np.arange(Npairs) + 0.5) * (L / Npairs)
    x = np.concatenate([x_base, x_base])

    # velocities: two equal beams at ±v0 with thermal spread vth
    v = np.empty(Np, dtype=float)
    half = Np // 2
    v[:half] = v0 + vth * rng.standard_normal(half)
    v[half:] = -v0 + vth * rng.standard_normal(Np - half)

    # Seed a single-mode density perturbation by slightly shifting positions
    # x <- x + (delta/k) sin(k x) is a common way (delta small).
    # Here perturb_amp is a small fraction of dx.
    x = (x + perturb_amp * dx * np.sin(k * x)) % L

    # --- Initial field and leapfrog alignment ---
    # Deposit electron charge density + add immobile ion background (+n0)
    rho = cic_deposit_charge_density(x, qe, weight, Nx, L) + n0
    phi, E, rho_c = solve_poisson_periodic(rho, L, epsilon0)

    # Gather E at particles
    Ep = cic_gather(x, E, Nx, L)

    # Leapfrog: store v at half step
    v_half = v + 0.5 * (qe / me) * Ep * dt

    # --- Diagnostics storage ---
    times = []
    Ek_mode = []
    KE = []
    FE = []
    TE = []
    Pmom = []
    netQ = []
    gaussL2 = []
    energy_error = []
    growth_rate = []
    initial_energy = None

    snapshots = []  # list of (t, x_sample, v_sample) for phase-space plots

    # Grid coordinates for plotting
    x_grid = np.arange(Nx) * dx

    # Live plotting setup
    if live_plot:
        plt.ion()
        fig, axes = plt.subplots(2, 2, figsize=(13.5, 5.5))
        phase_ax, field_ax, energy_ax, dist_ax = axes.flatten()
        error_ax = energy_ax.twinx()
        growth_ax = energy_ax.twinx()
        growth_ax.spines["right"].set_position(("axes", 1.08))
        fig.tight_layout()
        phase_stride = max(1, Np // 4000)

    # --- Time loop ---
    for n in range(nsteps):
        t = (n + 1) * dt

        # 1) Deposit rho
        rho = cic_deposit_charge_density(x, qe, weight, Nx, L) + n0

        # Diagnostics: net charge (should be ~0)
        net_charge = rho.mean() * L  # mean(rho) * L == integral rho dx
        # (We neutralize in solver anyway; this tracks numerical drift/noise.)

        # 2) Solve field (explicit FD integration of Gauss)
        phi, E, rho_c = solve_poisson_periodic(rho, L, epsilon0)

        # 3) Gather E to particles
        Ep = cic_gather(x, E, Nx, L)

        # 4) Push (explicit leapfrog)
        v_half += (qe / me) * Ep * dt
        x = (x + v_half * dt) % L

        # --- Diagnostics (every diagnostic_stride steps) ---
        if (n % diagnostic_stride) == 0:
            prev_amp = Ek_mode[-1] if Ek_mode else None
            prev_t = times[-1] if times else None

            # Mode amplitude from FFT of E (discrete Fourier mode index m)
            # rfft returns modes 0..Nx/2; we assume mode_m fits.
            Ehat = np.fft.rfft(E) / Nx
            mode_amp = np.abs(Ehat[mode_m])

            # Energies (approx: use v at half-step)
            kinetic = 0.5 * me * weight * np.sum(v_half**2)
            field = 0.5 * epsilon0 * np.sum(E**2) * dx
            total = kinetic + field
            if initial_energy is None:
                initial_energy = total
            rel_energy_err = abs(total - initial_energy) / abs(initial_energy)

            # Momentum (should be ~conserved in periodic with neutral background)
            momentum = me * weight * np.sum(v_half)

            # Gauss law residual
            l2, _ = gauss_law_residual(E, rho_c, L, epsilon0)

            times.append(t)
            Ek_mode.append(mode_amp)
            KE.append(kinetic)
            FE.append(field)
            TE.append(total)
            energy_error.append(max(rel_energy_err, 1e-30))
            Pmom.append(momentum)
            netQ.append(net_charge)
            gaussL2.append(l2)
            tiny = 1e-30
            if prev_amp is None or prev_t is None:
                growth_rate.append(np.nan)
            else:
                gamma_inst = (np.log(mode_amp + tiny) - np.log(prev_amp + tiny)) / (t - prev_t)
                growth_rate.append(gamma_inst)

            # --- Live plotting ---
            if live_plot and (n % live_stride) == 0:
                phase_ax.cla()
                phase_ax.scatter(x[::phase_stride], v_half[::phase_stride], s=0.5, marker='.')
                phase_ax.set_xlim(0, L)
                phase_ax.set_ylim(-3, 3)
                phase_ax.set_xlabel("x")
                phase_ax.set_ylabel("v")
                phase_ax.legend((mpatches.Patch(color='w'), ), (r'$\omega_{pe}t=$' + f"{t:.2f}", ), loc=1, frameon=False)

                field_ax.cla()
                field_ax.plot(x_grid, E, label="E", linewidth=2)
                field_ax.set_xlim(0, L)
                field_ax.set_xlabel("x")
                field_ax.set_ylabel("E")
                field_ax.legend(loc=1)

                energy_ax.cla()
                #error_ax.cla()
                growth_ax.cla()
                energy_ax.set_xlim(0, nsteps * dt)
                energy_ax.set_xlabel("time")
                energy_ax.set_yscale("log")
                energy_ax.set_ylabel("Energy")
                energy_ax.plot(times, FE, label="Field", linewidth=2)
                energy_ax.plot(times, KE, label="Kinetic", linewidth=2)
                energy_ax.plot(times, TE, label="Total", linestyle="--", linewidth=2)
                if FE:
                    times_arr = np.asarray(times)
                    fe0 = FE[0]
                    fe_theory = fe0 * np.exp(2.0 * gamma_th * (times_arr - times_arr[0]))
                    energy_ax.plot(times_arr, fe_theory, label="Field (theory)", linestyle="--", linewidth=1.8, color="tab:green")
                energy_ax.plot(times, energy_error, label=f"|dE|/E0 (last={energy_error[-1]:.3e})", color="k", linestyle=":", linewidth=2)
                energy_ax.set_ylim(1e-8, max(TE) * 1.2)
                #error_ax.set_ylabel("Energy error (|dE|/E0)")
                #error_ax.set_yscale("log")
                #error_ax.plot(times, energy_error, label=f"|dE|/E0 (last={energy_error[-1]:.3e})", color="k", linestyle=":", linewidth=2)

                #growth_ax.set_ylabel("Instability growth rate γ")
                #growth_ax.plot(times, growth_rate, label="γ_inst", color="tab:purple", linewidth=1.8)
                #growth_ax.axhline(gamma_th, color="tab:green", linestyle="--", linewidth=1.2, label="γ_theory")

                lines, labels = energy_ax.get_legend_handles_labels()
                lines_err, labels_err = error_ax.get_legend_handles_labels()
                lines_g, labels_g = growth_ax.get_legend_handles_labels()
                #growth_ax.legend(lines + lines_err + lines_g, labels + labels_err + labels_g, loc=4)
                growth_ax.legend(lines + lines_err, labels + labels_err, loc=4)

                dist_ax.cla()
                bins, edges = np.histogram(v_half, bins=40, range=(-4, 4))
                centers = 0.5 * (edges[:-1] + edges[1:])
                dist_ax.plot(centers, bins, label="f(v)", linewidth=2)
                dist_ax.set_xlim(-4, 4)
                dist_ax.set_xlabel("v")
                dist_ax.legend(loc=1)

                fig.canvas.draw_idle()
                plt.pause(0.05)

        # --- Phase-space snapshots (downsample particles for plotting clarity) ---
        if (n % snapshot_stride) == 0:
            sample = min(5000, Np)
            idx = rng.choice(Np, size=sample, replace=False)
            snapshots.append((t, x[idx].copy(), v_half[idx].copy()))

    # Convert diagnostics to arrays
    times = np.array(times)
    Ek_mode = np.array(Ek_mode)
    KE = np.array(KE)
    FE = np.array(FE)
    TE = np.array(TE)
    Pmom = np.array(Pmom)
    netQ = np.array(netQ)
    gaussL2 = np.array(gaussL2)
    energy_error = np.array(energy_error)
    growth_rate = np.array(growth_rate)

    if live_plot:
        plt.ioff()

    # --- Estimate growth rate from simulation ---
    # Fit ln|E_k| = a + gamma*t over a "linear growth" window.
    # Automatic heuristic: use points where amplitude is between two thresholds.
    Ek = Ek_mode
    tiny = 1e-30
    logEk = np.log(Ek + tiny)

    # Heuristic thresholds (tune as needed)
    lo = np.percentile(Ek, 20)
    hi = np.percentile(Ek, 80)

    mask = (Ek > lo) & (Ek < hi)
    gamma_fit = np.nan
    if np.sum(mask) >= 10:
        coef = np.polyfit(times[mask], logEk[mask], 1)
        gamma_fit = coef[0]  # slope

    results = {
        "params": {
            "Nx": Nx, "Np": Np, "L": L, "dx": dx, "dt": dt, "nsteps": nsteps,
            "v0": v0, "vth": vth, "mode_m": mode_m, "k": k,
            "omega_p": omega_p, "gamma_th": gamma_th, "gamma_fit": gamma_fit
        },
        "time": times,
        "Ek_mode": Ek_mode,
        "KE": KE,
        "FE": FE,
        "TE": TE,
        "P": Pmom,
        "netQ": netQ,
        "gaussL2": gaussL2,
        "energy_error": energy_error,
        "growth_rate": growth_rate,
        "snapshots": snapshots
    }
    return results



def plot_mode(results):
    p = results["params"]
    t = results["time"]

    print("\n=== Diagnostics summary ===")
    print(f"theory gamma_th = {p['gamma_th']:.6g}")
    print(f"fit    gamma_fit= {p['gamma_fit']:.6g}")
    print(f"max |netQ| (integral rho dx) ~ {np.max(np.abs(results['netQ'])):.3e}")
    print(f"max Gauss L2 residual          {np.max(results['gaussL2']):.3e}")
    print(f"relative total energy change   {(results['TE'][-1]-results['TE'][0])/results['TE'][0]:.3e}")
    print("===========================\n")

    # 1) Mode growth
    plt.figure()
    Ek = results["Ek_mode"]
    plt.semilogy(t, Ek + 1e-30, label="|E_k| (measured)")
    if len(Ek):
        ek0 = Ek[0] if Ek[0] != 0 else 1e-30
        ek_theory = ek0 * np.exp(p["gamma_th"] * (t - t[0]))
        plt.semilogy(t, ek_theory + 1e-30, linestyle="--", color="tab:green", label="|E_k| (theory)")
    plt.xlabel("t")
    plt.ylabel(r"$|E_k|$")
    plt.title("Two-stream: mode amplitude (semilog)")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_results(results):
    p = results["params"]
    t = results["time"]

    print("\n=== Diagnostics summary ===")
    print(f"theory gamma_th = {p['gamma_th']:.6g}")
    print(f"fit    gamma_fit= {p['gamma_fit']:.6g}")
    print(f"max |netQ| (integral rho dx) ~ {np.max(np.abs(results['netQ'])):.3e}")
    print(f"max Gauss L2 residual          {np.max(results['gaussL2']):.3e}")
    print(f"relative total energy change   {(results['TE'][-1]-results['TE'][0])/results['TE'][0]:.3e}")
    print("===========================\n")

    # 1) Mode growth
    plt.figure()
    plt.semilogy(t, results["Ek_mode"] + 1e-30)
    plt.xlabel("t")
    plt.ylabel(r"$|E_k|$")
    plt.title("Two-stream: mode amplitude (semilog)")
    plt.grid(True)

    # 2) Energies
    fig_energy, energy_ax = plt.subplots()
    growth_ax = energy_ax.twinx()
    growth_ax.spines["right"].set_position(("axes", 1.05))

    energy_ax.plot(t, results["KE"], label="Kinetic")
    energy_ax.plot(t, results["FE"], label="Field")
    energy_ax.plot(t, results["TE"], label="Total")
    fe0 = results["FE"][0] if len(results["FE"]) else None
    if fe0 is not None:
        fe_theory = fe0 * np.exp(2.0 * p["gamma_th"] * (t - t[0]))
        energy_ax.plot(t, fe_theory, label="Field (theory)", linestyle="--", linewidth=1.8, color="tab:green")
    energy_ax.set_xlabel("t")
    energy_ax.set_ylabel("Energy")
    energy_ax.set_title("Energy history with growth rate")
    energy_ax.grid(True)

    growth_ax.plot(t, results["growth_rate"], color="tab:purple", label="γ_inst")
    growth_ax.axhline(p["gamma_th"], color="tab:green", linestyle="--", label="γ_theory")
    if not np.isnan(p["gamma_fit"]):
        growth_ax.axhline(p["gamma_fit"], color="tab:cyan", linestyle=":", label="γ_fit")
    growth_ax.set_ylabel("Growth rate γ")

    lines, labels = energy_ax.get_legend_handles_labels()
    lines_g, labels_g = growth_ax.get_legend_handles_labels()
    growth_ax.legend(lines + lines_g, labels + labels_g, loc=4)

    # 3) Momentum
    plt.figure()
    plt.plot(t, results["P"])
    plt.xlabel("t")
    plt.ylabel("Total momentum")
    plt.title("Momentum history")
    plt.grid(True)

    # 4) Gauss residual
    plt.figure()
    plt.semilogy(t, results["gaussL2"] + 1e-30)
    plt.xlabel("t")
    plt.ylabel("Gauss residual (L2)")
    plt.title("Discrete Gauss law residual")
    plt.grid(True)

    # 5) Phase space snapshots
    for (ts, xs, vs) in results["snapshots"][:4]:
        plt.figure()
        plt.scatter(xs, vs, s=1)
        plt.xlabel("x")
        plt.ylabel("v")
        plt.title(f"Phase space (sample) at t={ts:.3g}")
        plt.grid(True)

    plt.show()


if __name__ == "__main__":
    results = run_two_stream_pic(
        Nx=256,
        particles_per_cell=400,
        L=2.0 * np.pi,
        dt=0.05,
        nsteps=2000,
        v0=0.7,
        vth=0.0,
        mode_m=1,
        perturb_amp=1e-3,
        seed=2,
        diagnostic_stride=1,
        snapshot_stride=400,
        live_plot=True,
        live_stride=20,
    )
    plot_mode(results)
    #plot_results(results)
