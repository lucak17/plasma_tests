#!/usr/bin/env python3
"""
1D-1V electrostatic PIC simulation of Landau damping (electrons + immobile ions).

Model:
- Electrons evolve with Vlasov-PIC
- Ions are a fixed uniform neutralizing background
- Periodic domain
- CIC (Cloud-in-Cell) deposition + gather
- Explicit leapfrog pusher
- Electric field solve via Poisson equation (FFT) and E = -∂ϕ/∂x

Diagnostics:
- |E_k(t)| for seeded mode and fitted damping rate gamma_fit
- Theoretical weak-damping estimate gamma_th (small k*lambda_D)
- Energy (kinetic, field, total), momentum
- Net charge integral
- Gauss law residual norm
- Phase space snapshots (downsampled)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# -----------------------------
# PIC utilities: CIC deposition and gather
# -----------------------------
def cic_deposit_charge_density(x, q, weight, Nx, L):
    """
    Deposit charge density rho(x) at Nx grid points using CIC (linear weighting).
    Grid nodes: x_i = i*dx, periodic.

    Each macro-particle carries charge q*weight.
    Returned rho has units of charge per unit length (1D), i.e. deposited charge / dx.
    """
    dx = L / Nx
    xi = x / dx
    i0 = np.floor(xi).astype(int) % Nx
    f = xi - np.floor(xi)           # fractional part in [0,1)
    i1 = (i0 + 1) % Nx

    rho = np.zeros(Nx, dtype=float)
    coeff = (q * weight) / dx
    np.add.at(rho, i0, coeff * (1.0 - f))
    np.add.at(rho, i1, coeff * f)
    return rho


def cic_gather(x, Fgrid, Nx, L):
    """Gather a nodal grid field Fgrid to particle positions x using CIC."""
    dx = L / Nx
    xi = x / dx
    i0 = np.floor(xi).astype(int) % Nx
    f = xi - np.floor(xi)
    i1 = (i0 + 1) % Nx
    return (1.0 - f) * Fgrid[i0] + f * Fgrid[i1]


# -----------------------------
# Field solve: Poisson equation -> electric field
# -----------------------------
def solve_phi_and_E_from_rho_poisson(rho, L, epsilon0=1.0):
    """
    Solve -d^2 phi/dx^2 = rho/epsilon0 (periodic) using FFT, then E = -dphi/dx.

    Steps:
    1) Remove the mean of rho to avoid the k=0 singularity (periodic neutrality).
    2) Solve in Fourier space: phi_k = rho_k/(epsilon0 * k^2) for k != 0.
    3) Differentiate spectrally: E_k = -i k phi_k, then inverse transform.
    4) Build midpoint rho (rho_c) for discrete Gauss residual checks.

    Returns:
      phi   : electrostatic potential (Nx,)
      E     : nodal electric field (Nx,)
      rho_c : midpoint rho used for Gauss residual checks (Nx,)
    """
    Nx = rho.size
    dx = L / Nx

    rho0 = rho - rho.mean()                     # enforce neutrality
    rho_c = 0.5 * (rho0 + np.roll(rho0, -1))    # midpoint values for diagnostics

    k = 2.0 * np.pi * np.fft.rfftfreq(Nx, d=dx)
    rho_hat = np.fft.rfft(rho0)

    phi_hat = np.zeros_like(rho_hat)
    nonzero = k != 0.0
    phi_hat[nonzero] = rho_hat[nonzero] / (epsilon0 * k[nonzero]**2)
    phi = np.fft.irfft(phi_hat, n=Nx)

    E_hat = -1j * k * phi_hat
    E = np.fft.irfft(E_hat, n=Nx)
    return phi, E, rho_c


def gauss_law_residual(E, rho_c, L, epsilon0=1.0):
    """Compute L2 norm of discrete Gauss residual: (E_{i+1}-E_i)/dx - rho_{i+1/2}/eps0."""
    Nx = E.size
    dx = L / Nx
    dEdx = (np.roll(E, -1) - E) / dx
    res = dEdx - rho_c / epsilon0
    return np.sqrt(np.mean(res**2))


# -----------------------------
# Landau damping theory (weak damping estimate)
# -----------------------------
def landau_gamma_weak(k, omega_p, v_t):
    """
    Weak-damping estimate for a Maxwellian (small k*lambda_D),
    using:
      omega_r^2 ~ omega_p^2 + 3 k^2 v_t^2
      gamma/omega_p ~ -(sqrt(pi)/2)/(k*lambda_D)^3 * exp[-1/(2(k*lambda_D)^2) - 3/2]
    where lambda_D = v_t/omega_p.

    Note: This is an asymptotic approximation; best when k*lambda_D is not too large
    and damping is weak.
    """
    lamD = v_t / omega_p
    kl = k * lamD
    if kl <= 0:
        return 0.0
    return -omega_p * (np.sqrt(np.pi) / 2.0) * (1.0 / kl**3) * np.exp(-1.0 / (2.0 * kl**2) - 1.5)


# -----------------------------
# Simulation
# -----------------------------
def run_landau_pic(
    Nx=256,
    ppc=400,
    L=4.0 * np.pi,         # choose so that m=1 gives k=0.5
    dt=0.1,
    nsteps=1200,
    v_t=1.0,               # Maxwellian stddev in v: f0 ~ exp(-v^2/(2 v_t^2))
    mode_m=1,
    alpha=1e-2,            # perturbation amplitude
    seed=1,
    diag_stride=1,
    snapshot_stride=300,
    live_plot=False,
    live_stride=None,
):
    """
    Run a Landau damping PIC simulation. Set live_plot=True to watch diagnostics update during the run.

    Normalized units:
      epsilon0 = 1, m = 1, |q| = 1, n0 = 1  => omega_p = 1
    """
    rng = np.random.default_rng(seed)

    # Normalized constants
    epsilon0 = 1.0
    me = 1.0
    qe = -1.0
    n0 = 1.0
    omega_p = np.sqrt(n0 * (qe**2) / (me * epsilon0))  # = 1 in these units

    dx = L / Nx
    Np = Nx * ppc
    weight = n0 * L / Np  # macro-particle weight so mean electron density = n0

    k = 2.0 * np.pi * mode_m / L
    gamma_th = landau_gamma_weak(k, omega_p, v_t)
    print(f"Theoretical weak-damping estimate: gamma_th = {gamma_th:.6g}")
    if live_stride is None:
        live_stride = diag_stride
    live_stride = max(1, live_stride)

    print("=== Landau damping PIC parameters ===")
    print(f"Nx={Nx}, dx={dx:.4g}, L={L:.4g}, k={k:.4g} (m={mode_m})")
    print(f"Np={Np} ({ppc} per cell), dt={dt:.4g}, nsteps={nsteps}, T={dt*nsteps:.4g}")
    print(f"v_t={v_t:.4g}, lambda_D=v_t/omega_p={v_t/omega_p:.4g}, k*lambda_D={k*v_t/omega_p:.4g}")
    print(f"alpha={alpha:.3g}")
    print(f"Weak-theory gamma_th={gamma_th:.6g} (negative = damping)")
    print("=====================================")

    # -----------------------------
    # Initialization (quiet-ish start)
    # -----------------------------
    # Positions: evenly spaced to reduce noise, then apply a small sinusoidal shift to seed density perturbation.
    x = (np.arange(Np) + 0.5) * (L / Np)
    x = x.astype(float)

    # Seed density perturbation ~ cos(kx) via small position shift:
    # x -> x + (alpha/k)*sin(kx) gives n(x) ≈ n0*(1 - alpha*cos(kx)) for small alpha.
    x = (x + (alpha / k) * np.sin(k * x)) % L

    # Velocities: Maxwellian using paired samples to reduce odd-moment noise (quiet start)
    half = Np // 2
    v_half_sample = v_t * rng.standard_normal(half)
    v = np.empty(Np, dtype=float)
    v[:half] = v_half_sample
    v[half:] = -v_half_sample  # enforce symmetry

    # Shuffle velocities to decorrelate with positions
    rng.shuffle(v)

    # -----------------------------
    # Initial field and leapfrog alignment
    # -----------------------------
    rho = cic_deposit_charge_density(x, qe, weight, Nx, L) + n0  # add ion background (+n0)
    _, E, rho_c = solve_phi_and_E_from_rho_poisson(rho, L, epsilon0)
    Ep = cic_gather(x, E, Nx, L)

    # Leapfrog: store v at half-step
    v_half = v + 0.5 * (qe / me) * Ep * dt

    # -----------------------------
    # Diagnostics storage
    # -----------------------------
    times = []
    Ek_mode = []
    KE = []
    FE = []
    TE = []
    P = []
    netQ = []
    gaussL2 = []
    energy_error = []
    growth_rate = []
    initial_energy = None
    snapshots = []

    x_grid = np.arange(Nx) * dx
    phase_stride = 1

    # Optional live plotting (similar to two-stream diagnostic panel)
    if live_plot:
        plt.ion()
        fig, axes = plt.subplots(2, 2, figsize=(13.5, 5.5))
        phase_ax, field_ax, energy_ax, dist_ax = axes.flatten()
        growth_ax = energy_ax.twinx()
        growth_ax.spines["right"].set_position(("axes", 1.14))
        growth_ax.yaxis.set_label_position("right")
        growth_ax.yaxis.tick_right()
        fig.tight_layout()
        phase_stride = max(1, Np // 4000)

    # -----------------------------
    # Time loop
    # -----------------------------
    for n in range(nsteps):
        t = (n + 1) * dt

        # 1) Deposit total charge density
        rho = cic_deposit_charge_density(x, qe, weight, Nx, L) + n0

        # Net charge integral (should be ~0)
        net_charge = rho.mean() * L

        # 2) Solve Poisson for phi then E = -dphi/dx
        _, E, rho_c = solve_phi_and_E_from_rho_poisson(rho, L, epsilon0)

        # 3) Gather
        Ep = cic_gather(x, E, Nx, L)

        # 4) Push (explicit leapfrog)
        v_half += (qe / me) * Ep * dt
        x = (x + v_half * dt) % L

        # Diagnostics
        if (n % diag_stride) == 0:
            prev_amp = Ek_mode[-1] if Ek_mode else None
            prev_t = times[-1] if times else None

            # Mode amplitude |E_k| from FFT (normalize by Nx)
            Ehat = np.fft.rfft(E) / Nx
            mode_amp = np.abs(Ehat[mode_m])

            kinetic = 0.5 * me * weight * np.sum(v_half**2)
            field = 0.5 * epsilon0 * np.sum(E**2) * dx
            total = kinetic + field
            if initial_energy is None:
                initial_energy = total
            rel_energy_err = abs(total - initial_energy) / abs(initial_energy)

            momentum = me * weight * np.sum(v_half)
            gl2 = gauss_law_residual(E, rho_c, L, epsilon0)

            times.append(t)
            Ek_mode.append(mode_amp)
            KE.append(kinetic)
            FE.append(field)
            TE.append(total)
            energy_error.append(max(rel_energy_err, 1e-40))
            P.append(momentum)
            netQ.append(net_charge)
            gaussL2.append(gl2)
            tiny = 1e-40
            if prev_amp is None or prev_t is None:
                growth_rate.append(np.nan)
            else:
                gamma_inst = (np.log(mode_amp + tiny) - np.log(prev_amp + tiny)) / (t - prev_t)
                growth_rate.append(gamma_inst)

            # Live diagnostics panel
            if live_plot and (n % live_stride) == 0:
                phase_ax.cla()
                phase_ax.scatter(x[::phase_stride], v_half[::phase_stride], s=0.5, marker='.')
                phase_ax.set_xlim(0, L)
                phase_ax.set_ylim(-4, 4)
                phase_ax.set_xlabel("x")
                phase_ax.set_ylabel("v")
                phase_ax.legend(
                    (mpatches.Patch(color='w'),),
                    (r'$\omega_{pe}t=$' + f"{t:.2f}",),
                    loc=1,
                    frameon=False,
                )

                field_ax.cla()
                field_ax.plot(x_grid, E, label="E", linewidth=2)
                field_ax.set_xlim(0, L)
                field_ax.set_xlabel("x")
                field_ax.set_ylabel("E")
                field_ax.legend(loc=1)

                energy_ax.cla()
                growth_ax.cla()
                energy_ax.set_xlim(0, nsteps * dt)
                energy_ax.set_xlabel("time")
                energy_ax.set_yscale("log")
                energy_ax.set_ylabel("Energy")
                energy_ax.plot(times, FE, label="Field", linewidth=2)
                energy_ax.plot(times, KE, label="Kinetic", linewidth=2)
                energy_ax.plot(times, TE, label="Total", linestyle="--", linewidth=2)
                energy_ax.set_title(f"Energies - grow rate th {gamma_th :.4e}")
                if FE:
                    times_arr = np.asarray(times)
                    fe0 = FE[0]
                    fe_theory = fe0 * np.exp(2.0 * gamma_th * (times_arr - times_arr[0]))
                    energy_ax.plot(times_arr, fe_theory, label="Field (theory)", linestyle="--", linewidth=1.8, color="tab:green")
                energy_ax.plot(
                    times,
                    energy_error,
                    label=f"|dE|/E0 (last={energy_error[-1]:.3e})",
                    color="k",
                    linestyle=":",
                    linewidth=2,
                )
                energy_ax.set_ylim(1e-12, max(TE) * 1.2)

                #growth_ax.spines["right"].set_position(("axes", 1))
                #growth_ax.yaxis.set_label_position("right")
                #growth_ax.set_yscale("log")
                #growth_ax.yaxis.tick_right()
                #growth_ax.set_ylabel("Damping rate γ (inst)", labelpad=12)
                #growth_ax.plot(times, growth_rate, label="γ_inst", color="tab:purple", linewidth=1.8)
                #growth_ax.axhline(gamma_th, color="tab:green", linestyle="--", linewidth=1.2, label="γ_theory")

                lines, labels = energy_ax.get_legend_handles_labels()
                lines_g, labels_g = growth_ax.get_legend_handles_labels()
                #growth_ax.legend(lines + lines_g, labels + labels_g, loc=4)
                growth_ax.legend(lines, labels, loc=4)

                dist_ax.cla()
                bins, edges = np.histogram(v_half, bins=40, range=(-6, 6))
                centers = 0.5 * (edges[:-1] + edges[1:])
                dist_ax.plot(centers, bins, label="f(v)", linewidth=2)
                dist_ax.set_xlim(-6, 6)
                dist_ax.set_xlabel("v")
                dist_ax.legend(loc=1)

                fig.canvas.draw_idle()
                plt.pause(0.05)

        if (n % snapshot_stride) == 0:
            sample = min(6000, Np)
            idx = rng.choice(Np, size=sample, replace=False)
            snapshots.append((t, x[idx].copy(), v_half[idx].copy()))

    if live_plot:
        plt.ioff()

    # Convert to arrays
    times = np.array(times)
    Ek_mode = np.array(Ek_mode)
    KE = np.array(KE)
    FE = np.array(FE)
    TE = np.array(TE)
    P = np.array(P)
    netQ = np.array(netQ)
    gaussL2 = np.array(gaussL2)
    energy_error = np.array(energy_error)
    growth_rate = np.array(growth_rate)

    # -----------------------------
    # Fit damping rate gamma from ln|E_k|
    # Choose an interval early enough before particle noise dominates.
    # Simple heuristic: fit over the middle portion where signal is above noise floor.
    # -----------------------------
    tiny = 1e-40
    logEk = np.log(Ek_mode + tiny)

    # Heuristic window: from first 10% to 50% of run
    i0 = int(0.10 * len(times))
    i1 = int(0.50 * len(times))
    gamma_fit = np.nan
    if i1 - i0 >= 10:
        coef = np.polyfit(times[i0:i1], logEk[i0:i1], 1)
        gamma_fit = coef[0]  # slope

    results = {
        "params": {
            "Nx": Nx, "Np": Np, "L": L, "dx": dx, "dt": dt, "nsteps": nsteps,
            "v_t": v_t, "omega_p": omega_p, "lambda_D": v_t/omega_p,
            "k": k, "mode_m": mode_m, "alpha": alpha,
            "gamma_th": gamma_th, "gamma_fit": gamma_fit,
        },
        "time": times,
        "Ek_mode": Ek_mode,
        "KE": KE,
        "FE": FE,
        "TE": TE,
        "P": P,
        "netQ": netQ,
        "gaussL2": gaussL2,
        "energy_error": energy_error,
        "growth_rate": growth_rate,
        "snapshots": snapshots,
    }
    return results


def plot_results(results):
    p = results["params"]
    t = results["time"]

    print("\n=== Diagnostics summary ===")
    print(f"k*lambda_D = {p['k']*p['lambda_D']:.4g}")
    print(f"gamma_th   = {p['gamma_th']:.6g} (weak-damping estimate)")
    print(f"gamma_fit  = {p['gamma_fit']:.6g} (from ln|E_k| slope)")
    print(f"max |netQ| = {np.max(np.abs(results['netQ'])):.3e} (integral rho dx)")
    print(f"max Gauss L2 residual = {np.max(results['gaussL2']):.3e}")
    rel_dE = (results["TE"][-1] - results["TE"][0]) / results["TE"][0]
    print(f"relative total energy change = {rel_dE:.3e}")
    print("===========================\n")

    # |E_k|
    plt.figure()
    plt.semilogy(t, results["Ek_mode"] + 1e-40)
    plt.xlabel("t")
    plt.ylabel(r"$|E_k|$")
    plt.title("Landau damping: mode amplitude (semilog)")
    plt.grid(True)

    # Energies
    plt.figure()
    plt.plot(t, results["KE"], label="Kinetic")
    plt.plot(t, results["FE"], label="Field")
    plt.plot(t, results["TE"], label="Total")
    fe0 = results["FE"][0] if len(results["FE"]) else None
    if fe0 is not None:
        fe_theory = fe0 * np.exp(2.0 * p["gamma_th"] * (t - t[0]))
        plt.plot(t, fe_theory, label="Field (theory)", linestyle="--", linewidth=1.8, color="tab:green")
    plt.xlabel("t")
    plt.ylabel("Energy")
    plt.title("Energy history")
    plt.legend()
    plt.grid(True)

    # Momentum
    plt.figure()
    plt.plot(t, results["P"])
    plt.xlabel("t")
    plt.ylabel("Total momentum")
    plt.title("Momentum history")
    plt.grid(True)

    # Gauss residual
    plt.figure()
    plt.semilogy(t, results["gaussL2"] + 1e-40)
    plt.xlabel("t")
    plt.ylabel("Gauss residual (L2)")
    plt.title("Discrete Gauss law residual")
    plt.grid(True)

    # Phase space snapshots (a few)
    for (ts, xs, vs) in results["snapshots"][:4]:
        plt.figure()
        plt.scatter(xs, vs, s=1)
        plt.xlabel("x")
        plt.ylabel("v")
        plt.title(f"Phase space (sample) at t={ts:.3g}")
        plt.grid(True)

    plt.show()


if __name__ == "__main__":
    results = run_landau_pic(
        Nx=256,
        ppc=800,
        L=4.0*np.pi,     # so k=0.5 for m=1
        dt=0.02,
        nsteps=3200,
        v_t=1.0,
        mode_m=1,
        alpha=1e-1,
        seed=2,
        diag_stride=1,
        snapshot_stride=300,
        live_plot=True,
        live_stride=100,
    )
    plot_results(results)
