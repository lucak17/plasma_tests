import numpy as np   #array syntax
import pylab as plt  #plot

import matplotlib.patches as mpatches   #plot

from scipy import sparse   #special functions, optimization, linear algebra
from scipy.sparse import linalg
#from poisson import poisson_solver
import time


# Timing
time_start = time.time()


# Set plotting parameters
params = {'axes.labelsize': 'large',
              'xtick.labelsize': 'medium',
              'ytick.labelsize': 'medium',
              'font.size': 15,
              'font.family': 'sans-serif',
              'text.usetex': False,
              'mathtext.fontset': 'stixsans',}

plt.rcParams.update(params)
plt.ion()  # enable live updating


# Simulation parameters

# Simulation parameters
L = 2*np.pi # Domain size
DT = 0.05 # Time step
NT = 3000 # 300  # Number of time steps

TOut = round(NT/100) # Output period

NG = 64 # Number of grid cells
N = NG * 100 # Number of particles
WP = 1. # Plasma frequency
QM = -1. # Charge/mass ratio
V0 = 0.1 #0.2 # Stream velocity
VT = 0.00 # Thermal speed

# perturbation
XP1 = 0.000
mode = 3


Q = WP**2 / (QM*N/L)  # rho0*L/N: charge carried by a single particle
rho_back = -Q*N/L  # Background charge density?
dx = L / NG # Grid step


# Auxilliary vectors
p = np.concatenate([np.arange(N), np.arange(N)])  # Some indices up to N
Poisson = sparse.spdiags(([1, -2, 1] * np.ones((1, NG-1), dtype=int).T).T, [-1, 0, 1], NG-1, NG-1)
Poisson = Poisson.tocsc()


# Cell center coordinates
xg = np.linspace(0, L-dx, NG) + dx/2


# electrons
xp = np.linspace(0, L-L/N, N).T   # Particle positions
vp = VT * np.random.randn(N) # particle thermal spread
pm = np.arange(N)
pm = 1 - 2 * np.mod(pm+1, 2)
vp += pm * V0 # Drift plus thermal spread



# Add electron perturbation to excite the desired mode
xp += XP1 * np.cos(2 * np.pi * mode / L * xp)
xp[np.where(xp < 0)] += L
xp[np.where(xp >= L)] -= L



histEnergy, histPotE, histKinE, histEnergyError, t = [], [], [], [], []
initial_energy = None

# Live-plot figure/axes
fig, axes = plt.subplots(2, 2, figsize=(13.5, 5.5))
phase_ax, field_ax, energy_ax, dist_ax = axes.flatten()
error_ax = energy_ax.twinx()
fig.tight_layout()



# Main cycle
for it in range(NT+1):

    # update particle position xp
    xp += vp * DT
    # Periodic boundary condition
    xp[np.where(xp < 0)] += L
    xp[np.where(xp >= L)] -= L

    # Project particles->grid
    g1 = np.floor(xp/dx - 0.5)
    g = np.concatenate((g1, g1+1))
    fraz1 = 1 - np.abs(xp/dx - g1 - 0.5)
    fraz = np.concatenate((fraz1, 1-fraz1))
    g[np.where(g < 0)] += NG
    g[np.where(g > NG-1)] -= NG


    mat = sparse.csc_matrix((fraz, (p, g)), shape=(N, NG))
    rho = Q / dx * mat.toarray().sum(axis=0) + rho_back


    # Compute electric field potential
    #
    # Here where we need to substitute with the quantum part #
    #
    Phi = linalg.spsolve(Poisson, -dx**2 * rho[0:NG-1])
    Phi = np.concatenate((Phi,[0]))

    # Electric field on the grid
    Eg = (np.roll(Phi, 1) - np.roll(Phi, -1)) / (2*dx)

    # Electric field fft

    #ft = abs(scipy.fft(Eg))
    #k = scipy.fftpack.fftfreq(Eg.size,xg[1]-xg[0])

    # interpolation grid->particle and velocity update
    vp += mat * QM * Eg * DT

    bins,edges=np.histogram(vp,bins=200,range=(-1,1),density=True)
    left,right = edges[:-1],edges[1:]
    vc = np.array([left,right]).T.flatten()
    fv = np.array([bins,bins]).T.flatten()


    # Energies
    field_energy = 0.5 * (Eg**2).sum() * dx
    kinetic_energy = 0.5 * Q/QM * (vp**2).sum()
    total_energy = field_energy + kinetic_energy
    if initial_energy is None:
        initial_energy = total_energy

    histEnergy.append(total_energy)
    histPotE.append(field_energy)
    histKinE.append(kinetic_energy)
    # Use a tiny floor to keep log scale happy when the error is zero
    energy_error = abs(total_energy - initial_energy) / abs(initial_energy)
    histEnergyError.append(max(energy_error, 1e-30))
    t.append(it*DT)

    if (np.mod(it, TOut) == 0):

        # Phase space
        phase_ax.cla()
        phase_ax.scatter(xp[0:-1:2], vp[0:-1:2], s=0.5, marker='.', color='blue')
        phase_ax.scatter(xp[1:-1:2], vp[1:-1:2], s=0.5, marker='.', color='red')
        phase_ax.set_xlim(0, L)
        phase_ax.set_ylim(-0.4, 0.4)
        phase_ax.set_xlabel('x')
        phase_ax.set_ylabel('v')
        phase_ax.legend((mpatches.Patch(color='w'), ), (r'$\omega_{pe}t=$' + str(DT*it), ), loc=1, frameon=False)

        # Electric field
        field_ax.cla()
        field_ax.set_xlim(0, L)
        field_ax.set_ylim(-0.15, 0.15)
        field_ax.set_xlabel('x')
        field_ax.set_ylabel('$\Phi$')
        field_ax.plot(xg, Phi, label='$\Phi$', linewidth=2)
        field_ax.legend(loc=1)

        # Energies
        energy_ax.cla()
        error_ax.cla()
        energy_ax.set_xlim(0, NT*DT)
        energy_ax.set_xlabel('time')
        energy_ax.set_yscale('log')
        energy_ax.set_ylabel('Energy')
        energy_ax.plot(t, histPotE, label='Potential', linewidth=2)
        energy_ax.plot(t, histKinE, label='Kinetic', linewidth=2)
        energy_ax.plot(t, histEnergy, label='Total Energy', linestyle='--', linewidth=2)

        #error_ax.set_ylabel('Energy error (|ΔE|/E0)')
        #error_ax.set_yscale('log')
        error_ax.plot(t, histEnergyError, label=f"|ΔE|/E0 (last={histEnergyError[-1]:.3e})", color='k', linestyle=':', linewidth=2)

        lines, labels = energy_ax.get_legend_handles_labels()
        lines_err, labels_err = error_ax.get_legend_handles_labels()
        error_ax.legend(lines + lines_err, labels + labels_err, loc=4)

        # Electron distribution function
        dist_ax.cla()
        dist_ax.set_xlim(-1, 1)
        #plt.ylim(0, N/2)
        dist_ax.set_xlabel('v')
        dist_ax.plot(vc,fv, label='f(v)', linewidth=2)
        dist_ax.legend(loc=1)

        print(it)


        fig.canvas.draw_idle()
        plt.pause(0.05)

print('Time elapsed: ', time.time() - time_start)
plt.ioff()
plt.show()
