import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import sarsim

stack_size = 6
n_samples = 10000

# generate covariance matrix with exponential coherence loss and linearly changing phase
coh_vals = sarsim.coh.exp_decay_coh_mat(stack_size, 0.5)
phi_vals = np.exp(1j * np.linspace(-np.pi, np.pi, stack_size))
cov_mat = np.outer(phi_vals, phi_vals.conj()) * coh_vals

# circular complex Gaussian random samples
samples = sarsim.random.multivariate_complex_normal(cov_mat, n_samples)

scm = np.cov(samples.T)

############
#          #
# Plotting #
#          #
############

fig = plt.figure(figsize=(9, 3))
fig.subplots_adjust(top=0.9, bottom=0.18, left=0.02, right=0.99)

ax_abs = fig.add_subplot(1, 3, 1)
divider = make_axes_locatable(ax_abs)
cax_abs = divider.append_axes("right", size="5%", pad=0.05)

im_abs = ax_abs.matshow(np.abs(scm), vmin=0, vmax=1)
ax_abs.set_title('SCM abs. value', y=-0.25)
cbar_abs = fig.colorbar(im_abs, cax=cax_abs)

ax_phi = fig.add_subplot(1, 3, 2)
divider = make_axes_locatable(ax_phi)
cax_phi = divider.append_axes("right", size="5%", pad=0.05)

im_phi = ax_phi.matshow(
    np.angle(scm), vmin=-np.pi, vmax=np.pi, cmap='twilight_shifted')
ax_phi.set_title('SCM phase', y=-0.25)

cbar_phi = fig.colorbar(
    im_phi, cax=cax_phi, ticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
cbar_phi.ax.set_yticklabels(
    [r'$-\pi$', r'$-\frac{\pi}{2}$', '$0$', r'$\frac{\pi}{2}$', r'$\pi$'])

ax_phi_hist = fig.add_subplot(1, 3, 3)
for idx in range(stack_size - 1, 0, -2):
    # compensated for phase
    true_phase = phi_vals[idx].conj() * phi_vals[0]
    ax_phi_hist.hist(
        np.angle(true_phase.conj() * samples[:, idx].conj() * samples[:, 0]),
        range=(-np.pi, np.pi),
        bins=30,
        label=r'$t_\Delta = {}$'.format(idx),
        density=True,
        alpha=0.5)
ax_phi_hist.set_title('phase histograms', y=-0.25)
ax_phi_hist.legend()
ax_phi_hist.set_xticks(cbar_phi.get_ticks())
ax_phi_hist.set_xticklabels(cbar_phi.ax.get_yticklabels())

plt.show()
