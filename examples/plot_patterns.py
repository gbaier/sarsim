from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import sarsim


shape = (129, 129)

areas = OrderedDict()
areas[sarsim.patterns.plateau] = {'shape': shape, 'mid_width': 32}
areas[sarsim.patterns.const_slope] = {'shape': shape}
areas[sarsim.patterns.sine] = {'shape': shape, 'omega': 0.1}
areas[sarsim.patterns.logfreq] = {'shape': shape, 'doublerate': 43}
areas[sarsim.patterns.step_slope] = {'shape': shape}
areas[sarsim.patterns.unit_step] = {'shape': shape}
areas[sarsim.patterns.raised_cos] = {'shape': shape}
areas[sarsim.patterns.banana] = {'shape': shape}
areas[sarsim.patterns.peaks] = {'shape': shape}
areas[sarsim.patterns.zebra] = {'shape': shape}
areas[sarsim.patterns.logbar] = {'shape': shape, 'doublerate': 43}
areas[sarsim.patterns.squares] = {'shape': shape}
areas[sarsim.patterns.checkers] = {'shape': shape}
areas[sarsim.terrain.diamond_square] = {'size': shape[0], 'seed': 42}
areas[sarsim.patterns.pizza] = {'shape': shape, 'nslices': 16}

PLOTS = [('Unwrapped Phase', lambda x: x, {
    'cmap': plt.get_cmap('viridis')
}), ('Wrapped Phase', lambda x: sarsim.util.wrap_phase(5 * np.pi * x), {
    'cmap': plt.get_cmap('hsv'),
    'vmin': -np.pi,
    'vmax': np.pi
})]

for (name, modder, plot_opts) in PLOTS:
    fig = plt.figure(figsize=(8, 5.2))
    fig.suptitle(name, fontsize=12)
    fig.subplots_adjust(
        left=0.05,
        right=0.95,
        wspace=0.2,
        hspace=0.2,
        bottom=0.05,
        top=0.9)
    gs = gridspec.GridSpec(3, 5)

    for idx, (method, args) in enumerate(areas.items()):
        ax = plt.subplot(gs[idx])
        ax.imshow(modder(method(**args)), **plot_opts)
        ax.set_title(method.__name__, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    filename = 'sarsim_{}.png'.format(name.lower().replace(' ', '_'))

plt.show()
