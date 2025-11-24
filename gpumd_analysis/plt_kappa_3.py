from pylab import *
import os
from scipy.integrate import cumulative_trapezoid

aw = 2
fs = 16
font = {'size': fs}
matplotlib.rc('font', **font)
matplotlib.rc('axes', linewidth=aw)

def set_fig_properties(ax_list):
    tl = 8
    tw = 2
    tlm = 4
    for ax in ax_list:
        ax.tick_params(which='major', length=tl, width=tw)
        ax.tick_params(which='minor', length=tlm, width=tw)
        ax.tick_params(which='both', axis='both', direction='in', right=True, top=True)

labels_kappa = ['kxi', 'kxo', 'kyi', 'kyo', 'kz']

def load_kappa(path):
    arr = np.loadtxt(path)
    d = {}
    for i, key in enumerate(labels_kappa):
        d[key] = arr[:, i]
    return d

def running_ave(y, x):
    return cumulative_trapezoid(y, x, initial=0) / x

k1 = load_kappa('fold1/kappa.out')
k2 = load_kappa('fold2/kappa.out')
k3 = load_kappa('fold3/kappa.out')

n = min(k1['kz'].shape[0], k2['kz'].shape[0], k3['kz'].shape[0])
t = np.arange(1, n + 1) * 0.001

k1_ra = running_ave(k1['kz'][:n], t)
k2_ra = running_ave(k2['kz'][:n], t)
k3_ra = running_ave(k3['kz'][:n], t)
mean_ra = (k1_ra + k2_ra + k3_ra) / 3.0

set_fig_properties([gca()])
plot(t, k1_ra, linewidth=2, color='C0')
plot(t, k2_ra, linewidth=2, color='C1')
plot(t, k3_ra, linewidth=2, color='C2')
plot(t, mean_ra, linewidth=2, color='k')
ylim([-100, 2000])
xlabel('time (ns)')
ylabel(r'$\kappa$ (W/m/K)')
legend(['fold1', 'fold2', 'fold3', 'mean'])

print('最后时间的平均热导率:', float(mean_ra[-1]))

os.makedirs('figs', exist_ok=True)
tight_layout()
savefig('figs/kappa_mean.png', dpi=300)
show()