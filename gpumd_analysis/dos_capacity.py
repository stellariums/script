from pylab import *
from ase.build import graphene_nanoribbon
from ase.io import write
#%%
aw = 2
fs = 16
font = {'size'   : fs}
matplotlib.rc('font', **font)
matplotlib.rc('axes' , linewidth=aw)

def set_fig_properties(ax_list):
    tl = 8
    tw = 2
    tlm = 4

    for ax in ax_list:
        ax.tick_params(which='major', length=tl, width=tw)
        ax.tick_params(which='minor', length=tlm, width=tw)
        ax.tick_params(which='both', axis='both', direction='in', right=True, top=True)

#%%
dos_array = np.loadtxt("dos.out")
dos = {}
dos["omega"], dos['DOSx'], dos['DOSy'], dos['DOSz'] = dos_array[:,0], dos_array[:,1], dos_array[:,2], dos_array[:,3]
dos['DOSxyz'] = dos['DOSx']+dos['DOSy']+dos['DOSz']
dos["nu"] = dos["omega"] / (2 * np.pi)

vac_array = np.loadtxt("mvac.out")
vac = {}
vac["t"], vac['VACx'], vac['VACy'], vac['VACz'] = vac_array[:,0], vac_array[:,1], vac_array[:,2], vac_array[:,3]
vac['VACxyz'] = vac['VACx']+vac['VACy']+vac['VACz']
vac['VACxyz'] /= vac['VACxyz'].max()

print('DOS:', list(dos.keys()))
print('VAC:', list(vac.keys()))

#%%
figure(figsize=(12,10))
subplot(2,2,1)
set_fig_properties([gca()])
plot(vac['t'], vac['VACx']/vac['VACx'].max(), color='C3',linewidth=3)
plot(vac['t'], vac['VACy']/vac['VACy'].max(), color='C0', linestyle='--',linewidth=3)
plot(vac['t'], vac['VACz']/vac['VACz'].max(), color='C2', linestyle='-.',zorder=100,linewidth=3)
xlim([0, 0.25])
gca().set_xticks([0,0.25])
ylim([-0.5, 1])
gca().set_yticks([-0.5,0,0.5,1])
ylabel('VAC (Normalized)')
xlabel('Correlation Time (ps)')
legend(['x','y', 'z'])
title('(a)')

subplot(2,2,2)
set_fig_properties([gca()])
plot(dos['nu'], dos['DOSx'], color='C3',linewidth=3)
plot(dos['nu'], dos['DOSy'], color='C0', linestyle='--',linewidth=3)
plot(dos['nu'], dos['DOSz'], color='C2', linestyle='-.',zorder=100, linewidth=3)
xlim([0, 60])
gca().set_xticks(range(0,61,20))
#ylim([0, 1500])
gca().set_yticks(np.arange(0,1501,500))
ylabel('PDOS (1/THz)')
xlabel(r'$\nu$ (THz)')
legend(['x','y', 'z'])
title('(b)')

subplot(2,2,3)
set_fig_properties([gca()])
plot(vac['t'], vac['VACxyz'], color='k',linewidth=3)
xlim([0,0.25])
gca().set_xticks([0,0.25])
ylim([-0.5, 1])
gca().set_yticks([-0.5,0,0.5,1])
ylabel('VAC (Normalized)')
xlabel('Correlation Time (ps)')
title('(c)')

subplot(2,2,4)
set_fig_properties([gca()])
plot(dos['nu'], dos['DOSxyz'], color='k',linewidth=3)
xlim([0, 60])
gca().set_xticks(range(0,61,20))
#ylim([0, 2500])
#gca().set_yticks(np.arange(0,2501,500))
ylabel('PDOS (1/THz)')
xlabel(r'$\nu$ (THz)')
title('(d)')

tight_layout()
savefig('dos.png')
show()

#%%
temperatures = np.arange(250,501,50)  # [K]
with open('model.xyz','r') as fp:
    num_atoms = int(fp.readline())
    Volume = fp.readline()
lattice_values = Volume.split('Lattice="')[1].split('"')[0].split()
lattice_floats = list(map(float, lattice_values))
lattice_matrix = np.array(lattice_floats).reshape(3, 3)
volume = np.linalg.det(lattice_matrix)

Cx, Cy, Cz = list(), list(), list()  # [k_B/atom] Heat capacity per atom
hnu = 6.63e-34*dos['nu']*1.e12  # [J]

for temperature in temperatures:
    kBT = 1.38e-23*temperature  # [J]
    x = hnu/kBT
    expr = np.square(x)*np.exp(x)/(np.square(np.expm1(x)))
    Cx.append(np.trapz(dos['DOSx']*expr, dos['nu'])/num_atoms)
    Cy.append(np.trapz(dos['DOSy']*expr, dos['nu'])/num_atoms)
    Cz.append(np.trapz(dos['DOSz']*expr, dos['nu'])/num_atoms)

figure(figsize=(8,6))
set_fig_properties([gca()])
mew, ms, mfc, lw = 1, 8, 'none', 2.5
for i in range(0,len(Cx)):
    # Cx[i] = Cx[i] *1.38e-23/1.992e-26                       # kB/atom to J/kg/K
    # Cy[i] = Cy[i] *1.38e-23/1.992e-26                       # kB/atom to J/kg/K
    # Cz[i] = Cz[i] *1.38e-23/1.992e-26                       # kB/atom to J/kg/K
    Cx[i] = Cx[i] *1.38e-23*num_atoms/volume*1e30                      # kB/atom to J/m3/K
    Cy[i] = Cy[i] *1.38e-23*num_atoms/volume*1e30                        # kB/atom to J/m3/K
    Cz[i] = Cz[i] *1.38e-23*num_atoms/volume*1e30
     
plot(temperatures, Cx, lw=lw,marker='d',mfc=mfc,ms=ms,mew=mew)
plot(temperatures, Cy, lw=lw,marker='s',mfc=mfc,ms=ms,mew=mew)
plot(temperatures, Cz, lw=lw,marker='o',mfc=mfc,ms=ms,mew=mew)
xlim([250,500])
gca().set_xticks(range(250,500,50))
#ylim([0, 1.1])
# gca().set_yticks(np.linspace(0,1,6))
#ylabel(r'Heat Capacity (k$_B$/atom)')
# ylabel(r'Heat Capacity $(J \cdot kg^{-1} \cdot K^{-1})$')
ylabel(r'Heat Capacity $(J \cdot m^{-3} \cdot K^{-1})$')
xlabel('Temperature (K)')
legend(['x','y','z'])

tight_layout()
savefig('heat_capacity.png')
show()

with open('heat_capacity.txt','w') as fp:
    # fp.write("heat capacity of x in 300K: {} J/kg/K\n".format(Cx[1]))
    # fp.write("heat capacity of y in 300K: {} J/kg/K\n".format(Cy[1]))
    # fp.write("heat capacity of z in 300K: {} J/kg/K\n".format(Cz[1]))
    # fp.write("total capacity in 300K: {} J/kg/K\n".format(Cx[1]+Cy[1]+Cz[1]))
    fp.write("heat capacity of x in 300K: {} J/m3/K\n".format(Cx[1]))
    fp.write("heat capacity of y in 300K: {} J/m3/K\n".format(Cy[1]))
    fp.write("heat capacity of z in 300K: {} J/m3/K\n".format(Cz[1]))
    fp.write("total capacity in 300K: {} J/m3/K\n".format(Cx[1]+Cy[1]+Cz[1]))