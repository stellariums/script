"""
    Run:
        python plot_nep_2024.py
    Author:
        Ke Xu <twtdq(at)qq.com>
"""


from pylab import *

##set figure properties
aw = 2.0    #轴线宽度
fs = 28     #字体大小
lw = 4.0    #线宽
font = {'size'   : fs}  #设置字体大小
matplotlib.rc('font', **font)#设置字体大小
matplotlib.rc('axes' , lw=aw)#设置轴线
plt.rcParams['font.family'] = 'Arial'

def set_fig_properties(ax_list):   #刻度线设置
    tl = 6
    tw = 1.5
    tlm = 3

    for ax in ax_list:
        ax.tick_params(which='major', length=tl, width=tw)  #主刻度线
        ax.tick_params(which='minor', length=tlm, width=tw) #次刻度线
        ax.tick_params(which='both', axis='both', direction='in', right=False, top=False)  #刻度线向外

def plot_nep(pout):     #绘制nep.txt文件数据汇总的直方图和散点图 ？？？
    nep = np.loadtxt("./nep.txt", skiprows=6) #跳过6行
    figure(figsize=(16, 7))
    plt.subplot(1,2,1)
    plt.hist(np.log(np.abs(nep)), bins=50)
    plt.subplot(1,2,2)
    plt.scatter(range(len(nep)), nep, s=0.5)
    plt.gcf().set_size_inches(9,3)
    plt.savefig(pout, dpi=300)


def com_RMSE(fin):  #
    nclo = int(fin.shape[1]/2)
    pids = fin[:, nclo] > -1e5
    targe = fin[pids, :nclo].reshape(-1)
    predi = fin[pids, nclo:].reshape(-1)
    return np.sqrt(((predi - targe) ** 2).mean())


loss = loadtxt('loss.out')
loss[:,0] = np.arange(1, len(loss) + 1)*100
print("We have run %s steps!"%loss[-1, 0])
energy_train = loadtxt('energy_train.out')
force_train = loadtxt('force_train.out')
virial_train = loadtxt('virial_train.out')
stress_train = loadtxt('stress_train.out')
# print("{:.3f}".format(com_RMSE(energy_train)*1000), end=" ")
# print("{:.3f}".format(com_RMSE(force_train)*1000), end=" ")
# print("{:.3f}".format(com_RMSE(virial_train)*1000))
# print("{:.3f}".format(com_RMSE(stress_train)*1000))

test_flag = 0               #设置是否有测试集
if test_flag == 1:
    energy_test = loadtxt('energy_test.out')
    force_test = loadtxt('force_test.out')
    virial_test = loadtxt('virial_test.out')
    stress_test = loadtxt('stress_test.out')

figure(figsize=(20, 15)) #20*15的大小进行loss函数的绘制
subplot(2, 2, 1)
set_fig_properties([gca()])
loglog(loss[:, 0], loss[:, 1],  ls="-", lw=lw, c = "C1", label="Total")
loglog(loss[:, 0], loss[:, 2],  ls="-", lw=lw, c = "C4", label=r"$L_{1}$")
loglog(loss[:, 0], loss[:, 3],  ls="-", lw=lw, c = "C5", label=r"$L_{2}$")
loglog(loss[:, 0], loss[:, 4],  ls="-", lw=lw, c = "C0", label="Energy_train")
loglog(loss[:, 0], loss[:, 5],  ls="-", lw=lw, c = "C2", label="Force_train")
loglog(loss[:, 0], loss[:, 6],  ls="-", lw=lw, c = "C3", label="Virial_train")

#if test_flag == 1:
#    loglog(loss[:, 0], loss[:, 7],  ls="--", lw=lw, c = "C6", label="Energy_test")
#    loglog(loss[:, 0], loss[:, 8],  ls="--", lw=lw, c = "C7", label="Force_test")
#    loglog(loss[:, 0], loss[:, 9],  ls="--", lw=lw, c = "C8", label="Virial_test")

#xlim([1e2, 10e5])
#ylim([1e-3, 5e0])
xlabel('Generation')
ylabel('Loss')
#添加图例 
legend(loc="lower left",  #在左下角；
        ncol = 2,               #两列显示
        fontsize = 18,          #字体大小
        frameon = False,    #不显示图例边框
        columnspacing = 0.1)  #列间空间为0.2
        
        

subplot(2, 2, 2)
set_fig_properties([gca()])
if test_flag == 1:
    ene_min = np.min([np.min(energy_train),np.min(energy_test)])
    ene_max = np.max([np.max(energy_train),np.max(energy_test)])
else:
    ene_min = np.min(energy_train)
    ene_max = np.max(energy_train)
ene_min -= (ene_max-ene_min)*0.1
ene_max += (ene_max-ene_min)*0.1

plot([ene_min, ene_max], [ene_min, ene_max], c = "grey", lw = 4)
plot(energy_train[:, 1], energy_train[:, 0], 'o', c="C0", ms = 6, label="Train dataset (RMSE={0:4.2f} meV/atom)".format(loss[-1, 4]*1000))






if test_flag == 1:
    plot(energy_test[:, 1], energy_test[:, 0], 'o', c="C6", ms = 4,label="Test dataset")
 #   text(ene_min*0.9+ene_max*0.1, ene_min*0.25+ene_max*0.75, 'RMSE = {0:4.2f} mev/atom'.format(loss[-1, 4]*1000), fontsize=13)
 
#plt.text(20,-20,"Train dataset/na")
xlim([ene_min, ene_max])
ylim([ene_min, ene_max])
xlabel('DFT energy (eV/atom)')
ylabel('NEP energy (eV/atom)')
legend(loc="lower left", fontsize=18)
plt.legend(frameon=False, fontsize=18)

subplot(2, 2, 3)
set_fig_properties([gca()])
if test_flag == 1:
    for_min = np.min([np.min(force_train),np.min(force_test)])
    for_max = np.max([np.max(force_train),np.max(force_test)])
else:
    for_min = np.min(force_train)
    for_max = np.max(force_train)
for_min -= (for_max-for_min)*0.1
for_max += (for_max-for_min)*0.1
plot([for_min, for_max], [for_min, for_max], c = "grey", lw = 4)
plot(force_train[:, 3], force_train[:, 0], 'o', c="C2", ms = 6, label="Train dataset (RMSE={0:4.2f} meV/atom)".format(loss[-1, 5]*1000))
plot(force_train[:, 4:6], force_train[:, 1:3], 'o', c="C2", ms = 6)
if test_flag == 1:
    plot(force_test[:, 3], force_test[:, 0], 'o', c="C15", ms = 4,label="Test dataset")
    plot(force_test[:, 4:6], force_test[:, 1:3], 'o', c="C15", ms = 4)
#text(for_min*0.9+for_max*0.1, for_min*0.25+for_max*0.75, 'RMSE = {0:4.2f} mev/A'.format(loss[-1, 5]*1000), fontsize=13)
xlim([for_min, for_max])
ylim([for_min, for_max])
xlabel(r'DFT force (eV/$\rm{\AA}$)')
ylabel(r'NEP force (eV/$\rm{\AA}$)')
legend(loc="upper left")
plt.legend(frameon=False, fontsize=18)

# subplot(2, 2, 4)
# set_fig_properties([gca()])
# if test_flag == 1:
#     ptra = virial_train[:,-1] > -1e-5
#     ptes = virial_test[:,-1] > -1e-5
#     vir_min = np.min([np.min(virial_train[ptra, :]),np.min(virial_test[ptes, :])])
#     vir_max = np.max([np.max(virial_train[ptra, :]),np.max(virial_test[ptes, :])])
# else:
#     ptra = virial_train[:,-1] > -1e-5
#     vir_min = np.min(virial_train[ptra, :])
#     vir_max = np.max(virial_train[ptra, :])
# vir_min -= (vir_max-vir_min)*0.1
# vir_max += (vir_max-vir_min)*0.1
# #vir_min = -0.09
# #vir_max =  0.04
# plot([vir_min, vir_max], [vir_min, vir_max], c = "grey", lw = 1)
# if virial_train.shape[1] == 2:
#     plot(virial_train[ptra, 1], virial_train[ptra, 0], 'o', c="C3", ms = 5, label="Train dataset (RMSE={0:4.2f} mev/atom)".format(loss[-1, 6]*1000))
# elif virial_train.shape[1] == 12:
#     plot(virial_train[ptra, 6], virial_train[ptra, 0], 'o', c="C3", ms = 5, label="Train dataset (RMSE={0:4.2f} mev/atom)".format(loss[-1, 6]*1000))
#     plot(virial_train[ptra, 7:12], virial_train[ptra, 1:6], 'o', c="C3", ms = 5)
# if test_flag == 1:
#     if virial_test.shape[1] == 2:
#         plot(virial_test[ptes, 1], virial_test[ptes, 0], 'o', c="C3", ms = 2, label="Train dataset (RMSE={0:4.2f} mev/atom)".format(loss[-1, 6]*1000))
#     elif virial_test.shape[1] == 12:
#         plot(virial_test[ptes, 6], virial_test[ptes, 0], 'o', c="C8", ms = 2, label="Test dataset (RMSE={0:4.2f} mev/atom)".format(loss[-1, 9]*1000))
#         plot(virial_test[ptes, 7:12], virial_test[ptes, 1:6], 'o', c="C8", ms = 2)
# #text(vir_min*0.9+vir_max*0.1, vir_min*0.25+vir_max*0.75, 'RMSE = {0:4.2f} mev/atom'.format(loss[-1, 6]*1000), fontsize=13)
# xlim([vir_min, vir_max])
# ylim([vir_min, vir_max])
# xlabel('DFT virial (eV/atom)')
# ylabel('NEP virial (eV/atom)')
# legend(loc="upper left")

subplot(2, 2, 4)
set_fig_properties([gca()])
if test_flag == 1:
    ptra = stress_train[:,-1] > -1e-5
    ptes = stress_test[:,-1] > -1e-5
    vir_min = np.min([np.min(stress_train[ptra, :]),np.min(stress_test[ptes, :])])
    vir_max = np.max([np.max(stress_train[ptra, :]),np.max(stress_test[ptes, :])])
else:
    ptra = stress_train[:,-1] > -1e-5
    vir_min = np.min(stress_train[ptra, :])
    vir_max = np.max(stress_train[ptra, :])
vir_min -= (vir_max-vir_min)*0.1
vir_max += (vir_max-vir_min)*0.1
#vir_min = -0.09
#vir_max =  0.04
plot([vir_min, vir_max], [vir_min, vir_max], c = "grey", lw = 4)
if stress_train.shape[1] == 2:
    plot(stress_train[ptra, 1], stress_train[ptra, 0], 'o', c="C3", ms = 6, label="Train dataset (RMSE={0:4.2f} MPa)".format(loss[-1, 6]*1000))
elif stress_train.shape[1] == 12:
    plot(stress_train[ptra, 6], stress_train[ptra, 0], 'o', c="C3", ms = 6, label="Train dataset (RMSE={0:4.2f} MPa)".format(loss[-1, 6]*1000))
    plot(stress_train[ptra, 7:12], stress_train[ptra, 1:6], 'o', c="C3", ms = 5)
if test_flag == 1:
    if stress_test.shape[1] == 2:
        plot(stress_test[ptes, 1], stress_test[ptes, 0], 'o', c="C3", ms = 4, label="Train dataset (RMSE={0:4.2f} MPa)".format(loss[-1, 6]*1000))
    elif stress_test.shape[1] == 12:
        plot(stress_test[ptes, 6], stress_test[ptes, 0], 'o', c="black", ms = 4,label="Test dataset")
        plot(stress_test[ptes, 7:12], stress_test[ptes, 1:6], 'o', c="black", ms = 2)
#text(vir_min*0.9+vir_max*0.1, vir_min*0.25+vir_max*0.75, 'RMSE = {0:4.2f} MPa'.format(loss[-1, 6]*1000), fontsize=13)
xlim([vir_min, vir_max])
ylim([vir_min, vir_max])
xlabel('DFT stress (GPa)')
ylabel('NEP stress (GPa)')
legend(loc="upper left")

plt.legend(frameon=False, fontsize=18)
subplots_adjust(wspace=0.35, hspace=0.3)
savefig("RMSE.png", bbox_inches='tight')
plt.close()

plot_nep("nep_txt.png")
