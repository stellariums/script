import matplotlib.pyplot as plt

# 由 phonopy-bandplot 导出的数据文件
filename = "phonon.dat"

# 对应 band.conf 里的 BAND_LABELS
xticks = [0.00000000, 0.10896130, 0.14748460, 0.26305540, 0.35741800]
xlabels = [r"$\Gamma$", "X", "U", r"$\Gamma$", "L"]

bands = []
current_band = []

with open(filename, "r") as f:
    for line in f:
        line = line.strip()

        # 空行通常表示一条声子支结束
        if not line:
            if current_band:
                bands.append(current_band)
                current_band = []
            continue

        # 跳过注释行
        if line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) >= 2:
            x = float(parts[0])
            y = float(parts[1])
            current_band.append((x, y))

if current_band:
    bands.append(current_band)

plt.rcParams["font.family"] = "Arial"
plt.figure(figsize=(6, 4), dpi=300)

# 画每一条声子支
for band in bands:
    x = [p[0] for p in band]
    y = [p[1] for p in band]
    plt.plot(x, y, linewidth=1.1)

# 画高对称点竖线
for x in xticks:
    plt.axvline(x=x, linestyle="--", linewidth=0.7)

# 画 y=0 参考线，方便判断虚频
plt.axhline(y=0, linestyle="-", linewidth=0.8)

plt.xticks(xticks, xlabels)
plt.ylabel("Frequency (THz)")
plt.xlabel("Wave vector")
plt.xlim(xticks[0], xticks[-1])

plt.tight_layout()
plt.savefig("UN_phonon_band.png", dpi=300)
plt.savefig("UN_phonon_band.pdf")
plt.show()