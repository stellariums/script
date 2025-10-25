from ase.io import read, write

# 读取 POSCAR 文件
atoms = read('N2.poscar', format='vasp')

# 导出为 CIF 文件
write('N2.cif', atoms)

print("✅ 已成功将 N2.poscar 转换为 N2.cif")
