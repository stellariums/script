import numpy as np
from scipy.integrate import simpson
from ovito.io import import_file
from ovito.modifiers import CoordinationAnalysisModifier
import sys
import matplotlib.pyplot as plt  # 新增可视化模块

def compute_s_ex(r, g, rho, r_cut):
    """计算过剩熵，自动处理 r=0 处的积分奇点"""
    mask = (r > 0) & (r <= r_cut)  # 排除 r=0 点
    r_sub, g_sub = r[mask], g[mask]
    
    # 安全计算被积函数
    with np.errstate(divide='ignore', invalid='ignore'):
        g_safe = np.where(g_sub > 0, g_sub, 1.0)
        integrand = g_safe * np.log(g_safe) - (g_safe - 1)
        integrand = np.where(g_sub > 0, integrand, 0.0)
    
    # 积分计算
    integral = simpson(integrand * r_sub**2, r_sub) * 2 * np.pi
    return -rho * integral

def plot_results(r_cuts, s_ex_values):
    """可视化过剩熵随截断半径的变化"""
    plt.figure(figsize=(10, 6))
    plt.plot(r_cuts, s_ex_values, 'o-', markersize=8)
    plt.xlabel('截断半径 r_cut (Å)', fontsize=12)
    plt.ylabel('过剩熵 S_ex (k_B)', fontsize=12)
    plt.title('过剩熵随截断半径的变化趋势', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 保存图像并显示
    plt.savefig('s_ex_vs_r_cut.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    # 参数配置
    r_cuts = np.linspace(4.0, 15.0, 20)  # 生成8个从8Å到15Å的截断半径[7](@ref)
    s_ex_results = []
    
    # 加载数据
    pipeline = import_file(sys.argv[1])
    modifier = CoordinationAnalysisModifier(
        number_of_bins=500,
        cutoff=np.max(r_cuts) * 1.2,  # 确保包含最大截断半径[1](@ref)
        partial=False
    )
    pipeline.modifiers.append(modifier)
    data = pipeline.compute(0)
    
    # 获取RDF数据
    table = data.tables["coordination-rdf"]
    if "Bin Center" in table.keys():
        r_array = table["Bin Center"].array
    elif "bin_center" in table.keys():
        r_array = table["bin_center"].array
    else:
        r_array = np.linspace(0, modifier.cutoff, modifier.number_of_bins)
    
    g_array = table["g(r)"].array if "g(r)" in table.keys() else table["RDF"].array
    
    # 计算数密度
    rho = data.particles.count / data.cell.volume
    
    # 多截断半径计算循环
    print(f"{'截断半径(Å)':<15}{'过剩熵(k_B)':<15}")
    print("-" * 30)
    
    for r_cut in r_cuts:
        s_ex = compute_s_ex(r_array, g_array, rho, r_cut)
        s_ex_results.append(s_ex)
        print(f"{r_cut:<15.2f}{s_ex:<15.6f}")
    
    # 结果可视化
    plot_results(r_cuts, s_ex_results)