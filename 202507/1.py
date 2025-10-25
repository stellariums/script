import numpy as np
from scipy.integrate import simpson
from ovito.io import import_file
from ovito.modifiers import CoordinationAnalysisModifier
import sys

def compute_s_ex(r, g, rho, r_cut):
    """计算过剩熵，自动处理 r=0 处的积分奇点"""
    mask = (r > 0) & (r <= r_cut)  # 排除 r=0 点（g(0)=0 会导致对数发散）
    r_sub, g_sub = r[mask], g[mask]
    
    # 安全计算被积函数：g*ln(g) - (g-1)
    with np.errstate(divide='ignore', invalid='ignore'):
        g_safe = np.where(g_sub > 0, g_sub, 1.0)
        integrand = g_safe * np.log(g_safe) - (g_safe - 1)
        integrand = np.where(g_sub > 0, integrand, 0.0)
    
    # 积分 ∫[g*ln(g) - (g-1)] * 2πr² dr
    integral = simpson(integrand * r_sub**2, r_sub) * 2 * np.pi
    return -rho * integral

if __name__ == "__main__":
    pipeline = import_file(sys.argv[1])
    modifier = CoordinationAnalysisModifier(
        number_of_bins=500,  # 分箱数（与 g_array 长度一致）
        cutoff=12.0,         # 截断半径（Å）
        partial=False        # 总 RDF
    )
    pipeline.modifiers.append(modifier)
    data = pipeline.compute(0)  # 计算第一帧

    # 获取 RDF 表
    table = data.tables["coordination-rdf"]
    
    # === 关键修复：统一维度 ===
    # 方案1：优先使用 OVITO 生成的距离数组
    if "Bin Center" in table.keys():
        r_array = table["Bin Center"].array
    elif "bin_center" in table.keys():
        r_array = table["bin_center"].array
    else:
        # 方案2：手动生成完整分箱（含 r=0）
        r_array = np.linspace(0, modifier.cutoff, modifier.number_of_bins)
    
    # 读取 RDF 值（兼容不同版本列名）
    g_array = table["g(r)"].array if "g(r)" in table.keys() else table["RDF"].array
    
    # 验证维度一致性
    if len(r_array) != len(g_array):
        raise ValueError(
            f"维度仍不匹配！请检查 OVITO 版本。\n"
            f"r_array 长度: {len(r_array)}, g_array 长度: {len(g_array)}\n"
            f"建议：升级 OVITO 至最新版（pip install ovito --upgrade）"
        )

    # 计算数密度
    rho = data.particles.count / data.cell.volume
    s_ex = compute_s_ex(r_array, g_array, rho, r_cut=10.0)
    print(f"过剩熵 S_ex: {s_ex:.6f} k_B")