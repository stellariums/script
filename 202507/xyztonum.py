import numpy as np
import sys
from ovito.io import import_file
from ovito.modifiers import CreateBondsModifier, CoordinationAnalysisModifier, IdentifyDiamondModifier

def calculate_system_properties(file_path):
    pipeline = import_file(file_path)
    pipeline.modifiers.append(CreateBondsModifier(cutoff=1.85))
    data = pipeline.compute()
    
    # 基础属性计算
    volume = data.cell.volume
    num_atoms = data.particles.count
    relative_mass_per_atom = 12.01
    total_relative_mass = num_atoms * relative_mass_per_atom
    actual_mass_per_atom = 12.01 / 6.022e23
    density = (num_atoms * actual_mass_per_atom * 1e24) / volume

    # 配位数分析
    pipeline.modifiers.append(CoordinationAnalysisModifier(cutoff=1.85))
    data = pipeline.compute()
    coord_numbers = data.particles['Coordination']
    sp2_percent = np.sum(coord_numbers == 3) / num_atoms * 100
    sp3_percent = np.sum(coord_numbers == 4) / num_atoms * 100

    # ==== 结晶度分析修正 ====
    pipeline.modifiers.append(IdentifyDiamondModifier())
    data = pipeline.compute()
    structure_types = data.particles['Structure Type']
    
    # 统计晶体原子（类型1-6）
    crystal_atoms = np.sum((structure_types >= 1) & (structure_types <= 6))
    type_counts = {t: np.sum(structure_types == t) for t in range(7)}
    crystallinity = crystal_atoms / num_atoms * 100

    # 调试输出
    type_names = {
        0: "非晶/Other",
        1: "CUBIC_DIAMOND",
        2: "CUBIC_DIAMOND_FIRST_NEIGHBOR",
        3: "CUBIC_DIAMOND_SECOND_NEIGHBOR",
        4: "HEX_DIAMOND",
        5: "HEX_DIAMOND_FIRST_NEIGHBOR",
        6: "HEX_DIAMOND_SECOND_NEIGHBOR"
    }
    print("\n[DEBUG] 金刚石结构分布 (IdentifyDiamondModifier):")
    for type_id in range(7):
        count = type_counts[type_id]
        name = type_names.get(type_id, f"未知类型({type_id})")
        print(f"  类型{type_id} ({name}): {count}原子 ({count/num_atoms*100:.4f}%)")

    return {
        "Total Relative Mass": total_relative_mass,
        "Volume (Å³)": volume,
        "Density (g/cm³)": density,
        "sp² Percentage (%)": sp2_percent,
        "sp³ Percentage (%)": sp3_percent,
        "Crystallinity (%)": crystallinity
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ovs.py <file.xyz>")
        sys.exit(1)
    results = calculate_system_properties(sys.argv[1])
    print("\nSystem Properties Analysis:")
    print("-" * 40)
    for k, v in results.items():
        print(f"{k}: {v:.4f}" if "Percentage" in k or "Crystallinity" in k else f"{k}: {v:.2f}")