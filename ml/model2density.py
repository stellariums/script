import os
import argparse
import logging
import re
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    # 尝试导入 ASE，用于解析 XYZ 文件并获取模拟盒信息
    from ase.io import read, iread
except ImportError:  # pragma: no cover - 环境中可能未安装 ase
    read = None
    iread = None


"""
model2density.py
================

本脚本用于递归扫描指定根目录下的所有子目录，自动查找名为 `model.xyz` 的结构文件，
并在假定所有原子均为碳原子的前提下，计算其密度 (g/cm^3)，然后将结果追加到
`kappa.csv` 文件中，同时保留原有内容并避免重复记录。

功能概述：
1. 密度计算流程（改为使用 ASE）：
   - 使用 ASE 解析 XYZ 文件
   - 统计原子数量
   - 获取模拟盒体积
   - 利用 ρ = (N * m_C / N_A) / V 并考虑 Å^3 到 cm^3 的单位转换
2. 参考 extract_kappa.py 中的目录遍历方式：
   - 使用 os.walk 递归扫描根目录
   - 自动识别所有名为 model.xyz 的文件
   - 跳过非目标文件
3. 对每个 model.xyz：
   - 计算密度值
   - 记录文件路径
   - 提取 Folder Name / Subfile Name 等元数据
4. 将结果追加写入 kappa.csv：
   - 保留原有 CSV 内容
   - 若文件中尚不存在密度相关列则自动添加
   - 确保脚本多次运行不会产生重复记录（基于文件完整路径去重）

技术要求实现说明：
- 使用 Python 3.6+ 语法
- 对文件读取、ASE 解析、CSV 读写等关键步骤添加 try/except 错误处理
- 使用 logging 模块记录运行信息与错误信息
- 通过在写入前检查既有 CSV 中是否已包含相同文件路径，保证脚本可重复执行

使用说明 (Usage):
---------------
1. 环境依赖:
   确保已安装 Python 3.6+ 以及以下库:
   - numpy
   - pandas
   - ase (必须安装，否则无法计算)

   安装命令示例:
   pip install numpy pandas ase

2. 运行命令:
   在命令行中运行脚本，并指定包含 model.xyz 文件的根目录路径。
   
   格式:
   python model2density.py <根目录路径>

   示例:
   python model2density.py C:\\Users\\USTC\\Desktop\\monolayer\\ML

3. 输出结果:
   - 结果将追加到脚本同级目录下的 `kappa.csv` 文件中。
   - 如果 `kappa.csv` 不存在，将自动创建。
   - 新增/更新的列包括:
     - Folder Name: 上级目录名
     - Subfile Name: 文件所在目录名
     - File Path: model.xyz 的绝对路径
     - Density (g/cm3): 计算得到的密度值
   - 运行日志保存在同级目录下的 `model2density.log`。

注意事项:
- 脚本会自动跳过已经记录在 CSV 中的文件（基于文件路径去重）。
- 如果需要重新计算某些文件，请先在 CSV 中删除对应的行，或者删除整个 CSV 文件。
"""


def setup_logger(log_path: str) -> logging.Logger:
    """
    初始化日志记录器。

    日志同时输出到终端和指定的日志文件，方便排查问题。
    """
    logger = logging.getLogger("model2density")
    logger.setLevel(logging.INFO)

    # 避免重复添加 handler（例如脚本被多次导入时）
    if logger.handlers:
        return logger

    # 创建日志目录（若不存在）
    log_dir = os.path.dirname(log_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 文件日志
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 控制台日志
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def compute_density_from_xyz(file_path: str, logger: logging.Logger) -> Optional[float]:
    """
    使用 ASE 解析单个 XYZ 文件并计算密度，假定所有原子为碳。

    算法流程：
    1. 使用 ase.io.read 读取 XYZ 结构文件
    2. 获取模拟盒体积（单位：Å^3）
    3. 获取原子数 N
    4. 使用碳原子摩尔质量 m_C = 12.01 g/mol，阿伏伽德罗常数 N_A = 6.022e23
    5. 单个原子质量：m_atom = m_C / N_A (g)
    6. 总质量：m_total = N * m_atom (g)
    7. 体积换算：1 Å^3 = 1e-24 cm^3，因此 V_cm3 = V_angs^3 * 1e-24
    8. 密度：rho = m_total / V_cm3，整理后为
       rho = (N * m_atom * 1e24) / V_angs^3 (g/cm^3)

    返回：
        float : 计算得到的密度（单位：g/cm^3）
        None  : 当 ASE 不可用或计算失败时返回 None
    """
    try:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                first_line = f.readline()
                second_line = f.readline()
        except Exception:
            first_line = ""
            second_line = ""

        num_atoms = None
        volume = None

        if first_line:
            try:
                num_atoms = int(first_line.strip().split()[0])
            except Exception:
                num_atoms = None

        if second_line:
            m = re.search(r'Lattice="([^"]+)"', second_line)
            if m:
                parts = m.group(1).split()
                if len(parts) == 9:
                    try:
                        cell = np.array([float(x) for x in parts], dtype=float).reshape((3, 3))
                        volume = float(abs(np.linalg.det(cell)))
                    except Exception:
                        volume = None

        if num_atoms is None or volume is None or volume <= 1e-12:
            if iread is None and read is None:
                logger.error("未安装 ase 库，且无法从文件头解析晶胞信息，无法计算密度。")
                return None

            try:
                if iread is not None:
                    atoms = next(iread(file_path, format="extxyz"))
                else:
                    atoms = read(file_path, format="extxyz", index=0)
            except Exception:
                try:
                    if iread is not None:
                        atoms = next(iread(file_path))
                    else:
                        atoms = read(file_path, index=0)
                except Exception as exc:
                    logger.error("计算文件 %s 密度时发生错误：%s", file_path, exc)
                    return None

            num_atoms = len(atoms)
            try:
                volume = atoms.get_volume()
            except Exception:
                volume = 0.0

        if num_atoms <= 0:
            logger.error("文件 %s 中原子数为零，无法计算密度。", file_path)
            return None
        if volume <= 1e-12: # 使用一个小正数判断，防止浮点误差
            logger.error("文件 %s 中模拟盒体积非正值或未定义(Volume=%.2e)，无法计算密度。请确保 XYZ 文件包含 Lattice 信息。", file_path, volume)
            return None

        # 单个碳原子的质量（单位：g）
        actual_mass_per_atom = 12.01 / 6.022e23
        # 按照公式，将 Å^3 转换为 cm^3
        density = (num_atoms * actual_mass_per_atom * 1e24) / volume

        logger.info("成功计算密度：%s -> %.6f g/cm^3", file_path, density)
        return float(density)

    except Exception as exc:
        logger.error("计算文件 %s 密度时发生错误：%s", file_path, exc)
        return None


def find_model_xyz_files(root_dir: str, logger: logging.Logger) -> List[str]:
    """
    递归扫描根目录，查找所有名为 model.xyz 的文件。

    参数：
        root_dir : 根目录路径（绝对或相对路径均可）

    返回：
        包含所有找到的 model.xyz 绝对路径的列表。
    """
    model_files: List[str] = []
    root_dir = os.path.abspath(root_dir)

    if not os.path.isdir(root_dir):
        logger.error("提供的根目录不存在：%s", root_dir)
        return model_files

    logger.info("开始扫描根目录：%s", root_dir)

    for dirpath, _, filenames in os.walk(root_dir):
        # 仅当当前目录中存在目标文件名时才记录
        if "model.xyz" in filenames:
            full_path = os.path.join(dirpath, "model.xyz")
            model_files.append(full_path)
            logger.info("发现目标文件：%s", full_path)

    logger.info("扫描完成，共发现 %d 个 model.xyz 文件。", len(model_files))
    return model_files


def extract_folder_and_subfile(file_path: str) -> Tuple[str, str]:
    """
    根据文件路径提取 Folder Name 和 Subfile Name。

    假定目录结构与 extract_kappa.py 一致：
        .../Folder Name/Subfile Name/model.xyz

    若目录层级不足，则使用空字符串占位。
    """
    # 文件所在目录即 Subfile Name
    parent_dir = os.path.dirname(file_path)
    subfile_name = os.path.basename(parent_dir) if parent_dir else ""

    # 上一层目录作为 Folder Name
    folder_dir = os.path.dirname(parent_dir) if parent_dir else ""
    folder_name = os.path.basename(folder_dir) if folder_dir else ""

    return folder_name, subfile_name


def load_existing_kappa(csv_path: str, logger: logging.Logger) -> pd.DataFrame:
    """
    读取现有的 kappa.csv 文件。

    若文件不存在，则返回一个空的 DataFrame。
    """
    if not os.path.exists(csv_path):
        logger.info("未找到现有 kappa.csv，将创建新的文件：%s", csv_path)
        return pd.DataFrame()

    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
        logger.info("成功读取现有 CSV 文件：%s，当前记录数：%d", csv_path, len(df))
        return df
    except Exception as exc:
        logger.error("读取 CSV 文件 %s 时发生错误：%s", csv_path, exc)
        # 出于安全考虑，出现读取错误时不覆盖原文件，而是返回空表
        return pd.DataFrame()


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    确保 DataFrame 中包含脚本所需的关键列。

    当前需要的列包括：
    - Folder Name
    - Subfile Name
    - File Path
    - Density (g/cm3)

    对于缺失的列，将自动补充并填充为 NaN。
    """
    required_columns = ["Folder Name", "Subfile Name", "File Path", "Density (g/cm3)"]
    for col in required_columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def append_density_records(
    root_dir: str, csv_path: str, logger: logging.Logger
) -> None:
    """
    主逻辑：扫描 model.xyz，计算密度，并将结果追加到 kappa.csv 中。

    关键点：
    - 保留原有 CSV 内容（先读后写）
    - 优先回填到既有行（按 File Path 或 Folder/Subfile 匹配）
    - 保证列名风格与 extract_kappa.py 生成的文件一致
    """
    df_existing = load_existing_kappa(csv_path, logger)
    df_existing = ensure_columns(df_existing)

    model_files = find_model_xyz_files(root_dir, logger)

    if not model_files:
        logger.info("未找到任何 model.xyz 文件，退出。")
        return

    new_rows: List[Dict[str, object]] = []
    updated_count = 0

    folder_series = df_existing["Folder Name"].fillna("").astype(str)
    subfile_series = df_existing["Subfile Name"].fillna("").astype(str)
    path_series = df_existing["File Path"].fillna("").astype(str)

    folder_subfile_to_indices: Dict[Tuple[str, str], List[int]] = {}
    for idx, (folder_val, subfile_val) in enumerate(zip(folder_series, subfile_series)):
        key = (folder_val, subfile_val)
        folder_subfile_to_indices.setdefault(key, []).append(idx)

    path_to_indices: Dict[str, List[int]] = {}
    for idx, path_val in enumerate(path_series):
        if not path_val:
            continue
        abs_path_val = os.path.abspath(path_val)
        path_to_indices.setdefault(abs_path_val, []).append(idx)

    for file_path in model_files:
        abs_path = os.path.abspath(file_path)

        folder_name, subfile_name = extract_folder_and_subfile(abs_path)
        indices_by_path = path_to_indices.get(abs_path, [])
        indices_by_folder = folder_subfile_to_indices.get((folder_name, subfile_name), [])
        indices = sorted(set(indices_by_path) | set(indices_by_folder))

        if indices:
            density_col = df_existing.loc[indices, "Density (g/cm3)"]
            need_density = bool(density_col.isna().any())

            if need_density:
                density = compute_density_from_xyz(abs_path, logger)
                if density is None:
                    continue

                for idx in indices:
                    if pd.isna(df_existing.at[idx, "File Path"]) or str(df_existing.at[idx, "File Path"]) == "":
                        df_existing.at[idx, "File Path"] = abs_path
                    if pd.isna(df_existing.at[idx, "Density (g/cm3)"]):
                        df_existing.at[idx, "Density (g/cm3)"] = density
                        updated_count += 1
            else:
                for idx in indices:
                    if pd.isna(df_existing.at[idx, "File Path"]) or str(df_existing.at[idx, "File Path"]) == "":
                        df_existing.at[idx, "File Path"] = abs_path

            continue

        density = compute_density_from_xyz(abs_path, logger)
        if density is None:
            continue

        row: Dict[str, object] = {
            "Folder Name": folder_name,
            "Subfile Name": subfile_name,
            "File Path": abs_path,
            "Density (g/cm3)": density,
        }
        new_rows.append(row)

    if updated_count == 0 and not new_rows:
        logger.info("没有新的密度记录需要写入，kappa.csv 保持不变。")
        return

    if new_rows:
        df_new = pd.DataFrame(new_rows)
        df_all = pd.concat([df_existing, df_new], ignore_index=True, sort=False)
    else:
        df_all = df_existing.copy()

    preferred_order = ["Folder Name", "Subfile Name", "File Path", "Density (g/cm3)"]
    remaining_cols = [c for c in df_all.columns if c not in preferred_order]
    df_all = df_all[preferred_order + remaining_cols]

    try:
        df_all.to_csv(csv_path, index=False, encoding="utf-8")
        logger.info(
            "成功写入密度数据：更新 %d 行，新增 %d 行：%s（总记录数：%d）",
            updated_count,
            len(new_rows),
            csv_path,
            len(df_all),
        )
    except Exception as exc:
        logger.error("写入 CSV 文件 %s 时发生错误：%s", csv_path, exc)


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    用法示例：
        python model2density.py C:\\Users\\USTC\\Desktop\\monolayer\\ML
        python model2density.py
    """
    parser = argparse.ArgumentParser(
        description=(
            "递归扫描根目录下的 model.xyz 文件，计算密度并追加写入 kappa.csv。"
        )
    )
    parser.add_argument(
        "root_dir",
        type=str,
        nargs="?",
        default=".",
        help="包含多级子文件夹的根目录路径，用于搜索 model.xyz。默认为当前目录。",
    )
    return parser.parse_args()


def main() -> None:
    """
    脚本入口函数。

    步骤：
    1. 解析命令行参数，获取根目录路径
    2. 根据当前脚本所在目录确定 kappa.csv 的绝对路径
    3. 初始化日志系统
    4. 调用 append_density_records 执行核心逻辑
    """
    args = parse_args()

    # 当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 输出 CSV 文件路径，按照需求固定为当前脚本目录下的 kappa.csv
    csv_path = os.path.join(script_dir, "kappa.csv")
    # 日志文件路径
    log_path = os.path.join(script_dir, "model2density.log")

    logger = setup_logger(log_path)
    logger.info("启动 model2density 脚本。根目录：%s，CSV 路径：%s", args.root_dir, csv_path)

    append_density_records(args.root_dir, csv_path, logger)

    logger.info("脚本执行结束。")


if __name__ == "__main__":
    main()
