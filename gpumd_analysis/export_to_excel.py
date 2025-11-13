import os
import sys
import csv
from typing import Dict, List, Tuple

def read_results(results_path: str) -> List[Tuple[str, float, float, float, float]]:
    rows: List[Tuple[str, float, float, float, float]] = []
    with open(results_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        for r in reader:
            file_path = r[0]
            density = float(r[1])
            sp3_pct = float(r[3])
            ratio = float(r[4])
            crystallinity = float(r[5])
            rows.append((file_path, density, sp3_pct, ratio, crystallinity))
    return rows

def read_pressure(pressure_path: str) -> Dict[str, float]:
    data: Dict[str, float] = {}
    with open(pressure_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        for r in reader:
            p_path = r[0]
            val = float(r[1])
            data[p_path] = val
    return data

def canonical_dir_from_results(path: str) -> str:
    p = path.replace("./", ".\\").replace("/", "\\")
    parts = p.rsplit("\\", 1)
    return parts[0]

def canonical_dir_from_system(path: str) -> str:
    p = path.replace("/", "\\")
    parts = p.rsplit("\\", 1)
    return parts[0]

def build_pressure_by_dir(pressure_map: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in pressure_map.items():
        d = canonical_dir_from_system(k)
        out[d] = v
    return out

def build_results_by_dir(rows: List[Tuple[str, float, float, float, float]]) -> Dict[str, Tuple[float, float, float, float, str]]:
    out: Dict[str, Tuple[float, float, float, float, str]] = {}
    for file_path, density, sp3_pct, ratio, crystallinity in rows:
        d = canonical_dir_from_results(file_path)
        out[d] = (density, sp3_pct, ratio, crystallinity, file_path)
    return out

def write_excel(output_path: str, rows: List[Tuple[str, float, float, float, float, float]]):
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append(["路径", "密度", "温度", "sp3", "sp3/sp2的比例", "结晶度", "压强"])
    for r in rows:
        ws.append(list(r))
    wb.save(output_path)

def main():
    base_dir = os.getcwd()
    results_path = os.path.join(base_dir, "results.txt")
    pressure_path = os.path.join(base_dir, "system_pressure_summary.txt")
    output_path = os.path.join(base_dir, "output.xlsx")

    if not os.path.exists(results_path):
        print(f"缺少文件: {results_path}")
        sys.exit(1)
    if not os.path.exists(pressure_path):
        print(f"缺少文件: {pressure_path}")
        sys.exit(1)

    results_rows = read_results(results_path)
    pressure_map = read_pressure(pressure_path)
    pressure_by_dir = build_pressure_by_dir(pressure_map)
    results_by_dir = build_results_by_dir(results_rows)

    merged: List[Tuple[str, float, float, float, float, float]] = []
    missing_in_results: List[str] = []
    for system_path, pressure_value in pressure_map.items():
        d = canonical_dir_from_system(system_path)
        if d not in results_by_dir:
            missing_in_results.append(d)
            continue
        density, sp3_pct, ratio, crystallinity, _ = results_by_dir[d]
        merged.append((system_path, density, None, sp3_pct, ratio, crystallinity, pressure_value))

    if missing_in_results:
        print("以下路径在结果文件中缺失:")
        for m in missing_in_results:
            print(m)
        sys.exit(2)

    write_excel(output_path, merged)
    print(f"已生成: {output_path}")

if __name__ == "__main__":
    main()