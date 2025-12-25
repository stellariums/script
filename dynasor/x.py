"""
x.py
-----

脚本用途：
    - 在当前目录递归查找所有子文件夹（排除 fig、隐藏目录等）
    - 将根目录下的 `cl-ct.py` 复制到每个子文件夹中运行，批量生成谱图
    - 收集各子文件夹输出的 `cl_ct_fit.png`，统一拷贝到根目录的 `fig` 目录

使用方法：
    1. 将本脚本与 `cl-ct.py` 放在同一目录，该目录的子文件夹中包含待分析的数据
    2. 在该目录下执行：
           python x.py
    3. 脚本会逐个子文件夹运行 `cl-ct.py`，并将生成的 `cl_ct_fit.png` 汇总到 `fig` 中

输出结果：
    - fig/ 子目录：保存以子文件夹名称命名的 png 图像
    - 终端进度条与统计信息：显示成功 / 失败的子目录数量及输出路径
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


def unique_target(path: Path) -> Path:
    """If path exists, append _2, _3... to avoid overwrite."""
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    i = 2
    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1

def run_one(folder: Path, src_py: Path, fig_dir: Path) -> tuple:
    try:
        dst_py = folder / src_py.name
        if dst_py.exists():
            print(f"[INFO] 子文件夹已存在旧 cl-ct.py，删除：{dst_py}")
            dst_py.unlink()
        else:
            print(f"[INFO] 子文件夹不存在 cl-ct.py，将复制新文件到：{dst_py}")
        shutil.copy2(src_py, dst_py)
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
        result = subprocess.run(
            [sys.executable, "-X", "utf8", str(dst_py)],
            cwd=str(folder),
            capture_output=True,
            text=True,
            env=env,
        )
        if result.returncode != 0:
            return "fail", folder, result.stdout, result.stderr
        out_png = folder / "cl_ct_fit.png"
        if not out_png.exists():
            return "no_png", folder, "", ""
        target_path = unique_target(fig_dir / f"{folder.name}.png")
        shutil.copy2(out_png, target_path)
        return "ok", folder, "", ""
    except Exception as e:
        return "exception", folder, str(e), ""


def show_progress(done: int, total_count: int) -> None:
    if total_count == 0:
        return
    bar_len = 20
    filled = int(bar_len * done / total_count)
    bar = "#" * filled + "-" * (bar_len - filled)
    print(f"\r[PROGRESS] [{bar}] {done}/{total_count}", end="", flush=True)


def main():
    root = Path(__file__).resolve().parent
    src_py = root / "cl-ct.py"
    fig_dir = root / "fig"

    if not src_py.exists():
        print(f"[ERROR] 找不到源文件：{src_py}")
        sys.exit(1)

    fig_dir.mkdir(exist_ok=True)

    targets = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirpath = Path(dirpath)
        dirnames[:] = [
            d for d in dirnames
            if d not in ("fig", "__pycache__") and not d.startswith(".")
        ]
        if dirpath == root:
            continue
        if fig_dir == dirpath or fig_dir in dirpath.parents:
            continue
        targets.append(dirpath)

    if not targets:
        print("[INFO] 没有找到需要处理的子文件夹。")
        return

    total = len(targets)
    print(f"[INFO] 找到 {total} 个子文件夹需要处理（已跳过 fig）。")

    ok, fail = 0, 0
    done = 0
    with ProcessPoolExecutor() as executor:
        future_to_folder = {
            executor.submit(run_one, folder, src_py, fig_dir): folder
            for folder in targets
        }
        for future in as_completed(future_to_folder):
            status, folder, stdout_text, stderr_text = future.result()
            done += 1
            show_progress(done, total)
            if status == "ok":
                ok += 1
            elif status == "no_png":
                ok += 1
                print(f"\n[WARN] {folder} 没找到输出图片 cl_ct_fit.png，跳过收集。")
            elif status in ("fail", "exception"):
                fail += 1
                print(f"\n[ERROR] {folder} 运行失败：")
                if stdout_text:
                    print("---- stdout ----")
                    print(stdout_text)
                if stderr_text:
                    print("---- stderr ----")
                    print(stderr_text)

    print()
    print(f"\n[DONE] 成功处理：{ok}，失败：{fail}")
    print(f"[INFO] 输出目录：{fig_dir}")

if __name__ == "__main__":
    main()
