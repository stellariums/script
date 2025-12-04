import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import concurrent.futures
import time
import traceback

"""
批量热力学数据绘图脚本 (Batch Thermo Plotter)

功能描述:
    该脚本用于递归遍历当前目录及其子目录，查找并处理分子动力学模拟生成的 `thermo.out` 文件。
    它会读取每个文件中的热力学数据（如温度、压力、能量、晶格参数等），生成可视化的趋势图，
    并将图表保存为 `thermo_plot.png`。

主要特性:
    1. 递归遍历: 自动扫描当前目录下的所有子文件夹。
    2. 智能识别: 自动读取 `run.in` 文件以确定时间步长和输出间隔。
    3. 并行处理: 使用多进程技术加速批量文件处理。
    4. 错误处理: 自动跳过空文件或损坏文件，并报告错误信息。
    5. 绘图输出: 生成包含温度、压力、能量、晶格常数、体积等信息的组合图表。

使用方法:
    在包含模拟数据的根目录下运行:
    python batch_plot_thermo.py

依赖库:
    - numpy
    - matplotlib

作者: Trae AI Assistant
日期: 2025-12-02
"""

def calculate_angle(x, y):
    """
    计算两个向量数组之间的夹角（度）。
    """
    dot_product = np.einsum('ij,ij->i', x, y)

    norm_x = np.linalg.norm(x, axis=1)
    norm_y = np.linalg.norm(y, axis=1)
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        angle_radians = np.arccos(np.clip(dot_product / (norm_x * norm_y), -1.0, 1.0))
    return np.degrees(angle_radians)

def calculate_volume(a, b, c):
    """
    计算由三个向量定义的平行六面体体积。
    """
    volume = np.einsum('ij,ij->i', a, np.cross(b, c))
    return np.abs(volume)

def get_dump_interval(folder_path):
    """
    从 run.in 文件中读取 time_step 和 dump_thermo 参数，计算输出时间间隔（ps）。
    如果读取失败，则使用默认值。
    """
    timestep = 1.0  # Default
    dump_interval = 10  # Default 
    
    run_in_path = os.path.join(folder_path, 'run.in')
    if os.path.exists(run_in_path):
        try:
            with open(run_in_path, 'r') as file:
                for line in file:
                    # Read timestep value
                    if "time_step" in line:
                        parts = line.split()
                        if len(parts) > 1:
                            try:
                                timestep = float(parts[1])  # timestep in fs
                            except ValueError:
                                pass
                    # Read dump_thermo interval
                    elif "dump_thermo" in line:
                        parts = line.split()
                        if len(parts) > 1:
                            try:
                                dump_interval = int(parts[1])  # number of timesteps between dumps
                                break
                            except ValueError:
                                pass
        except Exception:
            pass # Fallback to defaults if read fails
    
    # Calculate total time interval per dump in ps
    total_interval_ps = timestep * dump_interval / 1000.0
    
    return total_interval_ps

def process_single_folder(folder_path):
    """
    处理单个文件夹：读取数据并绘图。
    
    Args:
        folder_path (str): 文件夹路径
        
    Returns:
        tuple: (success (bool), message (str))
    """
    thermo_path = os.path.join(folder_path, 'thermo.out')
    output_plot_path = os.path.join(folder_path, 'thermo_plot.png')
    
    try:
        # Check if file exists and is not empty
        if not os.path.exists(thermo_path) or os.path.getsize(thermo_path) == 0:
            return False, "文件不存在或为空 (File missing or empty)"

        # Load data
        try:
            data = np.loadtxt(thermo_path)
        except Exception as e:
            return False, f"数据加载失败 (Failed to load data): {str(e)}"

        if data.size == 0:
            return False, "数据为空 (Data is empty)"
            
        # Handle case where loadtxt returns 1D array for single line
        if data.ndim == 1:
            data = data.reshape(1, -1)

        dump_interval_ps = get_dump_interval(folder_path)
        time_axis = np.arange(0, len(data) * dump_interval_ps, dump_interval_ps)
        
        # Adjust time axis length if it doesn't match data (e.g. due to rounding or implementation details)
        if len(time_axis) != len(data):
            time_axis = np.arange(len(data)) * dump_interval_ps

        # Read columns
        # Assuming standard columns based on plt_nep_thermo.py
        # 0: Temp, 1: KE, 2: PE, 3: Px, 4: Py, 5: Pz
        temperature = data[:, 0]
        kinetic_energy = data[:, 1]
        potential_energy = data[:, 2]
        pressure_x = data[:, 3]
        pressure_y = data[:, 4]
        pressure_z = data[:, 5]

        num_columns = data.shape[1]
        
        box_length_x = None
        box_length_y = None
        box_length_z = None
        volume = None
        
        box_angle_alpha = None
        box_angle_beta = None
        box_angle_gamma = None

        if num_columns == 12:
            box_length_x = data[:, 9]
            box_length_y = data[:, 10]
            box_length_z = data[:, 11]
            volume = box_length_x * box_length_y * box_length_z
        elif num_columns == 18:
            ax, ay, az = data[:, 9], data[:, 10], data[:, 11]
            bx, by, bz = data[:, 12], data[:, 13], data[:, 14]
            cx, cy, cz = data[:, 15], data[:, 16], data[:, 17]

            a_vectors = np.column_stack((ax, ay, az))
            b_vectors = np.column_stack((bx, by, bz))
            c_vectors = np.column_stack((cx, cy, cz))

            box_length_x = np.sqrt(ax**2 + ay**2 + az**2)
            box_length_y = np.sqrt(bx**2 + by**2 + bz**2)
            box_length_z = np.sqrt(cx**2 + cy**2 + cz**2)

            box_angle_alpha = calculate_angle(b_vectors, c_vectors)
            box_angle_beta = calculate_angle(c_vectors, a_vectors)
            box_angle_gamma = calculate_angle(a_vectors, b_vectors)

            volume = calculate_volume(a_vectors, b_vectors, c_vectors)
        else:
            return False, f"不支持的列数 (Unsupported column count): {num_columns}"

        # Plotting
        # Use a non-interactive backend to be safe in threads/processes
        plt.switch_backend('Agg') 
        
        fig, axs = plt.subplots(2, 3, figsize=(12, 6), dpi=100)

        # Temperature
        axs[0, 0].plot(time_axis, temperature)
        axs[0, 0].set_title('Temperature')
        axs[0, 0].set_xlabel('Time (ps)')
        axs[0, 0].set_ylabel('Temperature (K)')

        # Pressure
        axs[0, 1].plot(time_axis, pressure_x, label='Px')
        axs[0, 1].plot(time_axis, pressure_y, label='Py')
        axs[0, 1].plot(time_axis, pressure_z, label='Pz')
        axs[0, 1].set_title('Pressure')
        axs[0, 1].set_xlabel('Time (ps)')
        axs[0, 1].set_ylabel('Pressure (GPa)')
        axs[0, 1].legend()

        # Potential Energy and Kinetic Energy
        pe_min, pe_max = np.min(potential_energy), np.max(potential_energy)
        pe_range = pe_max - pe_min if pe_max != pe_min else 1.0
        pe_ylim_lower = pe_min - 0.6 * pe_range
        pe_ylim_upper = pe_max + 0.05 * pe_range
        
        axs[0, 2].set_title(r'$P_E$ vs $K_E$')
        axs[0, 2].set_xlabel('Time (ps)')
        axs[0, 2].set_ylabel(r'Potential Energy (eV)', color='tab:orange')
        axs[0, 2].plot(time_axis, potential_energy, color='tab:orange')
        axs[0, 2].set_ylim(pe_ylim_lower, pe_ylim_upper)
        axs[0, 2].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        axs[0, 2].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        axs[0, 2].tick_params(axis='y', labelcolor='tab:orange')

        ke_min, ke_max = np.min(kinetic_energy), np.max(kinetic_energy)
        ke_range = ke_max - ke_min if ke_max != ke_min else 1.0
        ke_ylim_lower = ke_min - 0.05 * ke_range
        ke_ylim_upper = ke_max + 0.6 * ke_range
        
        axs_kinetic = axs[0, 2].twinx()
        axs_kinetic.set_ylabel('Kinetic Energy (eV)', color='tab:green')
        axs_kinetic.plot(time_axis, kinetic_energy, color='tab:green')
        axs_kinetic.set_ylim(ke_ylim_lower, ke_ylim_upper)
        axs_kinetic.tick_params(axis='y', labelcolor='tab:green')
        axs_kinetic.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        axs_kinetic.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

        # Lattice
        axs[1, 0].plot(time_axis, box_length_x, label='Lx')
        axs[1, 0].plot(time_axis, box_length_y, label='Ly')
        axs[1, 0].plot(time_axis, box_length_z, label='Lz')
        axs[1, 0].set_title('Lattice Parameters')
        axs[1, 0].set_xlabel('Time (ps)')
        axs[1, 0].set_ylabel(r'Lattice Parameters ($\AA$)')
        axs[1, 0].legend()

        # Volume
        axs[1, 1].plot(time_axis, volume, label='Volume', color='tab:purple')
        axs[1, 1].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        axs[1, 1].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        axs[1, 1].set_title('Volume')
        axs[1, 1].set_xlabel('Time (ps)')
        axs[1, 1].set_ylabel(r'Volume ($\AA^3$)')
        axs[1, 1].legend()

        # Angles
        if num_columns == 18:
            axs[1, 2].plot(time_axis, box_angle_alpha, label=r'$\alpha$')
            axs[1, 2].plot(time_axis, box_angle_beta, label=r'$\beta$')
            axs[1, 2].plot(time_axis, box_angle_gamma, label=r'$\gamma$')
            axs[1, 2].set_title('Interaxial Angles')
            axs[1, 2].set_xlabel('Time (ps)')
            axs[1, 2].set_ylabel(r'Interaxial Angles ($\degree$)')
            axs[1, 2].legend()
        else:
             axs[1, 2].axis('off') # Hide if not applicable

        plt.tight_layout()
        plt.savefig(output_plot_path, dpi=150)
        plt.close(fig)
        
        return True, "成功 (Success)"

    except Exception as e:
        # traceback.print_exc()
        return False, f"错误 (Error): {str(e)}"

def main():
    """
    主函数：扫描目录并管理并行处理任务。
    """
    root_dir = os.getcwd()
    target_folders = []

    print(f"正在扫描目录 (Scanning directories starting from): {root_dir}")
    
    # 1. Traverse directories
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if 'thermo.out' in filenames:
            target_folders.append(dirpath)

    total_folders = len(target_folders)
    print(f"发现 {total_folders} 个包含 'thermo.out' 的文件夹 (Found {total_folders} folders).")

    if total_folders == 0:
        print("未找到需要处理的文件夹 (No folders to process).")
        return

    print("开始批量处理 (Starting batch processing)...")
    start_time = time.time()
    
    success_count = 0
    fail_count = 0
    failed_folders = []

    # 2. Parallel processing
    # Using ProcessPoolExecutor for CPU/Plotting tasks
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit all tasks
        future_to_folder = {executor.submit(process_single_folder, folder): folder for folder in target_folders}
        
        for i, future in enumerate(concurrent.futures.as_completed(future_to_folder)):
            folder = future_to_folder[future]
            try:
                success, message = future.result()
                if success:
                    success_count += 1
                    # print(f"[{i+1}/{total_folders}] Processed: {os.path.basename(folder)}")
                else:
                    fail_count += 1
                    failed_folders.append((folder, message))
                    print(f"[{i+1}/{total_folders}] 失败 (Failed): {os.path.basename(folder)} - {message}")
            except Exception as exc:
                fail_count += 1
                failed_folders.append((folder, str(exc)))
                print(f"[{i+1}/{total_folders}] 异常 (Exception in) {os.path.basename(folder)}: {exc}")

    end_time = time.time()
    duration = end_time - start_time

    print("\n" + "="*40)
    print(f"处理完成，耗时 (Processing Complete in) {duration:.2f} 秒 (seconds)")
    print(f"总文件夹数 (Total folders): {total_folders}")
    print(f"成功 (Successful): {success_count}")
    print(f"失败 (Failed): {fail_count}")
    
    if failed_folders:
        print("\n失败文件夹列表 (Failed Folders):")
        for folder, reason in failed_folders:
            print(f"- {folder}: {reason}")
    print("="*40)

if __name__ == "__main__":
    # Ensure safe multiprocessing on Windows
    import multiprocessing
    multiprocessing.freeze_support()
    main()
