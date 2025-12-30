#!/bin/bash

# =============================================================================
# 脚本名称: run_simulations.sh
# 功能描述: 
#   1. 遍历当前目录下所有 .xyz 文件
#   2. 为每个 .xyz 文件创建同名目录
#   3. 将 .xyz 文件复制到目录中并重命名为 model.xyz
#   4. 复制 run.in 和 nep.txt 到目录中
#   5. 进入每个目录执行 gpumd
# =============================================================================

# 启用错误检测，如果任何命令失败则脚本可能会继续（根据逻辑），但这里我们手动处理错误
# set -e 

# 获取当前工作目录的绝对路径
WORK_DIR=$(pwd)
echo "[INFO] Current working directory: $WORK_DIR"

# 定义源文件路径 (使用绝对路径)
RUN_IN="$WORK_DIR/run.in"
NEP_TXT="$WORK_DIR/nep.txt"

# -----------------------------------------------------------------------------
# 预检查
# -----------------------------------------------------------------------------

# 检查 run.in 是否存在
if [ ! -f "$RUN_IN" ]; then
    echo "[ERROR] run.in not found at $RUN_IN"
    exit 1
fi

# 检查 nep.txt 是否存在
if [ ! -f "$NEP_TXT" ]; then
    echo "[WARNING] nep.txt not found at $NEP_TXT. Proceeding without it."
else
    echo "[INFO] Found nep.txt at $NEP_TXT"
fi

# 检查是否有名为 model.xyz 的文件，防止混淆
if [ -f "$WORK_DIR/model.xyz" ]; then
    echo "[WARNING] Found model.xyz in root directory. It will not be processed as a task input unless it matches *.xyz pattern."
fi

# -----------------------------------------------------------------------------
# 开始处理
# -----------------------------------------------------------------------------

# 计数器
count=0
total=$(ls "$WORK_DIR"/*.xyz 2>/dev/null | wc -l)

if [ "$total" -eq 0 ]; then
    echo "[ERROR] No .xyz files found in $WORK_DIR"
    exit 1
fi

echo "[INFO] Found $total .xyz files to process."

# 遍历所有 .xyz 文件
for xyz_file in "$WORK_DIR"/*.xyz; do
    # 检查文件是否存在（处理通配符不匹配的情况）
    if [ ! -e "$xyz_file" ]; then
        continue
    fi
    
    ((count++))
    
    # 获取文件名（例如 a.100.xyz）
    filename=$(basename "$xyz_file")
    
    # 获取目录名（去除后缀，例如 a.100）
    dirname="${filename%.*}"
    
    # 构建目标目录的绝对路径
    target_dir="$WORK_DIR/$dirname"
    
    echo "[PROGRESS] ($count/$total) Processing $filename..."
    echo "  -> Target Directory: $target_dir"

    # 1. 创建文件夹
    if [ ! -d "$target_dir" ]; then
        mkdir -p "$target_dir"
        if [ $? -ne 0 ]; then
            echo "[ERROR] Failed to create directory $target_dir"
            continue
        fi
    fi

    # 2. 复制并重命名 .xyz 文件 -> model.xyz
    cp "$xyz_file" "$target_dir/model.xyz"
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to copy $filename to $target_dir/model.xyz"
        continue
    fi

    # [新增] 修改 model.xyz 第二行，追加 pbc="T T F"
    # 使用 sed 将第二行末尾($)替换为 ' pbc="T T F"'
    sed -i '2s/$/ pbc="T T F"/' "$target_dir/model.xyz"
    if [ $? -ne 0 ]; then
        echo "[WARNING] Failed to append PBC info to $target_dir/model.xyz"
    else
        echo "  -> Appended pbc=\"T T F\" to model.xyz"
    fi

    # 3. 复制 run.in
    cp "$RUN_IN" "$target_dir/"
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to copy run.in to $target_dir"
        continue
    fi

    # 4. 复制 nep.txt (如果存在)
    if [ -f "$NEP_TXT" ]; then
        cp "$NEP_TXT" "$target_dir/"
        if [ $? -ne 0 ]; then
            echo "[ERROR] Failed to copy nep.txt to $target_dir"
            # nep.txt 复制失败可能不是致命错误，视具体情况而定，这里继续
        fi
    fi

    # 5. 进入文件夹并执行 gpumd
    echo "  -> Starting GPUMD simulation..."
    
    # 使用子 shell 或 pushd/popd 确保目录切换不影响主循环
    (
        cd "$target_dir" || exit 1
        
        # 记录开始时间
        start_ts=$(date +%s)
        
        # 执行 gpumd，重定向输出到日志文件
        # 注意：这里假设 gpumd 已经在环境变量中（由 slurm 脚本加载）
        gpumd > gpumd_log.txt 2>&1
        exit_code=$?
        
        end_ts=$(date +%s)
        duration=$((end_ts - start_ts))
        
        if [ $exit_code -eq 0 ]; then
            echo "  -> [SUCCESS] GPUMD finished in ${duration}s"
        else
            echo "  -> [FAILURE] GPUMD failed with exit code $exit_code. Check $target_dir/gpumd_log.txt for details."
            # 可以选择在这里 exit 1 终止整个脚本，或者 continue 继续下一个
            # 这里我们只记录错误，继续下一个任务
        fi
    )
    
    echo "--------------------------------------------------"

done

echo "[INFO] Batch processing completed."
