#!/bin/bash

# submit_jobs_struture.sh
# 批量提交 GPUMD 作业脚本
# 功能：
# 1. 遍历 model/ 下的 .xyz 文件
# 2. 创建对应文件夹并复制必要文件
# 3. 修改作业名称并提交

# 定义变量
MODEL_DIR="model"
CURRENT_DIR=$(pwd)
LOG_FILE="submit_jobs.log"

# 初始化日志
echo "Job submission started at $(date)" > "$LOG_FILE"

# 辅助函数：日志记录
log_msg() {
    local msg="$1"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $msg" | tee -a "$LOG_FILE"
}

# 辅助函数：错误检查
check_error() {
    if [ $? -ne 0 ]; then
        log_msg "ERROR: $1"
        return 1
    fi
    return 0
}

# 1. 检查必要文件是否存在
log_msg "Checking required files..."
REQUIRED_FILES=("nep.txt" "gmd.slurm" "run.in")
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        log_msg "Critical Error: Required file '$file' not found in current directory."
        exit 1
    fi
done

if [ ! -d "$MODEL_DIR" ]; then
    log_msg "Critical Error: Model directory '$MODEL_DIR' not found."
    exit 1
fi

# 2. 遍历 model 目录下的 .xyz 文件
# 使用 sort 确保处理顺序
found_files=$(ls "$MODEL_DIR"/*.xyz 2>/dev/null | sort)

if [ -z "$found_files" ]; then
    log_msg "No .xyz files found in $MODEL_DIR"
    exit 1
fi

count=0
for xyz_file in $found_files; do
    # 获取文件名（不带路径）
    filename=$(basename "$xyz_file")
    # 获取结构名称（去掉 .xyz 后缀）
    structure_name="${filename%.xyz}"
    
    log_msg "----------------------------------------"
    log_msg "Processing structure: $structure_name"
    
    # 创建任务文件夹（与 model 同级，即在当前目录下）
    work_dir="$CURRENT_DIR/$structure_name"
    
    if [ -d "$work_dir" ]; then
        log_msg "Warning: Directory '$structure_name' already exists. Skipping creation."
    else
        mkdir "$work_dir"
        check_error "Failed to create directory $work_dir" || continue
        log_msg "Created directory: $structure_name"
    fi
    
    # 复制 .xyz 文件
    cp "$xyz_file" "$work_dir/"
    check_error "Failed to copy $filename" || continue
    
    # 重要：GPUMD 默认读取 model.xyz，因此创建副本或重命名
    # 虽然用户只要求复制 .xyz，为了保证作业能运行，这里额外复制一份为 model.xyz
    cp "$xyz_file" "$work_dir/model.xyz"
    log_msg "Created model.xyz copy for GPUMD compatibility"

    # 复制运行所需文件
    for file in "${REQUIRED_FILES[@]}"; do
        cp "$file" "$work_dir/"
        check_error "Failed to copy $file" || continue
    done
    
    # 修改 gmd.slurm 中的任务名称
    # 假设原名称格式为 #SBATCH -J name
    # 替换为 #SBATCH -J lzy_结构名
    slurm_file="$work_dir/gmd.slurm"
    new_job_name="lzy_${structure_name}"
    
    # 使用 sed 进行替换 (兼容 Linux GNU sed)
    sed -i "s/#SBATCH -J .*/#SBATCH -J $new_job_name/" "$slurm_file"
    check_error "Failed to update job name in $slurm_file" || continue
    log_msg "Updated job name to: $new_job_name"
    
    # 提交作业
    # 切换到工作目录提交，确保日志文件生成在正确位置
    cd "$work_dir"
    submit_output=$(sbatch gmd.slurm 2>&1)
    submit_status=$?
    cd "$CURRENT_DIR"
    
    if [ $submit_status -eq 0 ]; then
        log_msg "Job submitted successfully: $submit_output"
        ((count++))
    else
        log_msg "Error submitting job: $submit_output"
    fi
done

log_msg "----------------------------------------"
log_msg "All tasks completed. Total jobs submitted: $count"
