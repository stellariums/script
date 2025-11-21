#!/bin/bash

# 遍历当前目录下所有子目录
for DIR in */; do
    # 排除当前目录中不是需要算的目录（例如没有下划线的可以自己加判断）
    [ -d "$DIR" ] || continue

    # 去掉末尾的 /
    DIR=${DIR%/}

    cp run.in "$DIR"
    cp gmd.slurm "$DIR"
    echo "提交目录 $DIR 的任务"
    cd "$DIR"
    # 通过环境变量把目录名传给 gmd.slurm
    sbatch  gmd.slurm
    cd ..
done
