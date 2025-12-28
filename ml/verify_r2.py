"""
verify_r2.py
============

此脚本用于验证 `baseline.py` 中新添加的训练集 R² 分数计算功能。

主要功能：
1. 复用 `baseline.py` 中的 `MLPipeline` 类。
2. 加载数据并初始化模型。
3. 运行交叉验证评估 (`evaluate_models_cv`)，打印训练集和测试集的 R² 分数以及 MSE。

用法：
    python verify_r2.py

输出：
    控制台将显示每个模型的训练集 R²、测试集 R² 和均方误差 (MSE)。
"""

import pandas as pd
from baseline import MLPipeline
import logging

# 配置日志输出到控制台
logging.basicConfig(level=logging.INFO, format='%(message)s')

# 数据路径和特征定义
data_path = r'c:\Users\USTC\Desktop\monolayer\ML\kappa.csv'
features = ['Total_Rings', '5-ring(count)', '6-ring(count)', '7-ring(count)', '8-ring(count)', '9-ring(count)',
            '5-ring(%)', '6-ring(%)', '7-ring(%)', '8-ring(%)', '9-ring(%)']
target = 'kx (W/m/K)'

# 初始化并运行 pipeline 的评估部分
pipeline = MLPipeline(data_path, features, target)
pipeline.load_data()
pipeline.init_models()

# 仅运行评估步骤，不进行绘图和 SHAP 分析
pipeline.evaluate_models_cv()
