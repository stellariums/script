import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_validate
from sklearn.preprocessing import StandardScaler
import logging

"""
tune_knn.py
===========

此脚本用于优化 K-近邻回归模型 (KNN Regressor) 的超参数。
KNN 是一种基于实例的学习方法，其性能高度依赖于距离度量和邻居数量的选择。

调优参数：
- n_neighbors: 邻居数量 (K值)。K值过小容易过拟合，过大容易欠拟合。
- weights: 权重函数。'uniform' 表示所有邻居权重相同，'distance' 表示权重与距离成反比。
- p: 距离度量的幂参数。p=1 为曼哈顿距离，p=2 为欧几里得距离。
- algorithm: 用于计算最近邻的算法 ('auto', 'ball_tree', 'kd_tree', 'brute')。

用法：
    python tune_knn.py
"""

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')

def tune_knn():
    # 1. 加载数据
    data_path = r'c:\Users\USTC\Desktop\monolayer\ML\kappa.csv'
    features = ['Total_Rings', '5-ring(count)', '6-ring(count)', '7-ring(count)', '8-ring(count)', '9-ring(count)',
                '5-ring(%)', '6-ring(%)', '7-ring(%)', '8-ring(%)', '9-ring(%)']
    target = 'kx (W/m/K)'
    
    try:
        data = pd.read_csv(data_path)
        logging.info(f"KNN - 成功加载数据: {data.shape}")
        
        if data[features + [target]].isnull().any().any():
            data = data.dropna(subset=features + [target])
            
        X = data[features]
        y = data[target]
        
        # 2. 数据标准化 (KNN 对数据尺度非常敏感，必须标准化)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=features)
        
        # 3. 定义参数网格
        param_grid = {
            'n_neighbors': [3, 4, 5, 6, 7, 8, 9, 10, 12, 15],
            'weights': ['uniform', 'distance'],
            'p': [1, 2],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }
        
        knn = KNeighborsRegressor()
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # 4. 执行网格搜索
        logging.info("开始执行 KNN 网格搜索...")
        grid_search = GridSearchCV(
            estimator=knn,
            param_grid=param_grid,
            scoring='r2',
            cv=kf,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_scaled_df, y)
        
        logging.info("\nKNN 最佳参数组合:")
        logging.info(grid_search.best_params_)
        logging.info(f"KNN 最佳 CV R² 得分: {grid_search.best_score_:.4f}")
        
        # 5. 验证
        best_knn = grid_search.best_estimator_
        cv_results = cross_validate(best_knn, X_scaled_df, y, cv=kf,
                                  scoring={'r2': 'r2', 'mse': 'neg_mean_squared_error'},
                                  return_train_score=True, n_jobs=-1)
        
        logging.info("\nKNN 最终验证结果:")
        logging.info(f"训练集 R²: {cv_results['train_r2'].mean():.4f}")
        logging.info(f"测试集 R²: {cv_results['test_r2'].mean():.4f}")
        logging.info(f"均方误差 MSE: {-cv_results['test_mse'].mean():.4f}")
        
    except Exception as e:
        logging.error(f"KNN 调优发生错误: {e}")

if __name__ == "__main__":
    tune_knn()
