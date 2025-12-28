import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.model_selection import GridSearchCV, KFold, cross_validate
from sklearn.preprocessing import StandardScaler
import logging

"""
tune_linear_models.py
=====================

此脚本用于优化线性回归模型的超参数。
虽然普通的线性回归 (Linear Regression) 没有太多可调参数，但其正则化变体 (Ridge, Lasso) 
对于防止过拟合和处理多重共线性至关重要。

包含模型：
1. **Ridge Regression (岭回归)**: 使用 L2 正则化。
2. **Lasso Regression**: 使用 L1 正则化，可用于特征选择。
3. **ElasticNet**: 结合了 L1 和 L2 正则化。

调优参数：
- alpha: 正则化强度。值越大，正则化越强，模型越简单（可能欠拟合）；值越小，越接近普通线性回归。
- l1_ratio: ElasticNet 特有，控制 L1 和 L2 的混合比例。

用法：
    python tune_linear_models.py
"""

logging.basicConfig(level=logging.INFO, format='%(message)s')

def tune_linear():
    # 1. 加载数据
    data_path = r'c:\Users\USTC\Desktop\monolayer\ML\kappa.csv'
    features = ['Density (g/cm3)','Total_Rings', '5-ring(count)', '6-ring(count)', '7-ring(count)', '8-ring(count)', '9-ring(count)',
                '5-ring(%)', '6-ring(%)', '7-ring(%)', '8-ring(%)', '9-ring(%)']
    target = 'kx (W/m/K)'
    
    try:
        data = pd.read_csv(data_path)
        logging.info(f"Linear Models - 成功加载数据: {data.shape}")
        
        if data[features + [target]].isnull().any().any():
            data = data.dropna(subset=features + [target])
            
        X = data[features]
        y = data[target]
        
        # 2. 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=features)
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # ==========================================
        # 3.1 Ridge Regression 调优
        # ==========================================
        logging.info("\n=== Ridge Regression 调优 ===")
        ridge_params = {
            'alpha': np.logspace(-4, 4, 100) # 生成从 10^-4 到 10^4 的 100 个值
        }
        
        grid_ridge = GridSearchCV(Ridge(random_state=42), ridge_params, scoring='r2', cv=kf, n_jobs=-1)
        grid_ridge.fit(X_scaled_df, y)
        
        logging.info(f"Ridge 最佳 Alpha: {grid_ridge.best_params_['alpha']:.4f}")
        logging.info(f"Ridge 最佳 CV R²: {grid_ridge.best_score_:.4f}")
        
        # 验证 Ridge
        cv_ridge = cross_validate(grid_ridge.best_estimator_, X_scaled_df, y, cv=kf,
                                scoring={'r2': 'r2', 'mse': 'neg_mean_squared_error'},
                                return_train_score=True, n_jobs=-1)
        logging.info(f"Ridge 最终验证 - 训练集 R²: {cv_ridge['train_r2'].mean():.4f}, 测试集 R²: {cv_ridge['test_r2'].mean():.4f}")

        # ==========================================
        # 3.2 Lasso Regression 调优
        # ==========================================
        logging.info("\n=== Lasso Regression 调优 ===")
        lasso_params = {
            'alpha': np.logspace(-4, 4, 100)
        }
        
        grid_lasso = GridSearchCV(Lasso(random_state=42), lasso_params, scoring='r2', cv=kf, n_jobs=-1)
        grid_lasso.fit(X_scaled_df, y)
        
        logging.info(f"Lasso 最佳 Alpha: {grid_lasso.best_params_['alpha']:.4f}")
        logging.info(f"Lasso 最佳 CV R²: {grid_lasso.best_score_:.4f}")
        
        # 验证 Lasso
        cv_lasso = cross_validate(grid_lasso.best_estimator_, X_scaled_df, y, cv=kf,
                                scoring={'r2': 'r2', 'mse': 'neg_mean_squared_error'},
                                return_train_score=True, n_jobs=-1)
        logging.info(f"Lasso 最终验证 - 训练集 R²: {cv_lasso['train_r2'].mean():.4f}, 测试集 R²: {cv_lasso['test_r2'].mean():.4f}")

    except Exception as e:
        logging.error(f"线性模型调优发生错误: {e}")

if __name__ == "__main__":
    tune_linear()
