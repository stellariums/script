import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import KFold, cross_validate, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
import logging
import optuna
import matplotlib.pyplot as plt
import os

try:
    import shap
    _has_shap = True
except ImportError:
    shap = None
    _has_shap = False

"""
tune_svm.py
===========

此脚本用于优化支持向量回归 (SVR) 的超参数。
SVR 通过寻找一个超平面，使得大多数样本点都落在该平面的一定间隔 (epsilon) 内。

调优参数：
- kernel: 核函数。决定了模型的非线性能力。常用 'linear' (线性), 'rbf' (径向基), 'poly' (多项式)。
- C: 正则化参数。C 越大，对误差的容忍度越低（容易过拟合）；C 越小，容忍度越高（容易欠拟合）。
- gamma: 核系数 (仅 rbf, poly, sigmoid)。定义了单个训练样本的影响范围。
- epsilon: 容忍间隔。在这个间隔内的预测误差不会被计入损失函数。

用法：
    python tune_svm.py
"""

logging.basicConfig(level=logging.INFO, format='%(message)s')

def tune_svm():
    # 1. 加载数据
    data_path = r'c:\Users\USTC\Desktop\monolayer\ML\kappa.csv'
    features = ['Density (g/cm3)', 
                '5-ring(%)', '6-ring(%)', '7-ring(%)', '8-ring(%)', '9-ring(%)',
                'avg_bond_length',"var_bond_angle_deg2","avg_bond_angle_deg","var_bond_length"]
    target = 'kx (W/m/K)'
    
    try:
        data = pd.read_csv(data_path)
        logging.info(f"SVR - 成功加载数据: {data.shape}")
        
        if data[features + [target]].isnull().any().any():
            data = data.dropna(subset=features + [target])
            
        X = data[features]
        y = data[target]
        
        # 2. 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=features)
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        output_dir = os.path.join(os.path.dirname(__file__), "svr")
        os.makedirs(output_dir, exist_ok=True)

        def objective(trial):
            kernel = trial.suggest_categorical('kernel', ['rbf', 'linear'])
            params = {
                'kernel': kernel,
                'C': trial.suggest_float('C', 0.1, 200.0, log=True),
                'epsilon': trial.suggest_float('epsilon', 0.01, 0.2)
            }
            if kernel == 'rbf':
                params['gamma'] = trial.suggest_float('gamma', 0.001, 1.0, log=True)
            svr = SVR(**params)
            scores = cross_val_score(svr, X_scaled_df, y, cv=kf, scoring='r2', n_jobs=-1)
            return scores.mean()

        logging.info("开始执行 SVR Optuna 调参...")
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100, n_jobs=-1)

        logging.info("\nSVR 最佳参数组合:")
        logging.info(study.best_params)
        logging.info(f"SVR 最佳 CV R² 得分: {study.best_value:.4f}")

        best_params = study.best_params
        best_svr = SVR(**best_params)
        cv_results = cross_validate(best_svr, X_scaled_df, y, cv=kf,
                                  scoring={'r2': 'r2', 'mse': 'neg_mean_squared_error'},
                                  return_train_score=True, n_jobs=-1)
        
        logging.info("\nSVR 最终验证结果:")
        logging.info(f"训练集 R²: {cv_results['train_r2'].mean():.4f}")
        logging.info(f"测试集 R²: {cv_results['test_r2'].mean():.4f}")
        logging.info(f"均方误差 MSE: {-cv_results['test_mse'].mean():.4f}")
        
        df_corr = X_scaled_df.copy()
        df_corr[target] = y.values
        spearman_series = df_corr.corr(method='spearman')[target].drop(target)
        spearman_series_sorted = spearman_series.sort_values(ascending=False)
        plt.figure(figsize=(8, 6))
        spearman_series_sorted.plot(kind='bar')
        plt.ylabel('Spearman correlation')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'svr_spearman_bar.png'), dpi=300, bbox_inches='tight')
        plt.close()
        logging.info("SVR Spearman 条形图已保存: svr_spearman_bar.png")

        pearson_corr = X_scaled_df.corr(method='pearson')
        pearson_corr.to_csv(os.path.join(output_dir, 'svr_pearson_corr.csv'))
        mask = np.triu(np.ones_like(pearson_corr, dtype=bool), k=1)
        masked_corr = pearson_corr.mask(mask)
        plt.figure(figsize=(8, 6))
        cmap = plt.cm.get_cmap('coolwarm').copy()
        cmap.set_bad(color='white')
        im = plt.imshow(masked_corr, cmap=cmap, vmin=-1, vmax=1)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks(range(len(masked_corr.columns)), masked_corr.columns, rotation=90)
        plt.yticks(range(len(masked_corr.index)), masked_corr.index)
        for i in range(masked_corr.shape[0]):
            for j in range(masked_corr.shape[1]):
                val = masked_corr.iloc[i, j]
                if not np.isnan(val):
                    plt.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=6, color='black')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'svr_pearson_corr_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        logging.info("SVR Pearson 相关系数矩阵及热力图已保存")

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled_df, y, test_size=0.2, random_state=42
        )
        best_svr.fit(X_train, y_train)
        y_train_pred = best_svr.predict(X_train)
        y_test_pred = best_svr.predict(X_test)
        plt.figure(figsize=(6, 6))
        plt.scatter(y_train, y_train_pred, alpha=0.7, label='Train', color='tab:blue')
        plt.scatter(y_test, y_test_pred, alpha=0.7, label='Test', color='tab:orange')
        all_actual = pd.concat([y_train, y_test])
        all_pred = np.concatenate([y_train_pred, y_test_pred])
        min_val = min(all_actual.min(), all_pred.min())
        max_val = max(all_actual.max(), all_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'svr_pred_vs_actual.png'), dpi=300, bbox_inches='tight')
        plt.close()
        logging.info("SVR Predicted vs Actual 散点图已保存: svr_pred_vs_actual.png")
        
        if _has_shap:
            try:
                background = X_train.sample(
                    n=min(50, len(X_train)), random_state=42
                )
                explainer = shap.KernelExplainer(
                    best_svr.predict, background
                )
                shap_values = explainer.shap_values(
                    X_scaled_df, nsamples=100
                )
                shap_values_arr = np.asarray(shap_values)
                plt.figure(figsize=(8, 6))
                last_scatter = None
                for i, f in enumerate(features):
                    vals = shap_values_arr[:, i]
                    feat_vals = X[f].values
                    vmin = feat_vals.min()
                    vmax = feat_vals.max()
                    if vmax > vmin:
                        colors = (feat_vals - vmin) / (vmax - vmin)
                    else:
                        colors = np.zeros_like(feat_vals)
                    y_jitter = np.random.normal(
                        loc=i, scale=0.1, size=vals.shape[0]
                    )
                    last_scatter = plt.scatter(
                        vals,
                        y_jitter,
                        c=colors,
                        cmap='coolwarm',
                        alpha=0.6,
                        s=10,
                    )
                plt.axvline(0, color='grey', linestyle='--', linewidth=1)
                plt.yticks(range(len(features)), features)
                plt.xlabel('SHAP value')
                if last_scatter is not None:
                    cbar = plt.colorbar(last_scatter, pad=0.01)
                    cbar.set_label('feature value')
                plt.tight_layout()
                plt.savefig(
                    os.path.join(output_dir, 'svr_shap_beeswarm.png'),
                    dpi=300,
                    bbox_inches='tight',
                )
                plt.close()
                logging.info(
                    "SVR SHAP 蜂群图已保存: svr_shap_beeswarm.png"
                )
            except Exception as e:
                logging.warning(f"SVR SHAP 可视化失败: {e}")
        
    except Exception as e:
        logging.error(f"SVR 调优发生错误: {e}")

if __name__ == "__main__":
    tune_svm()
