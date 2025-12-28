import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.model_selection import KFold, cross_validate, cross_val_score, cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler
import optuna
import logging
import matplotlib.pyplot as plt
import os

"""
tune_xgboost.py
===============

此脚本用于全面优化 XGBoost 回归模型的超参数。
XGBoost 是一个强大的梯度提升库，拥有大量可调参数。
考虑到数据集较小，我们重点对比 `gbtree` (树模型) 和 `gblinear` (线性模型) 的表现，并对相关参数进行细致调优。

调优策略：
- 首先使用 RandomizedSearchCV 在较大的参数空间中进行搜索。
- 同时包含针对 `gbtree` 和 `gblinear` 的特定参数。

调优参数：
- booster: 'gbtree' 或 'gblinear'。
- learning_rate: 学习率。
- n_estimators: 迭代次数。
- max_depth: 树深 (仅 gbtree)。
- min_child_weight: 叶子节点最小权重和 (仅 gbtree)。
- gamma: 分裂所需的最小损失减少 (仅 gbtree)。
- subsample: 样本采样率。
- colsample_bytree: 特征采样率。
- reg_alpha (L1), reg_lambda (L2): 正则化项。

用法：
    python tune_xgboost.py
"""

logging.basicConfig(level=logging.INFO, format='%(message)s')

def tune_xgboost():
    # 1. 加载数据
    data_path = r'c:\Users\USTC\Desktop\monolayer\ML\kappa.csv'
    features = [
                #  'Density (g/cm3)',
                # '5-ring(%)', '6-ring(%)', '7-ring(%)', '8-ring(%)', '9-ring(%)',
                # 'avg_bond_length',"var_bond_angle_deg2"
                "avg_bond_angle_deg","var_bond_length"
                ]
    target = 'kx (W/m/K)'
    
    try:
        data = pd.read_csv(data_path)
        logging.info(f"XGBoost - 成功加载数据: {data.shape}")
        
        if data[features + [target]].isnull().any().any():
            data = data.dropna(subset=features + [target])
            
        X = data[features]
        y = data[target]
        
        # 2. 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=features)
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        output_dir = os.path.join(os.path.dirname(__file__), "xgb")
        os.makedirs(output_dir, exist_ok=True)
        
        def objective(trial):
            booster = trial.suggest_categorical('booster', ['gbtree', 'gblinear'])
            params = {
                'booster': booster,
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
            }
            if booster == 'gbtree':
                params.update({
                    'max_depth': trial.suggest_int('max_depth', 3, 8),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                    'gamma': trial.suggest_float('gamma', 0.0, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                })
            model = XGBRegressor(random_state=42, n_jobs=-1, **params)
            scores = cross_val_score(model, X_scaled_df, y, cv=kf, scoring='r2', n_jobs=-1)
            return scores.mean()
        
        logging.info("开始执行 XGBoost Optuna 调参...")
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100, n_jobs=-1)
        
        logging.info("\nXGBoost 最佳参数组合:")
        logging.info(study.best_params)
        logging.info(f"XGBoost 最佳 CV R² 得分: {study.best_value:.4f}")
        
        best_params = study.best_params
        best_booster = best_params.get('booster', 'gbtree')
        best_xgb = XGBRegressor(random_state=42, n_jobs=-1, **best_params)
        
        cv_results = cross_validate(best_xgb, X_scaled_df, y, cv=kf,
                                  scoring={'r2': 'r2', 'mse': 'neg_mean_squared_error'},
                                  return_train_score=True, n_jobs=-1)
        
        logging.info("\nXGBoost 最终验证结果:")
        logging.info(f"训练集 R²: {cv_results['train_r2'].mean():.4f}")
        logging.info(f"测试集 R²: {cv_results['test_r2'].mean():.4f}")
        logging.info(f"均方误差 MSE: {-cv_results['test_mse'].mean():.4f}")

        best_xgb.fit(X_scaled_df, y)
        booster_for_imp = best_xgb.get_booster()
        gain_score = booster_for_imp.get_score(importance_type='gain')
        gain_series = pd.Series([gain_score.get(f, 0.0) for f in features], index=features)
        gain_series_sorted = gain_series.sort_values(ascending=False)
        plt.figure(figsize=(8, 6))
        gain_series_sorted.plot(kind='bar')
        plt.ylabel('feature importance (gain)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'xgb_feature_importance_gain.png'), dpi=300, bbox_inches='tight')
        plt.close()
        logging.info("XGBoost 特征重要性条形图已保存: xgb_feature_importance_gain.png")

        df_corr = X_scaled_df.copy()
        df_corr[target] = y.values
        spearman_series = df_corr.corr(method='spearman')[target].drop(target)
        spearman_series_sorted = spearman_series.sort_values(ascending=False)
        plt.figure(figsize=(8, 6))
        spearman_series_sorted.plot(kind='bar')
        plt.ylabel('Spearman correlation')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'xgb_spearman_bar.png'), dpi=300, bbox_inches='tight')
        plt.close()
        logging.info("XGBoost Spearman 条形图已保存: xgb_spearman_bar.png")

        pearson_corr = X_scaled_df.corr(method='pearson')
        pearson_corr.to_csv(os.path.join(output_dir, 'xgb_pearson_corr.csv'))
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
        plt.savefig(os.path.join(output_dir, 'xgb_pearson_corr_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        logging.info("XGBoost Pearson 相关系数矩阵及热力图已保存")

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled_df, y, test_size=0.2, random_state=42
        )
        best_xgb.fit(X_train, y_train)
        y_train_pred = best_xgb.predict(X_train)
        y_test_pred = best_xgb.predict(X_test)
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
        plt.savefig(os.path.join(output_dir, 'xgb_pred_vs_actual.png'), dpi=300, bbox_inches='tight')
        plt.close()
        logging.info("XGBoost Predicted vs Actual 散点图已保存: xgb_pred_vs_actual.png")
        
        if best_booster != 'gbtree':
            logging.info(f"最佳 booster 为 {best_booster}，跳过 SHAP 树模型可视化")
        else:
            try:
                best_xgb.fit(X_scaled_df, y)
                dmatrix = xgb.DMatrix(X_scaled_df, label=y, feature_names=features)
                booster = best_xgb.get_booster()
                shap_values = booster.predict(dmatrix, pred_contribs=True)
                shap_values_arr = np.asarray(shap_values)
                shap_values_arr = shap_values_arr[:, :-1]
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
                    y_jitter = np.random.normal(loc=i, scale=0.1, size=vals.shape[0])
                    last_scatter = plt.scatter(vals, y_jitter, c=colors, cmap='coolwarm', alpha=0.6, s=10)
                plt.axvline(0, color='grey', linestyle='--', linewidth=1)
                plt.yticks(range(len(features)), features)
                plt.xlabel('SHAP value')
                if last_scatter is not None:
                    cbar = plt.colorbar(last_scatter, pad=0.01)
                    cbar.set_label('feature value')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'xgb_shap_beeswarm.png'), dpi=300, bbox_inches='tight')
                plt.close()
                logging.info("XGBoost SHAP 蜂群图已保存: xgb_shap_beeswarm.png")
            except Exception as e:
                logging.warning(f"XGBoost SHAP 可视化失败: {e}")
        
    except Exception as e:
        logging.error(f"XGBoost 调优发生错误: {e}")

if __name__ == "__main__":
    tune_xgboost()
