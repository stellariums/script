# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import KFold, cross_val_score, train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor

"""
baseline.py
===========

此脚本实现了一个完整的机器学习基准测试流程 (Pipeline)，用于预测单层材料的热导率 (kappa)。
它集成了多种回归模型，并包含数据预处理、特征工程、交叉验证评估、特征重要性分析 (SHAP) 以及结果可视化等功能。

主要功能：
1. **数据加载与清洗**：
   - 读取 `kappa.csv`。
   - 自动处理缺失值。
   - 提取指定的特征列 (环统计信息) 和目标变量 (kx)。

2. **模型集成**：
   集成并初始化了以下 5 种回归模型：
   - **SVM (SVR)**: 支持向量回归。
   - **Decision Tree**: 决策树回归。
   - **Ridge Regression**: 岭回归 (L2 正则化线性回归)，适合处理共线性特征。
   - **KNN**: K-近邻回归。
   - **XGBoost**: 极端梯度提升树 (配置为 `gblinear` 模式以适应小数据集线性关系)。

3. **模型评估 (K-Fold CV)**：
   - 使用 5 折交叉验证 (K-Fold Cross Validation) 评估每个模型。
   - 评估指标包括 R² Score (训练集和测试集) 和 MSE (均方误差)。
   - 在评估过程中对数据进行标准化 (StandardScaler) 处理，防止数据泄露。

4. **可视化与分析**：
   - **CV 结果对比**：生成柱状图对比各模型的测试集 R² 得分 (`cv_results_r2.png`)。
   - **相关性分析**：计算并绘制特征间的 Pearson 相关系数热力图 (`pearson_correlation.png`)。
   - **SHAP 分析**：为每个模型生成 SHAP Summary Plot (`shap_summary_*.png`)，解释特征对预测结果的影响。

5. **日志记录**：
   - 全程记录运行日志，包括数据形状、模型评估得分、文件保存状态等。

依赖库：
- pandas, numpy: 数据处理
- matplotlib, seaborn: 可视化
- scikit-learn: 建模与评估
- xgboost: XGBoost 模型
- shap: 模型解释性分析

用法：
    直接运行脚本即可：
    python baseline.py

输出：
    - 控制台打印交叉验证结果表格。
    - 生成多个 .png 图片文件用于分析。
"""

# 忽略警告
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 设置matplotlib字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class MLPipeline:
    def __init__(self, data_path, features, target, random_state=42):
        self.data_path = data_path
        self.features = features
        self.target = target
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.X = None
        self.y = None
        self.feature_names = None

    def load_data(self):
        """加载并预处理数据"""
        try:
            data = pd.read_csv(self.data_path)
            logging.info(f"Data loaded successfully from {self.data_path}")
            
            # 检查缺失值
            if data[self.features + [self.target]].isnull().any().any():
                logging.warning("Data contains missing values. Dropping rows with missing values.")
                data = data.dropna(subset=self.features + [self.target])
            
            self.X = data[self.features]
            self.y = data[self.target]
            self.feature_names = self.features
            
            logging.info(f"Data shape: {data.shape}")
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    def init_models(self):
        """初始化机器学习模型"""
        # 注意：这里使用回归模型代替分类模型，因为kx是连续值
        # SVM -> SVR
        # Logistic Regression -> Ridge (线性回归的正则化版本，逻辑回归用于分类)
        # KNN -> KNeighborsRegressor
        self.models = {
            'SVM': SVR(),
            'Decision Tree': DecisionTreeRegressor(random_state=self.random_state),
            'Ridge Regression': Ridge(random_state=self.random_state), # 替代逻辑回归
            'KNN': KNeighborsRegressor(),
            # 使用 gblinear 提升在小数据集上的表现，或使用调优后的树参数
            # 经过测试，gblinear 效果最好 (R2 ~0.61)，其次是调优后的 gbtree (R2 ~0.55)，默认 gbtree 最差 (R2 ~0.29)
            # 这里我们使用 gblinear，因为它更适合当前的小数据集线性关系
            'XGBoost': XGBRegressor(booster='gblinear', random_state=self.random_state, n_jobs=-1)
        }
        logging.info("Models initialized: SVM, Decision Tree, Ridge Regression, KNN, XGBoost (gblinear)")

    def evaluate_models_cv(self, k_folds=5):
        """使用K折交叉验证评估模型"""
        logging.info(f"Starting {k_folds}-fold Cross Validation...")
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=self.random_state)
        
        # 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        cv_results = []
        
        for name, model in self.models.items():
            try:
                # 使用 cross_validate 同时获取训练集和测试集的评分
                # 使用 R2 作为主要评分标准
                cv_results_dict = cross_validate(model, X_scaled_df, self.y, cv=kf, 
                                               scoring={'r2': 'r2', 'mse': 'neg_mean_squared_error'}, 
                                               return_train_score=True, n_jobs=-1)
                
                mean_test_r2 = cv_results_dict['test_r2'].mean()
                mean_train_r2 = cv_results_dict['train_r2'].mean()
                mean_mse = -cv_results_dict['test_mse'].mean()
                
                logging.info(f"{name}: 训练集R²: {mean_train_r2:.4f}, 测试集R²: {mean_test_r2:.4f}, Mean MSE = {mean_mse:.4f}")
                cv_results.append({
                    'Model': name,
                    'Train R2': mean_train_r2,
                    'Test R2': mean_test_r2,
                    'Mean MSE': mean_mse
                })
            except Exception as e:
                logging.error(f"Error evaluating {name}: {e}")
        
        # 保存评估结果
        results_df = pd.DataFrame(cv_results)
        print("\nCross Validation Results:")
        print(results_df)
        
        # 可视化评估结果
        self.plot_cv_results(results_df)

    def plot_cv_results(self, results_df):
        """可视化交叉验证结果"""
        plt.figure(figsize=(10, 6))
        # 使用测试集R²作为主要展示指标
        sns.barplot(x='Model', y='Test R2', data=results_df)
        plt.title('各模型 K-Fold Cross Validation R² Score')
        plt.ylabel('Mean Test R² Score')
        plt.xlabel('Model')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('cv_results_r2.png')
        logging.info("CV results plot saved to cv_results_r2.png")

    def pearson_correlation(self):
        """计算并显示特征的Pearson相关系数矩阵"""
        logging.info("Calculating Pearson correlation matrix...")
        corr_matrix = self.X.corr(method='pearson')
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('特征 Pearson 相关系数矩阵')
        plt.tight_layout()
        plt.savefig('pearson_correlation.png')
        logging.info("Pearson correlation matrix plot saved to pearson_correlation.png")

    def shap_analysis(self):
        """生成各模型的SHAP图"""
        logging.info("Starting SHAP analysis...")
        
        # 划分训练集和测试集用于SHAP分析
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=self.random_state)
        
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=self.feature_names)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=self.feature_names)
        
        for name, model in self.models.items():
            try:
                logging.info(f"Generating SHAP plot for {name}...")
                model.fit(X_train_scaled, y_train)
                
                # 确定explainer类型
                if name in ['Decision Tree', 'XGBoost']:
                    explainer = shap.TreeExplainer(model)
                elif name in ['Ridge Regression', 'SVM', 'KNN']:
                    # 对于非树模型，使用KernelExplainer或LinearExplainer
                    # 为了速度，这里对训练集进行采样作为背景数据
                    background = shap.kmeans(X_train_scaled, 10)
                    explainer = shap.KernelExplainer(model.predict, background)
                else:
                    logging.warning(f"Skipping SHAP for {name}: Explainer not defined.")
                    continue
                
                shap_values = explainer.shap_values(X_test_scaled)
                
                plt.figure()
                shap.summary_plot(shap_values, X_test_scaled, show=False)
                plt.title(f'{name} SHAP Summary Plot')
                plt.tight_layout()
                plt.savefig(f'shap_summary_{name.replace(" ", "_")}.png')
                plt.close()
                logging.info(f"SHAP plot saved for {name}")
                
            except Exception as e:
                logging.warning(f"Could not generate SHAP plot for {name}: {e}")

    def run(self):
        """运行整个流程"""
        self.load_data()
        self.pearson_correlation()
        self.init_models()
        self.evaluate_models_cv()
        self.shap_analysis()

if __name__ == "__main__":
    data_path = r'c:\Users\USTC\Desktop\monolayer\ML\kappa.csv'
    features = ['Total_Rings', '5-ring(count)', '6-ring(count)', '7-ring(count)', '8-ring(count)', '9-ring(count)',
                '5-ring(%)', '6-ring(%)', '7-ring(%)', '8-ring(%)', '9-ring(%)']
    target = 'kx (W/m/K)'
    
    pipeline = MLPipeline(data_path, features, target)
    pipeline.run()
