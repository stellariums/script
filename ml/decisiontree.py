import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from datetime import datetime
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(message)s')

def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _sanitize_filename_part(text: str) -> str:
    invalid = '<>:"/\\|?*'
    out = "".join("_" if ch in invalid else ch for ch in str(text))
    out = out.replace(" ", "_")
    return out[:120] if len(out) > 120 else out

def _ensure_output_dirs(base_dir: Path) -> dict:
    base_dir.mkdir(parents=True, exist_ok=True)
    dirs = {
        "base": base_dir,
        "tuning": base_dir / "tuning",
        "shap": base_dir / "shap",
        "pearson": base_dir / "pearson",
        "parity": base_dir / "parity",
    }
    for p in dirs.values():
        if isinstance(p, Path):
            p.mkdir(parents=True, exist_ok=True)
    return dirs

def _save_tuning_plot(results: pd.DataFrame, output_png: Path) -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    plt.figure(figsize=(10, 6))
    plt.plot(results["max_depth"], results["train_mse"], marker="o", label="Train MSE")
    plt.plot(results["max_depth"], results["test_mse"], marker="o", label="Test MSE")
    plt.xlabel("max_depth")
    plt.ylabel("Loss (MSE)")
    plt.title("决策树最大深度调优分析")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()

def _plot_parity(
    y_true: pd.Series,
    y_pred: np.ndarray,
    output_png: Path,
    r2: float,
    title: str,
) -> None:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    vmin = float(np.nanmin([y_true_arr.min(), y_pred_arr.min()]))
    vmax = float(np.nanmax([y_true_arr.max(), y_pred_arr.max()]))

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    plt.figure(figsize=(7, 7))
    plt.scatter(y_true_arr, y_pred_arr, s=20, alpha=0.75, edgecolors="none")
    plt.plot([vmin, vmax], [vmin, vmax], linestyle="--", color="black", linewidth=1.5, label="y = x")
    plt.xlabel("真实值")
    plt.ylabel("预测值")
    plt.title(f"{title} (R²={r2:.4f})")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()

def _plot_pearson_matrix_heatmap(
    X: pd.DataFrame,
    output_svg: Path,
    title: str = "Pearson 相关系数矩阵",
) -> None:
    corr = X.corr(method="pearson")
    labels = list(corr.columns)
    mat = corr.values

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    fig_w = max(8, 0.55 * len(labels))
    fig_h = max(6, 0.55 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_title(title)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(
                j,
                i,
                f"{mat[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r")
    fig.tight_layout()
    fig.savefig(output_svg, format="svg", bbox_inches="tight")
    plt.close(fig)

def _safe_make_shap_visuals(
    model: DecisionTreeRegressor,
    X_background: pd.DataFrame,
    X_explain: pd.DataFrame,
    output_dir: Path,
    ts: str,
    max_dependence_plots: int = 6,
) -> None:
    try:
        import shap
    except Exception as e:
        logging.warning(f"未能导入 shap，跳过 SHAP 可视化: {e}")
        return

    try:
        explainer = shap.TreeExplainer(model, data=X_background)
        shap_values = explainer.shap_values(X_explain)
        shap_values_arr = np.asarray(shap_values)
    except Exception as e:
        logging.warning(f"SHAP 计算失败，跳过 SHAP 可视化: {e}")
        return

    try:
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values_arr, X_explain, plot_type="bar", show=False)
        plt.title("SHAP 特征重要性（平均绝对值）")
        out = output_dir / f"shap_importance_bar_{ts}.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        logging.info(f"SHAP 特征重要性条形图已保存: {out}")
    except Exception as e:
        logging.warning(f"SHAP 条形图生成失败: {e}")

    try:
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values_arr, X_explain, show=False)
        plt.title("SHAP Summary Plot（多特征对比）")
        out = output_dir / f"shap_summary_beeswarm_{ts}.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        logging.info(f"SHAP Summary Plot 已保存: {out}")
    except Exception as e:
        logging.warning(f"SHAP Summary Plot 生成失败: {e}")

    try:
        mean_abs = np.mean(np.abs(shap_values_arr), axis=0)
        top_idx = np.argsort(-mean_abs)[:max_dependence_plots]
        for rank, idx in enumerate(top_idx, start=1):
            feature_name = X_explain.columns[int(idx)]
            safe_name = _sanitize_filename_part(feature_name)
            plt.figure(figsize=(8, 6))
            shap.dependence_plot(
                feature_name,
                shap_values_arr,
                X_explain,
                interaction_index="auto",
                show=False,
            )
            out = output_dir / f"shap_dependence_top{rank}_{safe_name}_{ts}.png"
            plt.savefig(out, dpi=300, bbox_inches="tight")
            plt.close()
        logging.info(f"SHAP Dependence Plot（Top{len(top_idx)}）已保存至: {output_dir}")
    except Exception as e:
        logging.warning(f"SHAP Dependence Plot 生成失败: {e}")

    try:
        try:
            shap.initjs()
        except Exception as e:
            logging.warning(f"shap.initjs 初始化失败，继续导出 HTML: {e}")

        n_show = min(3, len(X_explain))
        for i in range(n_show):
            fp = shap.force_plot(
                explainer.expected_value,
                shap_values_arr[i, :],
                X_explain.iloc[i, :],
                matplotlib=False,
            )
            out = output_dir / f"shap_force_plot_sample{i+1}_{ts}.html"
            shap.save_html(str(out), fp)
        logging.info(f"SHAP 交互式 force_plot 已保存（HTML）至: {output_dir}")
    except Exception as e:
        logging.warning(f"SHAP 交互式可视化导出失败: {e}")

def tune_decision_tree():
    data_path = r'c:\Users\USTC\Desktop\monolayer\ML\kappa.csv'
    features = [
        'Density (g/cm3)',
        '5-ring(%)',
        '6-ring(%)',
        '7-ring(%)',
        '8-ring(%)',
        '9-ring(%)',
        'avg_bond_length'
    ]
    target = 'kx (W/m/K)'

    try:
        script_dir = Path(__file__).resolve().parent
        out_dirs = _ensure_output_dirs(script_dir / "decisiontree")
        ts = _timestamp()

        data = pd.read_csv(data_path)
        logging.info(f"成功加载数据: {data.shape}")

        data = data.dropna(subset=features + [target]).reset_index(drop=True)

        X = data[features]
        y = data[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        max_depth_values = list(range(1, 21))
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        records = []

        for depth in max_depth_values:
            model = DecisionTreeRegressor(max_depth=depth, random_state=42)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_mse = mean_squared_error(y_train, y_train_pred)
            test_mse = mean_squared_error(y_test, y_test_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)

            cv_res = cross_validate(
                model,
                X_train,
                y_train,
                cv=kf,
                scoring={"mse": "neg_mean_squared_error", "r2": "r2"},
                return_train_score=False,
                n_jobs=-1,
            )
            cv_mse = -cv_res["test_mse"].mean()
            cv_r2 = cv_res["test_r2"].mean()

            records.append(
                {
                    "max_depth": depth,
                    "train_mse": float(train_mse),
                    "test_mse": float(test_mse),
                    "cv_mse": float(cv_mse),
                    "train_r2": float(train_r2),
                    "test_r2": float(test_r2),
                    "cv_r2": float(cv_r2),
                }
            )

        results = pd.DataFrame.from_records(records).sort_values("max_depth")

        logging.info("\n各 max_depth 的训练/测试 MSE、R² 与 5 折CV指标:")
        show_cols = ["max_depth", "train_mse", "test_mse", "cv_mse", "train_r2", "test_r2", "cv_r2"]
        logging.info(results[show_cols].to_string(index=False, justify="center"))

        best_row = results.loc[results["cv_mse"].idxmin()]
        best_depth = int(best_row["max_depth"])

        logging.info(f"\n最优 max_depth: {best_depth}")
        logging.info(
            f"选择依据: 5 折交叉验证的平均 MSE 最小 (cv_mse={best_row['cv_mse']:.6f})"
        )
        logging.info(
            f"对应 R²: train_r2={best_row['train_r2']:.4f}, test_r2={best_row['test_r2']:.4f}, cv_r2={best_row['cv_r2']:.4f}"
        )

        tuning_png = out_dirs["tuning"] / f"decision_tree_max_depth_tuning_{ts}.png"
        _save_tuning_plot(results, tuning_png)
        logging.info(f"\n调优曲线图已保存: {tuning_png}")

        best_model = DecisionTreeRegressor(max_depth=best_depth, random_state=42)
        best_model.fit(X_train, y_train)
        y_test_pred = best_model.predict(X_test)
        test_r2_final = float(r2_score(y_test, y_test_pred))

        parity_png = out_dirs["parity"] / f"parity_true_vs_pred_{ts}.png"
        _plot_parity(
            y_true=y_test,
            y_pred=y_test_pred,
            output_png=parity_png,
            r2=test_r2_final,
            title="真实值 vs 预测值 对角线图（测试集）",
        )
        logging.info(f"对角线散点图已保存: {parity_png}")

        pearson_svg = out_dirs["pearson"] / f"pearson_matrix_{ts}.svg"
        _plot_pearson_matrix_heatmap(X, pearson_svg)
        logging.info(f"Pearson 相关系数矩阵已保存: {pearson_svg}")

        _safe_make_shap_visuals(
            model=best_model,
            X_background=X_train,
            X_explain=X_train,
            output_dir=out_dirs["shap"],
            ts=ts,
        )

    except Exception as e:
        logging.error(f"发生错误: {e}")
        import traceback
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    tune_decision_tree()
