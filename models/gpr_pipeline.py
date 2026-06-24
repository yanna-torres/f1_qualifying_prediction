"""
gpr_pipeline.py
=============
Advanced training pipeline using Gaussian Process Regression (GPR).
Executes an Ablation Study comparing two architectures:
1. Partitioned (Local): One GPR model per circuit.
2. Global: A single unified GPR model learning from all circuits combined.

Reads data centrally using utils.load_and_split().
Saves both models for reproducibility.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import (
    pairwise_distances_argmin_min,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from sklearn.model_selection import GridSearchCV
from scipy.stats import spearmanr

# Injeta a raiz do projeto no path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import TARGET_COL, OUT_PREDS, OUT_RESULTS, ABLATIONS
from utils import load_and_split, save_model

MODEL_NAME = "GPR"

# Limite máximo de amostras (Otimização O(n³))
MAX_INDUCING_POINTS_PARTITIONED = 300  
MAX_INDUCING_POINTS_GLOBAL = 600  # O global precisa de mais pontos para capturar a variação de todas as pistas

CATEGORICAL_COLS_BASE = ["Driver", "Team"]


class F1GaussianProcessPipeline:
    def __init__(self, X_train, X_test, y_train, train_df, test_df):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.train_df = train_df
        self.test_df = test_df

        # Estruturas para o Modelo Particionado
        self.models_part = {}
        self.preds_part = []
        self.cv_scores_part = {}
        self.best_params_part = {}

        # Estruturas para o Modelo Global
        self.model_glob = None
        self.preds_glob = []
        self.cv_score_glob = 0.0
        self.best_params_glob = {}

        os.makedirs(OUT_PREDS, exist_ok=True)
        os.makedirs(OUT_RESULTS.parent, exist_ok=True)

    def _build_preprocessor(self, X_sample, cat_features):
        """Constrói o preprocessor dinamicamente dependendo de quais features categóricas são passadas."""
        cat_cols = [c for c in cat_features if c in X_sample.columns]
        num_cols = [c for c in X_sample.columns if c not in cat_cols]

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_cols),
                (
                    "cat",
                    OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                    cat_cols,
                ),
            ]
        )
        return preprocessor

    def _extract_sparse_inducing_points(self, X, y, k_clusters):
        if len(X) <= k_clusters:
            return X, y

        print(f"  [K-Means] Comprimindo {len(X)} amostras para {k_clusters} pontos indutores...")
        kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init="auto")
        kmeans.fit(X)

        indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
        return X[indices], y[indices]

    # =========================================================================
    # ABORDAGEM 1: MODELO PARTICIONADO (LOCAL)
    # =========================================================================
    def run_partitioned_pipeline(self, model_tag):
        print(f"\n{'=' * 50}")
        print(f"🚀 INICIANDO ARQUITETURA 1: PARTICIONADO (POR CIRCUITO)")
        print(f"{'=' * 50}")
        
        circuitos = self.train_df["Circuit"].unique()
        
        for circuito in circuitos:
            print(f"--- Treinando Circuito ID: {circuito} ---")

            mask_train = self.train_df["Circuit"] == circuito
            mask_test = self.test_df["Circuit"] == circuito

            X_train_circ = self.X_train[mask_train].copy()
            X_test_circ = self.X_test[mask_test].copy()
            y_train_circ = self.y_train[mask_train.values]
            test_indices = self.test_df[mask_test].index

            if len(X_test_circ) == 0:
                continue

            # Remove a coluna Circuit pois a variância é zero localmente
            X_train_circ.drop(columns=["Circuit"], errors="ignore", inplace=True)
            X_test_circ.drop(columns=["Circuit"], errors="ignore", inplace=True)

            preprocessor = self._build_preprocessor(X_train_circ, CATEGORICAL_COLS_BASE)
            X_train_scaled = preprocessor.fit_transform(X_train_circ)
            X_test_scaled = preprocessor.transform(X_test_circ)

            X_train_sparse, y_train_sparse = self._extract_sparse_inducing_points(
                X_train_scaled, y_train_circ, k_clusters=MAX_INDUCING_POINTS_PARTITIONED
            )

            initial_length_scales = np.ones(X_train_scaled.shape[1])
            bounds = (1e-2, 1e4)

            kernel_matern = ConstantKernel(1.0) * Matern(length_scale=initial_length_scales, length_scale_bounds=bounds, nu=2.5) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-10, 1e1))
            kernel_rbf = ConstantKernel(1.0) * RBF(length_scale=initial_length_scales, length_scale_bounds=bounds) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-10, 1e1))
            
            param_grid = {"kernel": [kernel_matern, kernel_rbf], "alpha": [1e-5, 1e-2]}

            gpr_base = GaussianProcessRegressor(n_restarts_optimizer=5, normalize_y=True, random_state=42)
            grid_search = GridSearchCV(estimator=gpr_base, param_grid=param_grid, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1)
            
            grid_search.fit(X_train_sparse, y_train_sparse)

            best_model = grid_search.best_estimator_
            self.models_part[circuito] = best_model
            self.cv_scores_part[str(circuito)] = -grid_search.best_score_
            self.best_params_part[str(circuito)] = {k: str(v) for k, v in grid_search.best_params_.items()}

            y_pred, y_std = best_model.predict(X_test_scaled, return_std=True)

            for idx, pred, std in zip(test_indices, y_pred, y_std):
                self.preds_part.append({"index": idx, "pred_quali_position": pred, "pred_std": std})

        return self._evaluate_performance(self.preds_part, model_tag, self.models_part, self.best_params_part, np.mean(list(self.cv_scores_part.values())))

    # =========================================================================
    # ABORDAGEM 2: MODELO GLOBAL (ÚNICO)
    # =========================================================================
    def run_global_pipeline(self, model_tag):
        print(f"\n{'=' * 50}")
        print(f"🚀 INICIANDO ARQUITETURA 2: GLOBAL (TODAS AS PISTAS JUNTAS)")
        print(f"{'=' * 50}")

        X_train_global = self.X_train.copy()
        X_test_global = self.X_test.copy()
        y_train_global = self.y_train.copy()
        test_indices = self.test_df.index

        # MANTÉM a coluna "Circuit" e passa ela para o One-Hot Encoding
        cat_cols_global = CATEGORICAL_COLS_BASE + ["Circuit"]

        preprocessor = self._build_preprocessor(X_train_global, cat_cols_global)
        X_train_scaled = preprocessor.fit_transform(X_train_global)
        X_test_scaled = preprocessor.transform(X_test_global)
        
        print(f"  Features globais após One-Hot Encoding: {X_train_scaled.shape[1]}")

        X_train_sparse, y_train_sparse = self._extract_sparse_inducing_points(
            X_train_scaled, y_train_global, k_clusters=MAX_INDUCING_POINTS_GLOBAL
        )

        initial_length_scales = np.ones(X_train_scaled.shape[1])
        bounds = (1e-2, 1e4)

        # Para o Global, priorizamos o Matern por ser melhor com ruído e quebras de estacionaridade
        kernel_matern = ConstantKernel(1.0) * Matern(length_scale=initial_length_scales, length_scale_bounds=bounds, nu=2.5) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-10, 1e1))
        param_grid = {"kernel": [kernel_matern], "alpha": [1e-5, 1e-2]}

        gpr_base = GaussianProcessRegressor(n_restarts_optimizer=5, normalize_y=True, random_state=42)
        grid_search = GridSearchCV(estimator=gpr_base, param_grid=param_grid, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1)
        
        print("  [GridSearch] Treinando modelo gigante... Isto pode demorar um pouco.")
        grid_search.fit(X_train_sparse, y_train_sparse)

        best_model = grid_search.best_estimator_
        self.model_glob = best_model
        self.cv_score_glob = -grid_search.best_score_
        self.best_params_glob = {"GLOBAL_MODEL": {k: str(v) for k, v in grid_search.best_params_.items()}}

        print(f"  Melhor MAE Global (CV): {self.cv_score_glob:.3f}")

        y_pred, y_std = best_model.predict(X_test_scaled, return_std=True)

        for idx, pred, std in zip(test_indices, y_pred, y_std):
            self.preds_glob.append({"index": idx, "pred_quali_position": pred, "pred_std": std})

        return self._evaluate_performance(self.preds_glob, model_tag, self.model_glob, self.best_params_glob, self.cv_score_glob)

    # =========================================================================
    # AVALIADOR COMUM
    # =========================================================================
    def _evaluate_performance(self, predictions_list, model_tag, model_obj, best_params, cv_mae):
        preds_df = pd.DataFrame(predictions_list).set_index("index").sort_index()

        y_test_real = self.test_df[TARGET_COL].loc[preds_df.index]
        y_pred = preds_df["pred_quali_position"]

        rmse = np.sqrt(mean_squared_error(y_test_real, y_pred))
        mae = mean_absolute_error(y_test_real, y_pred)
        r2 = r2_score(y_test_real, y_pred)
        rho, pval = spearmanr(y_test_real, y_pred)

        final_df = self.test_df.copy()
        final_df["pred_quali_position"] = preds_df["pred_quali_position"]
        final_df["pred_std"] = preds_df["pred_std"]
        final_df["pred_quali_position_int"] = preds_df["pred_quali_position"].round().astype(int)

        erro_absoluto = abs(final_df["pred_quali_position_int"] - y_test_real)
        acerto_margem_1 = (erro_absoluto <= 1).mean() * 100
        acerto_margem_3 = (erro_absoluto <= 3).mean() * 100

        out_file = OUT_PREDS / f"{model_tag.lower()}_predictions.csv"
        final_df.to_csv(out_file, index=False)
        
        save_model(model_obj, model_tag, best_params=best_params, cv_mae=cv_mae)

        return {
            "model": model_tag,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Spearman_rho": rho,
            "Spearman_p": pval,
            "Top1_acc": acerto_margem_1 / 100,
            "Top3_acc": acerto_margem_3 / 100,
            "y_pred": y_pred.to_numpy(),
        }

def print_comparison_table(res_part, res_glob):
    """Gera um painel elegante no terminal comparando as duas arquiteturas."""
    print("\n\n" + "█" * 60)
    print(" 🏆 COMPARAÇÃO DE ARQUITETURAS GPR (ESTUDO DE ABLAÇÃO)")
    print("█" * 60)
    print(f"| {'Métrica':<15} | {'GPR Particionado (Local)':<25} | {'GPR Global (Único)':<15} |")
    print(f"|{'-'*17}|{'-'*27}|{'-'*20}|")
    
    mae_winner = "🏆" if res_part["MAE"] < res_glob["MAE"] else "   "
    mae_winner_glob = "🏆" if res_glob["MAE"] < res_part["MAE"] else "   "
    
    r2_winner = "🏆" if res_part["R2"] > res_glob["R2"] else "   "
    r2_winner_glob = "🏆" if res_glob["R2"] > res_part["R2"] else "   "
    
    print(f"| {'MAE (Erro)':<15} | {res_part['MAE']:<5.3f} posições {mae_winner:<11} | {res_glob['MAE']:<5.3f} posições {mae_winner_glob} |")
    print(f"| {'RMSE':<15} | {res_part['RMSE']:<25.3f} | {res_glob['RMSE']:<15.3f} |")
    print(f"| {'R² (Variância)':<15} | {res_part['R2']:<5.3f} {r2_winner:<19} | {res_glob['R2']:<5.3f} {r2_winner_glob:<8} |")
    print(f"| {'Spearman Rho':<15} | {res_part['Spearman_rho']:<25.3f} | {res_glob['Spearman_rho']:<15.3f} |")
    print(f"| {'Acurácia ±1':<15} | {res_part['Top1_acc']*100:<5.1f}% {'':<18} | {res_glob['Top1_acc']*100:<5.1f}% {'':<7} |")
    print("█" * 60 + "\n")


def main(ablation: str = "full", mode: str = "global"):
    """
    Parameters
    ----------
    ablation : str
        Key into config.ABLATIONS. 
    mode : str
        Define o que executar:
        - "global": Executa APENAS o modelo global vencedor (padrão oficial).
        - "partitioned": Executa APENAS o modelo antigo dividido por pista.
        - "compare": Executa ambos e imprime a tabela de comparação A/B.
    """
    if ablation not in ABLATIONS:
        raise ValueError(f"Unknown ablation '{ablation}'. Available: {list(ABLATIONS.keys())}")

    print(f"\n{'=' * 55}\n  {MODEL_NAME} PIPELINE (Mode: {mode.upper()})\n{'=' * 55}")

    X_train, X_test, y_train, y_test, train_df, test_df = load_and_split(extra_drop_cols=ABLATIONS[ablation])
    pipeline = F1GaussianProcessPipeline(X_train, X_test, y_train, train_df, test_df)

    base_tag = MODEL_NAME if ablation == "full" else f"{MODEL_NAME}_{ablation}"

    if mode == "compare":
        tag_part = f"{base_tag}_Partitioned"
        res_part = pipeline.run_partitioned_pipeline(model_tag=tag_part)

        tag_glob = f"{base_tag}_Global"
        res_glob = pipeline.run_global_pipeline(model_tag=tag_glob)

        print_comparison_table(res_part, res_glob)
        # Retornamos o modelo Global como vencedor para a tabela geral do run_all.py
        return res_glob 

    elif mode == "global":
        # Uso oficial diário: Rápido e limpo
        return pipeline.run_global_pipeline(model_tag=base_tag)

    elif mode == "partitioned":
        # Para testes específicos locais
        return pipeline.run_partitioned_pipeline(model_tag=base_tag)
        
    else:
        raise ValueError("O parâmetro 'mode' deve ser 'global', 'partitioned' ou 'compare'.")

if __name__ == "__main__":
    # Para testar apenas um, mude aqui embaixo. Exemplo: main(mode="partitioned")
    main(mode="global")