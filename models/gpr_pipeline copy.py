"""
gpr_pipeline.py
=============
Advanced training pipeline using Gaussian Process Regression (GPR).
Implements partitioned models (per circuit), safe One-Hot Encoding,
dimensionality reduction (K-Means / inducing points), and GridSearchCV.

Supports feature ablation studies via the `ablation` parameter to
main(), using the named column sets defined in config.ABLATIONS.
Reads data centrally using utils.load_and_split() to maintain perfect 
alignment with the rest of the team's models.
Saves the partitioned model dictionary and hyperparameters for reproducibility.
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
# Adicionamos o import do save_model da sua equipa
from utils import load_and_split, save_model

MODEL_NAME = "GPR"

# Limite máximo de amostras por circuito (Otimização O(n³))
MAX_INDUCING_POINTS = 300  

# Colunas que vieram com LabelEncoder do data.py e precisam virar binárias (OHE)
CATEGORICAL_COLS = ["Driver", "Team"]


class F1GaussianProcessPipeline:
    def __init__(self, X_train, X_test, y_train, train_df, test_df):
        print("Inicializando Pipeline GPR...")
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.train_df = train_df
        self.test_df = test_df

        self.models = {}  # Dicionário para guardar o melhor modelo de cada circuito
        self.predictions = []  # Lista para reconstruir as predições do teste
        
        # --- NOVOS ATRIBUTOS PARA GUARDAR OS PARÂMETROS ---
        self.best_params = {}
        self.cv_scores = {}

        # Garante que os diretórios de saída existam
        os.makedirs(OUT_PREDS, exist_ok=True)
        os.makedirs(OUT_RESULTS.parent, exist_ok=True)

    def _build_preprocessor(self, X_sample):
        cat_cols = [c for c in CATEGORICAL_COLS if c in X_sample.columns]
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

        print(f"[K-Means] Comprimindo {len(X)} amostras para {k_clusters} pontos indutores...")
        kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init="auto")
        kmeans.fit(X)

        indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
        return X[indices], y[indices]

    def run_pipeline(self, model_tag=MODEL_NAME):
        """Orquestra o treinamento particionado por circuito."""
        circuitos = self.train_df["Circuit"].unique()
        print(f"Encontrados {len(circuitos)} circuitos para particionamento.\n")

        for circuito in circuitos:
            print(f"{'=' * 50}")
            print(f"TREINANDO CIRCUITO ID: {circuito}")

            mask_train = self.train_df["Circuit"] == circuito
            mask_test = self.test_df["Circuit"] == circuito

            X_train_circ = self.X_train[mask_train].copy()
            X_test_circ = self.X_test[mask_test].copy()
            y_train_circ = self.y_train[mask_train.values]

            test_indices = self.test_df[mask_test].index

            if len(X_test_circ) == 0:
                print("Nenhum dado de teste para este circuito. Pulando predição...")
                continue

            X_train_circ.drop(columns=["Circuit"], errors="ignore", inplace=True)
            X_test_circ.drop(columns=["Circuit"], errors="ignore", inplace=True)

            preprocessor = self._build_preprocessor(X_train_circ)
            X_train_scaled = preprocessor.fit_transform(X_train_circ)
            X_test_scaled = preprocessor.transform(X_test_circ)

            num_features = X_train_scaled.shape[1]
            print(f"Features após One-Hot Encoding: {num_features}")

            X_train_sparse, y_train_sparse = self._extract_sparse_inducing_points(
                X_train_scaled, y_train_circ, k_clusters=MAX_INDUCING_POINTS
            )

            initial_length_scales = np.ones(num_features)
            bounds = (1e-2, 1e4)

            kernel_matern = ConstantKernel(1.0) * Matern(
                length_scale=initial_length_scales, length_scale_bounds=bounds, nu=2.5
            ) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-10, 1e1))
            
            kernel_rbf = ConstantKernel(1.0) * RBF(
                length_scale=initial_length_scales, length_scale_bounds=bounds
            ) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-10, 1e1))
            
            param_grid = {
                "kernel": [kernel_matern, kernel_rbf],
                "alpha": [1e-5, 1e-2], 
            }

            gpr_base = GaussianProcessRegressor(
                n_restarts_optimizer=5, normalize_y=True, random_state=42
            )

            grid_search = GridSearchCV(
                estimator=gpr_base,
                param_grid=param_grid,
                cv=3,
                scoring="neg_mean_absolute_error",
                n_jobs=-1, 
            )

            print("[GridSearch] Otimizando arquitetura e Hiperparâmetros L-BFGS-B...")
            grid_search.fit(X_train_sparse, y_train_sparse)

            best_model = grid_search.best_estimator_
            
            # --- SALVANDO AS INFORMAÇÕES DESTE CIRCUITO ---
            self.models[circuito] = best_model
            
            # Convertendo os objetos matemáticos (Kernels) para String para que o JSON funcione
            safe_params = {k: str(v) for k, v in grid_search.best_params_.items()}
            # Usando str(circuito) para garantir que as chaves do JSON são válidas
            self.best_params[str(circuito)] = safe_params
            self.cv_scores[str(circuito)] = -grid_search.best_score_
            # -----------------------------------------------

            print(f"Melhor MAE (CV): {-grid_search.best_score_:.3f}")
            print(f"LML Final: {best_model.log_marginal_likelihood(best_model.kernel_.theta):.2f}")

            y_pred, y_std = best_model.predict(X_test_scaled, return_std=True)

            for idx, pred, std in zip(test_indices, y_pred, y_std):
                self.predictions.append(
                    {"index": idx, "pred_quali_position": pred, "pred_std": std}
                )

        return self._evaluate_global_performance(model_tag=model_tag)

    def _evaluate_global_performance(self, model_tag=MODEL_NAME):
        print(f"\n{'=' * 50}")
        print("AVALIAÇÃO GLOBAL (TESTE 2025)")

        preds_df = pd.DataFrame(self.predictions).set_index("index").sort_index()

        y_test_real = self.test_df[TARGET_COL].loc[preds_df.index]
        y_pred = preds_df["pred_quali_position"]

        rmse = np.sqrt(mean_squared_error(y_test_real, y_pred))
        mae = mean_absolute_error(y_test_real, y_pred)
        r2 = r2_score(y_test_real, y_pred)
        rho, pval = spearmanr(y_test_real, y_pred)

        print(f"RMSE: {rmse:.3f}")
        print(f"MAE:  {mae:.3f}")
        print(f"R²:   {r2:.3f}")
        print(f"Spearman rho: {rho:.4f} (p = {pval:.4e})")

        final_df = self.test_df.copy()
        final_df["pred_quali_position"] = preds_df["pred_quali_position"]
        final_df["pred_std"] = preds_df["pred_std"]
        final_df["pred_quali_position_int"] = preds_df["pred_quali_position"].round().astype(int)

        erro_absoluto = abs(final_df["pred_quali_position_int"] - y_test_real)

        acerto_exato = (erro_absoluto == 0).mean() * 100
        acerto_margem_1 = (erro_absoluto <= 1).mean() * 100
        acerto_margem_2 = (erro_absoluto <= 2).mean() * 100
        acerto_margem_3 = (erro_absoluto <= 3).mean() * 100

        print(f"\nMÉTRICAS PERCENTUAIS DE NEGÓCIO:")
        print(f"Acerto Exato: {acerto_exato:.1f}%")
        print(f"Acerto com Margem de ±1: {acerto_margem_1:.1f}% das previsões")
        print(f"Acerto com Margem de ±2: {acerto_margem_2:.1f}% das previsões")
        print(f"Acerto com Margem de ±3: {acerto_margem_3:.1f}% das previsões")

        out_file = OUT_PREDS / f"{model_tag.lower()}_predictions.csv"
        final_df.to_csv(out_file, index=False)
        print(f"Predições completas salvas em: {out_file}")

        # --- EXPORTANDO O MODELO E OS PARÂMETROS ---
        # Como treinamos vários modelos, tiramos a média do erro da Validação Cruzada (CV)
        avg_cv_mae = float(np.mean(list(self.cv_scores.values()))) if self.cv_scores else 0.0
        
        save_model(
            self.models, 
            model_tag, 
            best_params=self.best_params, 
            cv_mae=avg_cv_mae
        )
        print(f"Modelo salvo! (Múltiplos GPRs agregados sob {model_tag})")
        # -------------------------------------------

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


def main(ablation: str = "full"):
    if ablation not in ABLATIONS:
        raise ValueError(
            f"Unknown ablation '{ablation}'. Available: {list(ABLATIONS.keys())}"
        )

    model_tag = MODEL_NAME if ablation == "full" else f"{MODEL_NAME}_{ablation}"
    print(f"\n{'=' * 55}\n  {model_tag} pipeline\n{'=' * 55}")

    X_train, X_test, y_train, y_test, train_df, test_df = load_and_split(
        extra_drop_cols=ABLATIONS[ablation]
    )

    pipeline = F1GaussianProcessPipeline(X_train, X_test, y_train, train_df, test_df)
    return pipeline.run_pipeline(model_tag=model_tag)


if __name__ == "__main__":
    main()