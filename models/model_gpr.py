"""
train_gpr.py
=============
Pipeline de Treinamento Avançado utilizando Processos Gaussianos (GPR).
Implementa Modelos Particionados (por Circuito), One-Hot Encoding seguro,
Redução de Dimensionalidade (K-Means / Inducing Points) e GridSearchCV.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min, mean_absolute_error, mean_squared_error, r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from sklearn.model_selection import GridSearchCV

# Injeta a raiz do projeto no path para importar o config
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from config import TARGET_COL, OUT_PREDS, OUT_RESULTS
except ImportError:
    # Fallback caso o config.py não esteja acessível diretamente
    TARGET_COL = 'quali_position'
    OUT_PREDS = Path("outputs/predictions")
    OUT_RESULTS = Path("outputs/results_table.csv")

# Constantes do Modelo
TRAIN_PATH = "data/qualifying_dataset_train.csv"
TEST_PATH = "data/qualifying_dataset_test.csv"
MAX_INDUCING_POINTS = 300  # Limite máximo de amostras por circuito (Otimização O(n³))

# Colunas que vieram com LabelEncoder do data.py e precisam virar binárias (OHE)
CATEGORICAL_COLS = ['Driver', 'Team', 'SubSession', 'q1_compound', 'q2_compound', 'q3_compound']

class F1GaussianProcessPipeline:
    def __init__(self, train_path=TRAIN_PATH, test_path=TEST_PATH):
        print("Inicializando Pipeline GPR...")
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        self.models = {}  # Dicionário para guardar o melhor modelo de cada circuito
        self.predictions = [] # Lista para reconstruir as predições do teste

        # Garante que os diretórios de saída existam
        os.makedirs(OUT_PREDS, exist_ok=True)
        os.makedirs(OUT_RESULTS.parent, exist_ok=True)

    def _build_preprocessor(self, X_sample):
        """
        Cria o pipeline de pré-processamento.
        Aplica OHE nas categóricas e StandardScaler nas numéricas.
        """
        # Identifica colunas presentes no dataset que precisam de OHE
        cat_cols = [c for c in CATEGORICAL_COLS if c in X_sample.columns]
        
        # O resto é considerado numérico (telemetria, clima, etc)
        num_cols = [c for c in X_sample.columns if c not in cat_cols]

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_cols),
                ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols)
            ]
        )
        return preprocessor

    def _extract_sparse_inducing_points(self, X, y, k_clusters):
        """
        Aplica K-Means para encontrar amostras representativas (Centroides reais),
        evitando redundância e destruindo o gargalo O(n³).
        """
        # Se temos menos linhas que o limite, não precisamos de K-Means
        if len(X) <= k_clusters:
            return X, y

        print(f"[K-Means] Comprimindo {len(X)} amostras para {k_clusters} pontos indutores...")
        kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init='auto')
        kmeans.fit(X)

        # Encontra o índice da linha real mais próxima de cada centroide matemático
        indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
        
        return X[indices], y.iloc[indices].values

    def run_pipeline(self):
        """Orquestra o treinamento particionado por circuito."""
        circuitos = self.train_df['Circuit'].unique()
        print(f"Encontrados {len(circuitos)} circuitos para particionamento.\n")

        for circuito in circuitos:
            print(f"{'='*50}")
            print(f"TREINANDO CIRCUITO ID: {circuito}")
            
            # 1. Isolar dados do circuito
            mask_train = self.train_df['Circuit'] == circuito
            mask_test = self.test_df['Circuit'] == circuito

            train_circ = self.train_df[mask_train].copy()
            test_circ = self.test_df[mask_test].copy()

            if len(test_circ) == 0:
                print("Nenhum dado de teste para este circuito. Pulando predição...")
                continue

            # 2. Separar Features (X) e Target (y), removendo a coluna 'Circuit'
            X_train_raw = train_circ.drop(columns=[TARGET_COL, 'Circuit'])
            y_train_raw = train_circ[TARGET_COL]
            
            X_test_raw = test_circ.drop(columns=[TARGET_COL, 'Circuit'])
            test_indices = test_circ.index # Guarda os índices originais para reconstrução

            # 3. Pré-processamento Anti-Leakage (Fit apenas no Treino!)
            preprocessor = self._build_preprocessor(X_train_raw)
            X_train_scaled = preprocessor.fit_transform(X_train_raw)
            X_test_scaled = preprocessor.transform(X_test_raw)

            num_features = X_train_scaled.shape[1]
            print(f"Features após One-Hot Encoding: {num_features}")

            # 4. K-Means: Extração de Pontos Indutores
            X_train_sparse, y_train_sparse = self._extract_sparse_inducing_points(
                X_train_scaled, y_train_raw, k_clusters=MAX_INDUCING_POINTS
            )

            # 5. Configuração do GridSearch e Kernels ARD
            initial_length_scales = np.ones(num_features)
            bounds = (1e-2, 1e4)

            # Definimos as arquiteturas candidatas
            # Otimizamos os limites do ruído branco para (1e-10, 1e1) para evitar o ConvergenceWarning
            kernel_matern = ConstantKernel(1.0) * Matern(length_scale=initial_length_scales, length_scale_bounds=bounds, nu=2.5) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-10, 1e1))
            kernel_rbf = ConstantKernel(1.0) * RBF(length_scale=initial_length_scales, length_scale_bounds=bounds) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-10, 1e1))
            param_grid = {
                'kernel': [kernel_matern, kernel_rbf],
                'alpha': [1e-5, 1e-2] # Diferentes níveis de jitter
            }

            gpr_base = GaussianProcessRegressor(
                n_restarts_optimizer=5, 
                normalize_y=True, 
                random_state=42
            )

            # CV=3 garante validação cruzada robusta no GridSearch
            grid_search = GridSearchCV(
                estimator=gpr_base,
                param_grid=param_grid,
                cv=3,
                scoring='neg_mean_absolute_error',
                n_jobs=-1 # Usa todos os núcleos da CPU
            )

            # 6. Treinamento Otimizado
            print("[GridSearch] Otimizando arquitetura e Hiperparâmetros L-BFGS-B...")
            grid_search.fit(X_train_sparse, y_train_sparse)
            
            best_model = grid_search.best_estimator_
            self.models[circuito] = best_model
            
            print(f"Melhor MAE (CV): {-grid_search.best_score_:.3f}")
            print(f"LML Final: {best_model.log_marginal_likelihood(best_model.kernel_.theta):.2f}")

            # 7. Inferência no conjunto de Teste do circuito
            y_pred, y_std = best_model.predict(X_test_scaled, return_std=True)

            for idx, pred, std in zip(test_indices, y_pred, y_std):
                self.predictions.append({
                    'index': idx,
                    'pred_quali_position': pred,
                    'pred_std': std
                })

        # 8. Reconstrução e Avaliação Global
        self._evaluate_global_performance()

    def _evaluate_global_performance(self):
        """Reconstroi as predições na ordem original e calcula métricas do ano de teste."""
        print(f"\n{'='*50}")
        print("AVALIAÇÃO GLOBAL (TESTE 2025)")

        # Transforma predições em DataFrame e ordena pelo index original
        preds_df = pd.DataFrame(self.predictions).set_index('index').sort_index()
        
        # Pega o alvo real (y_test) original e alinha os índices
        y_test_real = self.test_df[TARGET_COL].loc[preds_df.index]

        # Calcula as métricas
        rmse = np.sqrt(mean_squared_error(y_test_real, preds_df['pred_quali_position']))
        mae = mean_absolute_error(y_test_real, preds_df['pred_quali_position'])
        r2 = r2_score(y_test_real, preds_df['pred_quali_position'])

        print(f"RMSE: {rmse:.3f}")
        print(f"MAE:  {mae:.3f}")
        print(f"R²:   {r2:.3f}")

        # Salva resultados
        final_df = self.test_df.copy()
        final_df['pred_quali_position'] = preds_df['pred_quali_position']
        final_df['pred_std'] = preds_df['pred_std']
        # Salva resultados originais (floats)
        final_df = self.test_df.copy()
        final_df['pred_quali_position'] = preds_df['pred_quali_position']
        final_df['pred_std'] = preds_df['pred_std']
        
        # Cria uma nova coluna com o valor arredondado para inteiro (int)
        final_df['pred_quali_position_int'] = preds_df['pred_quali_position'].round().astype(int)

        # 1. Calcula a diferença absoluta entre o previsto (arredondado) e o real
        erro_absoluto = abs(final_df['pred_quali_position_int'] - y_test_real)

        # 2. Transforma em percentagens
        acerto_exato = (erro_absoluto == 0).mean() * 100
        acerto_margem_1 = (erro_absoluto <= 1).mean() * 100
        acerto_margem_2 = (erro_absoluto <= 2).mean() * 100

        print(f"\nMÉTRICAS PERCENTUAIS DE NEGÓCIO:")
        print(f"Acerto Exato: {acerto_exato:.1f}%")
        print(f"Acerto com Margem de ±1: {acerto_margem_1:.1f}% das previsões")
        print(f"Acerto com Margem de ±2: {acerto_margem_2:.1f}% das previsões")
        
        out_file = OUT_PREDS / "gpr_predictions_2025.csv"
        final_df.to_csv(out_file, index=False)
        print(f"Predições completas salvas em: {out_file}")

if __name__ == "__main__":
    pipeline = F1GaussianProcessPipeline()
    pipeline.run_pipeline()