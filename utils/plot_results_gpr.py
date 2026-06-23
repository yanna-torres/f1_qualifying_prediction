"""
utils/plot_results_gpr.py
=============
Script para geração de gráficos analíticos e de performance 
baseados nas predições do modelo de Processo Gaussiano.
Salva os gráficos em alta resolução para documentação/dissertação.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Injeta a raiz do projeto no path para importar o config
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from config import TARGET_COL, OUT_PREDS, OUT_FIGS
except ImportError:
    # Fallback caso o config.py não esteja acessível diretamente
    TARGET_COL = 'GridPosition'
    OUT_PREDS = Path("outputs/predictions")
    OUT_FIGS = Path("outputs/figures")

# Define o caminho do arquivo de predições gerado pelo train_gpr.py
PREDS_FILE = OUT_PREDS / "gpr_predictions_2025.csv"

# Configurações estéticas globais (Padrão artigo científico)
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.dpi': 300})

class GPRVisualizer:
    def __init__(self, preds_path=PREDS_FILE):
        print(f"Carregando predições de: {preds_path}")
        if not os.path.exists(preds_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {preds_path}. Execute o train_gpr.py primeiro.")
        
        self.df = pd.read_csv(preds_path)
        
        # Garante que a pasta de imagens exista
        os.makedirs(OUT_FIGS, exist_ok=True)
        
        # Calcula o resíduo (Erro = Previsto - Real)
        self.df['residual'] = self.df['pred_quali_position'] - self.df[TARGET_COL]

    def plot_actual_vs_predicted(self):
        """Gráfico de Dispersão: Posição Real vs Posição Prevista"""
        plt.figure(figsize=(8, 8))
        
        # Plot principal
        sns.scatterplot(
            data=self.df, 
            x=TARGET_COL, 
            y='pred_quali_position', 
            alpha=0.6, 
            color='#1f77b4',
            edgecolor='k'
        )
        
        # Linha de perfeição (y = x)
        max_pos = max(self.df[TARGET_COL].max(), self.df['pred_quali_position'].max())
        plt.plot([1, max_pos], [1, max_pos], 'r--', lw=2, label='Predição Perfeita (y = x)')
        
        plt.title('Performance do Modelo: Real vs. Previsto', fontweight='bold')
        plt.xlabel('Posição Real na Qualificação (GridPosition)')
        plt.ylabel('Posição Prevista (Processo Gaussiano)')
        plt.legend()
        
        out_path = OUT_FIGS / "actual_vs_predicted.png"
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        print(f"Gráfico salvo: {out_path}")

    def plot_residuals_distribution(self):
        """Histograma dos Resíduos para verificar a normalidade do erro"""
        plt.figure(figsize=(10, 6))
        
        sns.histplot(self.df['residual'], kde=True, bins=20, color='purple', edgecolor='k')
        
        # Linha do Erro Zero
        plt.axvline(x=0, color='r', linestyle='--', lw=2, label='Erro Zero')
        
        plt.title('Distribuição dos Resíduos (Previsto - Real)', fontweight='bold')
        plt.xlabel('Erro (Posições)')
        plt.ylabel('Frequência (Nº de Voltas/Pilotos)')
        plt.legend()
        
        out_path = OUT_FIGS / "residuals_distribution.png"
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        print(f"Gráfico salvo: {out_path}")

    def plot_uncertainty_bands(self, round_number=1):
        """
        Gera um gráfico com Intervalo de Confiança (95%) para uma corrida específica.
        É a prova de que o GPR quantifica a incerteza.
        """
        # Filtra os dados apenas para uma etapa (Round) do ano
        df_race = self.df[self.df['Round'] == round_number].copy()
        
        if len(df_race) == 0:
            print(f"Nenhum dado encontrado para o Round {round_number}. Pulando gráfico de incerteza.")
            return

        # Ordena pela posição prevista para o gráfico formar uma curva suave
        df_race = df_race.sort_values('pred_quali_position').reset_index(drop=True)

        plt.figure(figsize=(12, 6))
        
        # Eixo X simulado (Índice dos pilotos ordenados)
        x = np.arange(len(df_race))
        
        # Variáveis
        y_real = df_race[TARGET_COL]
        y_pred = df_race['pred_quali_position']
        y_std = df_race['pred_std']
        
        # Intervalo de 95% de confiança (Z = 1.96)
        lower_bound = y_pred - 1.96 * y_std
        upper_bound = y_pred + 1.96 * y_std

        # Plota a incerteza (Faixa Sombreada)
        plt.fill_between(x, lower_bound, upper_bound, color='#1f77b4', alpha=0.2, label='95% Intervalo de Confiança')
        
        # Plota a predição (Linha/Pontos)
        plt.plot(x, y_pred, 'b-', label='Predição do GPR', lw=2)
        plt.scatter(x, y_pred, color='blue', s=30)
        
        # Plota os valores reais
        plt.scatter(x, y_real, color='red', s=50, zorder=5, label='Posição Real')

        # Adiciona o nome do piloto no eixo X (se a coluna existir)
        if 'Driver' in df_race.columns:
            # Como Driver pode estar em LabelEncoder (inteiro), tentamos usá-lo ou usar o índice
            labels = df_race['Driver'].astype(str)
            plt.xticks(x, labels, rotation=45, ha='right')
            plt.xlabel('Piloto (ID Oculto/Label)')
        else:
            plt.xlabel('Pilotos (Ordenados pela Predição)')

        plt.title(f'Predição com Incerteza Gaussiana - Etapa {round_number} (Teste 2025)', fontweight='bold')
        plt.ylabel('Posição no Grid')
        plt.legend(loc='upper left')
        
        out_path = OUT_FIGS / f"uncertainty_round_{round_number}.png"
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        print(f"Gráfico salvo: {out_path}")

    def generate_all_plots(self):
        """Orquestra a geração de todos os gráficos."""
        print(f"{'='*50}")
        print("GERANDO GRÁFICOS DE AVALIAÇÃO...")
        
        self.plot_actual_vs_predicted()
        self.plot_residuals_distribution()
        
        # Tenta pegar o primeiro round disponível para o gráfico de incerteza
        if 'Round' in self.df.columns:
            primeiro_round = self.df['Round'].iloc[0]
            self.plot_uncertainty_bands(round_number=primeiro_round)
        else:
            print("Coluna 'Round' não encontrada. Pulando o gráfico de incerteza específico da corrida.")
            
        print("\nTodos os gráficos foram exportados com sucesso para a pasta 'outputs/figures/'.")

if __name__ == "__main__":
    visualizer = GPRVisualizer()
    visualizer.generate_all_plots()