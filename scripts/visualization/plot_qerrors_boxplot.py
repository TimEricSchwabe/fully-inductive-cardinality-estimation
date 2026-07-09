import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Plot Q-Error boxplots from predictions CSV.")
parser.add_argument("--csv_path", type=str, required=True, help="Path to predictions.csv")
parser.add_argument("--output_path", type=str, default="q_error_boxplot.png", help="Path to save the output plot")
args = parser.parse_args()

df = pd.read_csv(args.csv_path)

if 'stage' in df.columns:
    df = df[df['stage'] == 'test'].copy()

if 'q_error' not in df.columns:
    preds = np.clip(df['prediction'], a_min=1.0, a_max=None)
    trues = np.clip(df['true_value'], a_min=1.0, a_max=None)
    df['q_error'] = np.maximum(trues / preds, preds / trues)

plt.figure(figsize=(8, 5))
ax = sns.boxplot(data=df, x='query_type', y='q_error', palette='Set2')
ax.set_yscale('log')

plt.title('Distribution of Q-Errors by Query Type (Test Stage)')
plt.ylabel('Q-Error (Log Scale)')
plt.xlabel('Query Category')
plt.grid(True, which="both", ls="--", alpha=0.3)

plt.tight_layout()
plt.savefig(args.output_path)
print(f"Saved boxplot to {args.output_path}")
# plt.show() # Uncomment to display in an interactive window
