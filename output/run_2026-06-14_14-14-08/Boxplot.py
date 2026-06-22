import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

csv_path = "/Users/tomiris/PycharmProjects/fully-inductive-cardinality-estimation/output/run_2026-06-14_14-14-08/logs/predictions.csv"
df = pd.read_csv(csv_path)

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
plt.savefig('q_error_boxplot.png')
plt.show()