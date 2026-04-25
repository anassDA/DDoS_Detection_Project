"""
Script pour créer un dataset sample
"""
import numpy as np
import pandas as pd
import os

# Créer le dossier data
os.makedirs('data', exist_ok=True)

# Générer dataset sample
np.random.seed(42)
n_normal = 4000
n_attack = 1000

# Features normales
normal_traffic = np.random.normal(loc=50, scale=15, size=(n_normal, 15))
normal_data = pd.DataFrame(normal_traffic, columns=[f'feature_{i}' for i in range(15)])
normal_data['attack'] = 0

# Attaques DDoS
attack_traffic = np.random.normal(loc=120, scale=40, size=(n_attack, 15))
attack_data = pd.DataFrame(attack_traffic, columns=[f'feature_{i}' for i in range(15)])
attack_data['attack'] = 1

# Combiner et mélanger
df = pd.concat([normal_data, attack_data], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Sauvegarder
df.to_csv('data/ton_iot_sample.csv', index=False)

print('✓ Dataset sample créé: data/ton_iot_sample.csv')
print(f'  Shape: {df.shape}')
print(f'  Classes: Normal={sum(df["attack"]==0)}, DDoS={sum(df["attack"]==1)}')
