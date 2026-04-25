"""
Fonctions utilitaires
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def download_ton_iot_dataset():
    """
    Guide pour télécharger le dataset TON_IoT
    URL: https://research.unsw.edu.au/projects/toniot-datasets
    """
    print("""
    📥 TÉLÉCHARGEMENT DU DATASET TON_IoT
    
    1. Aller à: https://research.unsw.edu.au/projects/toniot-datasets
    2. Télécharger les fichiers CSV (flux normal + attaques DDoS)
    3. Placer les fichiers dans le dossier: data/
    
    Exemple de structure:
    - data/ToN_IoT_N_BLE.csv          (Normal BLE)
    - data/ToN_IoT_A_DDoS.csv         (Attaques DDoS)
    """)


def create_sample_dataset(n_samples=10000):
    """
    Crée un dataset sample pour tester le pipeline
    Si le vrai dataset n'est pas disponible
    """
    print(f"📊 Création d'un dataset sample ({n_samples} samples)...")
    
    np.random.seed(42)
    
    # Features normales
    normal_data = np.random.normal(loc=50, scale=15, size=(n_samples//2, 20))
    normal_labels = np.zeros(n_samples//2)
    
    # Features attaques DDoS (pattern différent)
    attack_data = np.random.normal(loc=150, scale=30, size=(n_samples//2, 20))
    attack_labels = np.ones(n_samples//2)
    
    # Combiner
    X = np.vstack([normal_data, attack_data])
    y = np.hstack([normal_labels, attack_labels])
    
    # Shuffle
    indices = np.random.permutation(len(y))
    X = X[indices]
    y = y[indices]
    
    # DataFrame
    columns = [f'feature_{i}' for i in range(20)]
    df = pd.DataFrame(X, columns=columns)
    df['attack'] = y.astype(int)
    
    print(f"✓ Dataset créé: {df.shape}")
    return df


def analyze_class_distribution(y, title="Distribution des Classes"):
    """Analyse la distribution des classes"""
    unique, counts = np.unique(y, return_counts=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Pie chart
    colors = ['green', 'red']
    ax1.pie(counts, labels=['Normal', 'DDoS'], autopct='%1.1f%%',
    colors=colors, startangle=90)
    ax1.set_title(title)
    
    # Bar chart
    ax2.bar(['Normal', 'DDoS'], counts, color=colors, alpha=0.7)
    ax2.set_ylabel('Count')
    ax2.set_title(f'Distribution (Total: {len(y)} samples)')
    ax2.grid(axis='y', alpha=0.3)
    
    # Ajouter les nombres
    for i, count in enumerate(counts):
        ax2.text(i, count + 100, str(count), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def analyze_features(df, target_col='attack', top_n=10):
    """Analyse les features importantes"""
    print(f"\n📈 ANALYSE DES FEATURES")
    print(f"{'='*60}")
    
    # Correlation avec la target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    correlations = X.corrwith(y).abs().sort_values(ascending=False)
    print(f"\n🔗 Top {top_n} Features (Corrélation avec target):")
    print(correlations.head(top_n))
    
    # Visualisation
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top 4 correlations
    for idx, (feature, corr) in enumerate(correlations.head(4).items()):
        ax = axes[idx//2, idx%2]
        
        # Box plot
        df_normal = df[df[target_col] == 0][feature]
        df_attack = df[df[target_col] == 1][feature]
        
        bp = ax.boxplot([df_normal, df_attack], labels=['Normal', 'DDoS'], patch_artist=True)
        for patch, color in zip(bp['boxes'], ['green', 'red']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title(f'{feature}\n(Correlation: {corr:.3f})', fontweight='bold')
        ax.set_ylabel('Value')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return correlations


def print_project_structure():
    """Affiche la structure du projet"""
    structure = """
    📁 DDoS_Detection_Project/
    ├── 📂 data/                    # Données brutes
    ├── 📂 src/                     # Code source
    │   ├── preprocessing.py        # Prétraitement
    │   ├── models.py              # Modèles ML
    │   ├── evaluation.py          # Évaluation
    │   └── utils.py               # Utilitaires
    ├── 📂 notebooks/              # Jupyter notebooks
    │   └── DDoS_Detection_4IIR.ipynb
    ├── 📂 results/                # Résultats & figures
    ├── requirements.txt           # Dépendances Python
    └── README.md                  # Documentation
    """
    print(structure)


def get_system_info():
    """Affiche les infos système"""
    import platform
    import sklearn
    
    print("\n🖥️  INFORMATIONS SYSTÈME")
    print(f"{'='*60}")
    print(f"Python:      {platform.python_version()}")
    print(f"Platform:    {platform.system()}")
    print(f"Scikit-learn: {sklearn.__version__}")
    print(f"{'='*60}\n")
