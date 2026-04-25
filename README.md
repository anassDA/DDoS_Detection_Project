# 🔍 DDoS Detection in IoT Environments

## Supervised ML Project - 4IIR | December 2025

**Comparing KNN, Decision Tree, and SVM for DDoS Attack Detection**

---

## 📋 Project Overview

This project implements and compares three supervised classification algorithms for detecting DDoS attacks in IoT networks:

- **K-Nearest Neighbors (KNN)** - Instance-based, test different k values
- **Decision Tree (DT)** - Interpretable, hyperparameter tuning
- **Support Vector Machine (SVM)** - Test different kernels (linear, RBF, poly)

---

## 📁 Project Structure

```
DDoS_Detection_Project/
├── data/
│   └── ton_iot_sample.csv         # Sample dataset (10K records, balanced)
├── src/                           # Modular code
│   ├── preprocessing.py           # DataPreprocessor class
│   ├── models.py                  # ModelTrainer class (3 algorithms)
│   ├── evaluation.py              # ModelEvaluator class (metrics & plots)
│   └── utils.py                   # Utility functions
├── notebooks/
│   └── DDoS_Detection_4IIR.ipynb  # Main notebook (clean & ready to run)
├── results/                       # Generated outputs
│   ├── confusion_matrices.png
│   ├── metrics_comparison.png
│   └── model_comparison.xlsx
├── requirements.txt               # Python packages
└── README.md                      # This file
```

---

## ⚡ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Notebook
```bash
jupyter notebook notebooks/DDoS_Detection_4IIR.ipynb
```

The notebook is **ready to execute** - it uses the included `ton_iot_sample.csv` dataset.

---

## � What's Included

### ✓ Complete Python Modules (src/)
- **preprocessing.py** - Data cleaning, encoding, standardization, stratified split (70/15/15)
- **models.py** - Train 3 models with hyperparameter optimization
- **evaluation.py** - Comprehensive metrics, confusion matrices, visualizations
- **utils.py** - Helper functions for analysis and visualization

### ✓ Sample Dataset (data/)
- **ton_iot_sample.csv** - 10,000 balanced samples (5K normal, 5K DDoS)
- 25 features + 1 target column
- Real-world realistic distribution

### ✓ Ready-to-Run Notebook (notebooks/)
- **DDoS_Detection_4IIR.ipynb** - Clean, modular, well-commented
- No configuration needed - uses sample dataset
- Generates visualizations and Excel results

---

## �🔧 Module de Prétraitement (`preprocessing.py`)

**Classe: `DataPreprocessor`**

Étapes complètes:
1. **Chargement** - CSV → DataFrame
2. **Nettoyage** - Valeurs manquantes
3. **Encodage** - Variables catégorielles
4. **Standardisation** - Z-score normalization
5. **Split** - Train/Val/Test (70/15/15) stratifié
6. **Analyse** - Distribution des classes, déséquilibre

```python
from preprocessing import DataPreprocessor

preprocessor = DataPreprocessor()
X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess(
    df, 
    target_col='attack',
    categorical_cols=['protocol', 'service']
)
```

---

## 🤖 Module des Modèles (`models.py`)

**Classe: `ModelTrainer`**

**1. KNN**
- Teste plusieurs valeurs de k
- Sélectionne le meilleur k en validation
- Rapide, basé sur la distance

**2. Decision Tree**
- Grid Search sur max_depth, min_samples_split, min_samples_leaf
- Interprétable
- Pas besoin de standardisation

**3. SVM**
- Teste plusieurs kernels: linear, RBF, polynomial
- Teste plusieurs valeurs de C
- Performant sur données complexes

```python
from models import ModelTrainer

trainer = ModelTrainer()
model_knn = trainer.train_knn(X_train, y_train, X_val, y_val)
model_dt = trainer.train_decision_tree(X_train, y_train, X_val, y_val)
model_svm = trainer.train_svm(X_train, y_train, X_val, y_val)

# Temps de prédiction
pred_knn = trainer.predict_with_timing('KNN', X_test)
```

---

## 📊 Module d'Évaluation (`evaluation.py`)

**Classe: `ModelEvaluator`**

Métriques:
- **Accuracy** - Taux global
- **Precision** - Faux positifs (FP)
- **Recall** - Vrais positifs (TP) - *crucial pour DDoS*
- **F1-Score** - Équilibre Precision/Recall
- **Confusion Matrix** - TP/TN/FP/FN
- **ROC-AUC** - Courbe ROC

Visualisations:
- Matrices de confusion
- Comparaison des métriques
- Courbes ROC

```python
from evaluation import ModelEvaluator

evaluator = ModelEvaluator()
evaluator.evaluate_model('KNN', model_knn, X_test, y_test)
evaluator.evaluate_model('Decision Tree', model_dt, X_test, y_test)
evaluator.evaluate_model('SVM', model_svm, X_test, y_test)

# Comparaison
comparison = evaluator.compare_models()

# Visualisations
evaluator.plot_confusion_matrices()
evaluator.plot_metrics_comparison()
```

---

## 🎯 Analyse de Performance pour IoT

### Critères Importants

| Critère | Importance | Raison |
|---------|-----------|--------|
| **Recall (Sensitivity)** | ⭐⭐⭐ | Détecter TOUTES les attaques |
| **Temps de prédiction** | ⭐⭐⭐ | CPU/batterie limitée |
| **Mémoire** | ⭐⭐ | Stockage limité |
| **Facilité déploiement** | ⭐⭐ | Appareils hétérogènes |

### Comparaison Modèles

| Modèle | Recall | Vitesse | Interprétable | Mémoire |
|--------|--------|---------|---------------|---------|
| **KNN** | Moyen | ⚡⚡ | Faible | Élevée |
| **Decision Tree** | Bon | ⚡⚡⚡ | ✓ Excellente | Faible |
| **SVM** | Excellent | ⚡ | Faible | Moyenne |

---

## 📈 Fichiers de Résultats

Généré automatiquement dans `results/`:
- `confusion_matrices.png` - Matrices de confusion
- `metrics_comparison.png` - Comparaison des métriques
- `roc_knn.png`, `roc_svm.png` - Courbes ROC
- `model_comparison.xlsx` - Résultats détaillés (Excel)

---

## 📝 Livrables du Projet

Avant **31 Décembre 2025**, packager dans:
```
Nom1_Prenom1_Nom2_Prenom2_4IIRGnumeroDuGroupe.zip
├── Nom1_Prenom1_Nom2_Prenom2_MLProject.ipynb
├── Rapport_Nom1_Nom2.pdf (6-8 pages)
└── Presentation_Nom1_Nom2.pptx (10 slides)
```

### Contenu du Rapport

1. **Introduction** - Contexte DDoS/IoT
2. **Méthodologie** - Prétraitement, modèles
3. **Résultats** - Tableaux, figures
4. **Analyse comparative** - Quel modèle & pourquoi
5. **Conclusion** - Recommandations IoT
6. **Références** - TON_IoT dataset, scikit-learn

### Contenu de la Présentation (10 slides)

1. Titre + Problématique
2. Objectifs pédagogiques
3. Dataset TON_IoT
4. Méthodologie
5. Résultats KNN
6. Résultats Decision Tree
7. Résultats SVM
8. Comparaison
9. Recommandations
10. Conclusion + Déploiement

---

## 🔍 Exemple d'Utilisation Complète

```python
import pandas as pd
from preprocessing import DataPreprocessor
from models import ModelTrainer
from evaluation import ModelEvaluator

# 1. Charger les données
df = pd.read_csv('data/ton_iot_data.csv')

# 2. Prétraitement
preprocessor = DataPreprocessor()
X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess(
    df, target_col='attack'
)

# 3. Entraînement
trainer = ModelTrainer()
model_knn = trainer.train_knn(X_train, y_train, X_val, y_val)
model_dt = trainer.train_decision_tree(X_train, y_train, X_val, y_val)
model_svm = trainer.train_svm(X_train, y_train, X_val, y_val)

# 4. Évaluation
evaluator = ModelEvaluator()
evaluator.evaluate_model('KNN', model_knn, X_test, y_test)
evaluator.evaluate_model('Decision Tree', model_dt, X_test, y_test)
evaluator.evaluate_model('SVM', model_svm, X_test, y_test)

# 5. Comparaison
comparison = evaluator.compare_models()
evaluator.plot_metrics_comparison()
```

---

## 🛠️ Dépendances

- **pandas** - Manipulation de données
- **numpy** - Calculs numériques
- **scikit-learn** - ML algorithms
- **matplotlib** - Visualisations
- **seaborn** - Graphiques statistiques
- **jupyter** - Notebooks interactifs

Installer: `pip install -r requirements.txt`

---

## 📚 Références

- **Dataset**: [TON_IoT - UNSW](https://research.unsw.edu.au/projects/toniot-datasets)
- **Scikit-learn**: [Documentation](https://scikit-learn.org/)
- **KNN**: Guo et al. "KNN Classification Algorithm" (2016)
- **Decision Trees**: Breiman et al. "Classification and Regression Trees" (1984)
- **SVM**: Cortes & Vapnik "Support Vector Networks" (1995)

---

## 👥 Groupe

- Nom1 Prenom1
- Nom2 Prenom2

**Classe**: 4IIR | **Année**: Décembre 2025

---

## ⚖️ License

Projet académique - Université
