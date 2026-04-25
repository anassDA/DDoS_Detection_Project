"""
Module de prétraitement des données DDoS IoT
Gère: nettoyage, encodage, standardisation, équilibrage
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Classe pour le prétraitement des données"""
    
    def __init__(self, test_size=0.15, val_size=0.15, random_state=42):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def load_data(self, filepath):
        """Charge les données depuis un fichier CSV"""
        df = pd.read_csv(filepath)
        print(f"✓ Données chargées: {df.shape}")
        return df
    
    def explore_data(self, df):
        """Affiche les statistiques de base"""
        print("\n📊 EXPLORATION DES DONNÉES")
        print(f"Shape: {df.shape}")
        print(f"\nColonnes: {df.columns.tolist()}")
        print(f"\nValeurs manquantes:\n{df.isnull().sum()}")
        print(f"\nTypes de données:\n{df.dtypes}")
        print(f"\nStatistiques:\n{df.describe()}")
        
        return df.head()
    
    def handle_missing_values(self, df, strategy='drop'):
        """Gère les valeurs manquantes"""
        if df.isnull().sum().sum() == 0:
            print("✓ Pas de valeurs manquantes")
            return df
        
        if strategy == 'drop':
            df = df.dropna()
            print(f"✓ Valeurs manquantes supprimées")
        elif strategy == 'mean':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            print(f"✓ Valeurs manquantes remplies (moyenne)")
        
        return df
    
    def encode_categorical(self, df, categorical_cols=None):
        """Encode les variables catégorielles"""
        if categorical_cols is None:
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if not categorical_cols:
            print("✓ Pas de colonnes catégorielles")
            return df
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col])
            else:
                df[col] = self.label_encoders[col].transform(df[col])
        
        print(f"✓ {len(categorical_cols)} colonnes catégorielles encodées")
        return df
    
    def standardize_data(self, X_train, X_val, X_test):
        """Standardise les features (z-score)"""
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        
        print("✓ Données standardisées (z-score)")
        return X_train, X_val, X_test
    
    def split_data(self, X, y):
        """Split en train/val/test (70/15/15)"""
        # Train/Test (70/30)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=(self.test_size + self.val_size), 
            random_state=self.random_state, stratify=y
        )
        
        # Val/Test (50/50 du 30% restant)
        X_val, X_test, y_val, y_test = train_test_split(
            X_test, y_test, test_size=0.5,
            random_state=self.random_state, stratify=y_test
        )
        
        print(f"✓ Split stratifié:")
        print(f"  Train: {X_train.shape[0]} samples")
        print(f"  Val:   {X_val.shape[0]} samples")
        print(f"  Test:  {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def preprocess(self, df, target_col, categorical_cols=None):
        """Pipeline complet de prétraitement"""
        print("\n🔧 PRÉTRAITEMENT")
        
        # 1. Nettoyage
        df = self.handle_missing_values(df)
        
        # 2. Séparation features/target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # 3. Encodage catégories
        X = self.encode_categorical(X, categorical_cols)
        
        # 4. Encodage target
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
            print(f"✓ Target encodée: {le.classes_}")
        
        # 5. Analyse déséquilibre
        unique, counts = np.unique(y, return_counts=True)
        print(f"\n⚠️  Distribution des classes:")
        for label, count in zip(unique, counts):
            pct = 100 * count / len(y)
            print(f"  Classe {label}: {count} ({pct:.1f}%)")
        
        # 6. Split stratifié
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # 7. Standardisation
        X_train, X_val, X_test = self.standardize_data(X_train, X_val, X_test)
        
        self.feature_names = X.columns.tolist()
        
        return X_train, X_val, X_test, y_train, y_val, y_test
