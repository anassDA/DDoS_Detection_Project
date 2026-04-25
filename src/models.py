"""
Module des modèles ML
KNN, Decision Tree, SVM
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import time


class ModelTrainer:
    """Classe pour l'entraînement des modèles"""
    
    def __init__(self):
        self.models = {}
        self.best_params = {}
        self.training_times = {}
        self.prediction_times = {}
    
    def train_knn(self, X_train, y_train, X_val, y_val, k_values=None):
        """Entraîne et optimise le modèle KNN"""
        if k_values is None:
            k_values = [3, 5, 7, 9, 11, 15, 21]
        
        print("\n🟦 K-NEAREST NEIGHBORS (KNN)")
        print(f"Test k values: {k_values}")
        
        best_score = 0
        best_k = k_values[0]
        
        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            score = knn.score(X_val, y_val)
            print(f"  k={k:2d} -> Val Accuracy: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_k = k
        
        # Entraîner le meilleur modèle
        print(f"\n✓ Meilleur k: {best_k}")
        self.models['KNN'] = KNeighborsClassifier(n_neighbors=best_k)
        
        start = time.time()
        self.models['KNN'].fit(X_train, y_train)
        self.training_times['KNN'] = time.time() - start
        
        self.best_params['KNN'] = {'n_neighbors': best_k}
        print(f"✓ Temps d'entraînement: {self.training_times['KNN']:.4f}s")
        
        return self.models['KNN']
    
    def train_decision_tree(self, X_train, y_train, X_val, y_val):
        """Entraîne et optimise le Decision Tree"""
        print("\n🟩 DECISION TREE")
        
        # Grid Search
        param_grid = {
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        dt = DecisionTreeClassifier(random_state=42)
        
        # Simple validation avec gridsearch
        best_score = 0
        best_params = {}
        
        for max_depth in param_grid['max_depth']:
            for min_split in param_grid['min_samples_split']:
                for min_leaf in param_grid['min_samples_leaf']:
                    dt = DecisionTreeClassifier(
                        max_depth=max_depth,
                        min_samples_split=min_split,
                        min_samples_leaf=min_leaf,
                        random_state=42
                    )
                    dt.fit(X_train, y_train)
                    score = dt.score(X_val, y_val)
                    
                    if score > best_score:
                        best_score = score
                        best_params = {
                            'max_depth': max_depth,
                            'min_samples_split': min_split,
                            'min_samples_leaf': min_leaf
                        }
        
        print(f"✓ Meilleurs paramètres: {best_params}")
        
        self.models['Decision Tree'] = DecisionTreeClassifier(
            random_state=42,
            **best_params
        )
        
        start = time.time()
        self.models['Decision Tree'].fit(X_train, y_train)
        self.training_times['Decision Tree'] = time.time() - start
        
        self.best_params['Decision Tree'] = best_params
        print(f"✓ Temps d'entraînement: {self.training_times['Decision Tree']:.4f}s")
        
        return self.models['Decision Tree']
    
    def train_svm(self, X_train, y_train, X_val, y_val):
        """Entraîne et optimise SVM"""
        print("\n🟪 SUPPORT VECTOR MACHINE (SVM)")
        
        kernels = ['linear', 'rbf', 'poly']
        Cs = [0.1, 1, 10, 100]
        
        print(f"Test kernels: {kernels}")
        print(f"Test C values: {Cs}")
        
        best_score = 0
        best_kernel = 'linear'
        best_c = 1
        
        for kernel in kernels:
            for c in Cs:
                svm = SVC(kernel=kernel, C=c, random_state=42)
                try:
                    svm.fit(X_train, y_train)
                    score = svm.score(X_val, y_val)
                    print(f"  kernel={kernel:5s}, C={c:6.1f} -> Val Accuracy: {score:.4f}")
                    
                    if score > best_score:
                        best_score = score
                        best_kernel = kernel
                        best_c = c
                except Exception as e:
                    print(f"  kernel={kernel:5s}, C={c:6.1f} -> Erreur: {str(e)[:30]}")
        
        print(f"\n✓ Meilleur kernel: {best_kernel}, C: {best_c}")
        
        self.models['SVM'] = SVC(kernel=best_kernel, C=best_c, random_state=42, probability=True)
        
        start = time.time()
        self.models['SVM'].fit(X_train, y_train)
        self.training_times['SVM'] = time.time() - start
        
        self.best_params['SVM'] = {'kernel': best_kernel, 'C': best_c}
        print(f"✓ Temps d'entraînement: {self.training_times['SVM']:.4f}s")
        
        return self.models['SVM']
    
    def predict_with_timing(self, model_name, X_test):
        """Prédit avec mesure du temps"""
        start = time.time()
        predictions = self.models[model_name].predict(X_test)
        self.prediction_times[model_name] = time.time() - start
        
        return predictions
    
    def get_summary(self):
        """Résumé des modèles"""
        summary = {
            'Training Times': self.training_times,
            'Prediction Times': self.prediction_times,
            'Best Parameters': self.best_params
        }
        return summary
