"""
Module d'évaluation des modèles
Métriques, matrices de confusion, comparaisons
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, auc, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """Classe pour l'évaluation des modèles"""
    
    def __init__(self):
        self.results = {}
        self.predictions = {}
    
    def evaluate_model(self, model_name, model, X_test, y_test, X_val=None, y_val=None):
        """Évalue complètement un modèle"""
        print(f"\n{'='*60}")
        print(f"ÉVALUATION: {model_name}")
        print(f"{'='*60}")
        
        # Prédictions
        y_pred = model.predict(X_test)
        self.predictions[model_name] = y_pred
        
        # Métriques
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        print(f"\n📊 MÉTRIQUES (Test Set):")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        
        # ROC-AUC si possible
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, 'decision_function'):
                y_proba = model.decision_function(X_test)
            else:
                y_proba = None
            
            if y_proba is not None:
                roc_auc = roc_auc_score(y_test, y_proba)
                print(f"  ROC-AUC:   {roc_auc:.4f}")
        except:
            pass
        
        # Rapport détaillé
        print(f"\n📋 RAPPORT DÉTAILLÉ:")
        print(classification_report(y_test, y_pred))
        
        # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n🔍 MATRICE DE CONFUSION:")
        print(f"  TN: {cm[0,0]:5d}  FP: {cm[0,1]:5d}")
        print(f"  FN: {cm[1,0]:5d}  TP: {cm[1,1]:5d}")
        
        # Validation metrics si disponible
        if X_val is not None and y_val is not None:
            y_val_pred = model.predict(X_val)
            val_accuracy = accuracy_score(y_val, y_val_pred)
            print(f"\n🔄 VALIDATION SET:")
            print(f"  Accuracy: {val_accuracy:.4f}")
        
        # Stocker les résultats
        self.results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'y_pred': y_pred
        }
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'cm': cm
        }
    
    def compare_models(self):
        """Compare tous les modèles"""
        print(f"\n{'='*60}")
        print("COMPARAISON DES MODÈLES")
        print(f"{'='*60}\n")
        
        df_comparison = pd.DataFrame({
            model: {
                'Accuracy': self.results[model]['accuracy'],
                'Precision': self.results[model]['precision'],
                'Recall': self.results[model]['recall'],
                'F1-Score': self.results[model]['f1']
            }
            for model in self.results.keys()
        }).T
        
        print(df_comparison.round(4))
        
        # Meilleur modèle pour chaque métrique
        print(f"\n🏆 MEILLEUR MODÈLE PAR MÉTRIQUE:")
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
            best_model = df_comparison[metric].idxmax()
            best_value = df_comparison[metric].max()
            print(f"  {metric:10s}: {best_model:20s} ({best_value:.4f})")
        
        return df_comparison
    
    def plot_confusion_matrices(self, save_path=None):
        """Affiche les matrices de confusion"""
        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, results) in enumerate(self.results.items()):
            cm = results['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       ax=axes[idx], cbar=False,
                       xticklabels=['Normal', 'DDoS'],
                       yticklabels=['Normal', 'DDoS'])
            axes[idx].set_title(f"{model_name}\n(Accuracy: {results['accuracy']:.4f})")
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_metrics_comparison(self, save_path=None):
        """Compare les métriques des modèles"""
        df_results = pd.DataFrame(self.results).T[['accuracy', 'precision', 'recall', 'f1']]
        df_results.columns = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        df_results.plot(kind='bar', ax=ax, width=0.8)
        
        ax.set_title('Comparaison des Métriques par Modèle', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12)
        ax.set_xlabel('Modèle', fontsize=12)
        ax.set_ylim([0, 1.05])
        ax.legend(loc='lower right', framealpha=0.9)
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45)
        
        # Ajouter les valeurs
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=2)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curves(self, model, model_name, X_test, y_test, save_path=None):
        """Affiche la courbe ROC"""
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, 'decision_function'):
                y_proba = model.decision_function(X_test)
            else:
                print(f"⚠️  {model_name} ne supporte pas les probabilités")
                return
            
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
            plt.legend(loc="lower right", fontsize=11)
            plt.grid(alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f"⚠️  Erreur ROC: {e}")
    
    def export_results(self, filepath):
        """Exporte les résultats en Excel"""
        df_results = pd.DataFrame(self.results).T[['accuracy', 'precision', 'recall', 'f1']]
        df_results.to_excel(filepath)
        print(f"✓ Résultats exportés: {filepath}")
