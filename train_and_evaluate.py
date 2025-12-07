#!/usr/bin/env python3
"""
Model Training and Evaluation Script
Trains Random Forest and XGBoost models with proper evaluation metrics.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.recommendation_engine import EngagementRecommender
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("="*70)
    print("ML ENGAGEMENT RECOMMENDER - TRAINING & EVALUATION")
    print("="*70)
    
    # Initialize
    recommender = EngagementRecommender(model_type='both')
    
    # Load data
    print("\n📊 Loading Data...")
    data = recommender.load_data('data/sample_data/engagement_history.csv')
    
    # Train models
    print("\n🤖 Training Models...")
    results = recommender.train_models(data, use_pca=False)
    
    # Save models
    print("\n💾 Saving Models...")
    recommender.save_models('models/trained')
    
    # Generate evaluation report
    print("\n📈 Generating Evaluation Report...")
    report = {
        'model_performance': {},
        'feature_importance': {},
        'recommendations': []
    }
    
    for model_name, metrics in results.items():
        report['model_performance'][model_name] = {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1_score': float(metrics['f1']),
            'roc_auc': float(metrics['roc_auc'])
        }
        
        # Top 10 features
        feat_imp = sorted(metrics['feature_importance'].items(), 
                         key=lambda x: x[1], reverse=True)[:10]
        report['feature_importance'][model_name] = {
            feat: float(imp) for feat, imp in feat_imp
        }
    
    # Save report
    os.makedirs('results', exist_ok=True)
    with open('results/model_evaluation.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n✅ Training and evaluation complete!")
    print(f"   Results saved to results/model_evaluation.json")
    print(f"   Models saved to models/trained/")
    
    return recommender, results

if __name__ == "__main__":
    recommender, results = main()



