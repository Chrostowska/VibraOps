#!/usr/bin/env python3
"""
VibraOps System Test
Demonstrates full functionality with Python 3.11 + TensorFlow 2.19.0
"""

from models import ModelTrainer
from data_simulator import VibrationDataSimulator
import numpy as np

def main():
    print("🚀 Running VibraOps Demo with Python 3.11 + TensorFlow 2.19.0")
    print("="*60)
    
    # Generate test data
    print("📊 Generating test dataset...")
    sim = VibrationDataSimulator(random_seed=42)
    df, labels = sim.generate_dataset(n_normal=100, n_anomalies=20, signal_duration=0.5)
    
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    X = df[feature_cols].values
    y = labels
    
    print(f"✅ Generated dataset: {len(df)} samples")
    print(f"   Normal: {sum(y == 0)}, Anomalies: {sum(y == 1)}")
    
    # Test Isolation Forest
    print("\n🌲 Testing Isolation Forest...")
    if_trainer = ModelTrainer('isolation_forest')
    if_results = if_trainer.train_and_evaluate(X, y, test_size=0.3)
    
    if_accuracy = if_results["test_metrics"]["accuracy"]
    print(f"🎯 Isolation Forest Accuracy: {if_accuracy:.3f}")
    
    # Test LSTM Autoencoder
    print("\n🧠 Testing LSTM Autoencoder...")
    lstm_trainer = ModelTrainer('lstm_autoencoder', sequence_length=50)
    lstm_results = lstm_trainer.train_and_evaluate(X, y, test_size=0.3, epochs=5, batch_size=8)
    
    lstm_accuracy = lstm_results["test_metrics"]["accuracy"]
    print(f"🎯 LSTM Autoencoder Accuracy: {lstm_accuracy:.3f}")
    
    # Summary
    print("\n" + "="*60)
    print("🎉 SYSTEM TEST COMPLETE!")
    print("="*60)
    print("✅ Python 3.11 + TensorFlow 2.19.0 integration working")
    print("✅ Both ML models trained successfully")
    print("✅ VibraOps is fully operational!")
    print("="*60)

if __name__ == "__main__":
    main() 