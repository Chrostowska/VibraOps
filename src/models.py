"""
ML Models for Vibration Anomaly Detection

This module implements two anomaly detection approaches:
1. Isolation Forest (scikit-learn) - Tree-based unsupervised anomaly detection
2. LSTM Autoencoder (TensorFlow) - Deep learning reconstruction-based detection

Both models are trained on normal vibration data and detect anomalies based on 
deviation from normal patterns.
"""

import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

# TensorFlow and Keras imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Optional, Union
import os
import json
from datetime import datetime


class IsolationForestAnomalyDetector:
    """
    Isolation Forest based anomaly detector for vibration data.
    """
    
    def __init__(self, 
                 contamination: float = 0.1,
                 n_estimators: int = 100,
                 max_samples: str = 'auto',
                 random_state: int = 42):
        """
        Initialize Isolation Forest detector.
        
        Args:
            contamination: Expected proportion of outliers in the data
            n_estimators: Number of base estimators in the ensemble
            max_samples: Number of samples to draw from X to train each base estimator
            random_state: Random state for reproducibility
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1  # Use all available cores
        )
        
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def train(self, X: np.ndarray, y: np.ndarray = None) -> Dict:
        """
        Train the Isolation Forest model.
        
        Args:
            X: Training data (n_samples, n_features)
            y: Labels (ignored, unsupervised learning)
            
        Returns:
            Training metrics dictionary
        """
        # Filter to normal data only for training (unsupervised)
        if y is not None:
            normal_indices = y == 0
            X_normal = X[normal_indices]
        else:
            X_normal = X
            
        # Scale features
        X_scaled = self.scaler.fit_transform(X_normal)
        
        # Train model
        print(f"Training Isolation Forest on {X_scaled.shape[0]} normal samples...")
        self.model.fit(X_scaled)
        self.is_trained = True
        
        # Calculate training anomaly scores
        train_scores = self.model.decision_function(X_scaled)
        
        training_metrics = {
            'model_type': 'isolation_forest',
            'n_samples': X_scaled.shape[0],
            'n_features': X_scaled.shape[1],
            'contamination': self.contamination,
            'mean_anomaly_score': float(np.mean(train_scores)),
            'std_anomaly_score': float(np.std(train_scores))
        }
        
        return training_metrics
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies in the data.
        
        Args:
            X: Input data (n_samples, n_features)
            
        Returns:
            Tuple of (predictions, anomaly_scores)
            predictions: Binary labels (0=normal, 1=anomaly)
            anomaly_scores: Anomaly scores (lower = more anomalous)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
            
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get predictions and scores
        predictions = self.model.predict(X_scaled)
        anomaly_scores = self.model.decision_function(X_scaled)
        
        # Convert predictions: Isolation Forest returns 1 for normal, -1 for anomaly
        # We want 0 for normal, 1 for anomaly
        predictions = (predictions == -1).astype(int)
        
        return predictions, anomaly_scores
    
    def plot_anomaly_scores(self, X: np.ndarray, y: np.ndarray = None, 
                           title: str = "Anomaly Scores Distribution"):
        """Plot distribution of anomaly scores."""
        _, scores = self.predict(X)
        
        plt.figure(figsize=(12, 6))
        
        if y is not None:
            # Separate scores by class
            normal_scores = scores[y == 0]
            anomaly_scores = scores[y == 1]
            
            plt.subplot(1, 2, 1)
            plt.hist([normal_scores, anomaly_scores], bins=50, alpha=0.7, 
                    label=['Normal', 'Anomaly'], color=['blue', 'red'])
            plt.xlabel('Anomaly Score')
            plt.ylabel('Frequency')
            plt.title('Score Distribution by Class')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.scatter(range(len(scores)), scores, c=y, cmap='coolwarm', alpha=0.6)
            plt.xlabel('Sample Index')
            plt.ylabel('Anomaly Score')
            plt.title('Anomaly Scores Over Time')
            plt.colorbar(label='Class (0=Normal, 1=Anomaly)')
        else:
            plt.hist(scores, bins=50, alpha=0.7, color='blue')
            plt.xlabel('Anomaly Score')
            plt.ylabel('Frequency')
            plt.title(title)
        
        plt.tight_layout()
        plt.savefig(f'anomaly_scores_isolation_forest.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'contamination': self.contamination,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.contamination = model_data['contamination']
        self.is_trained = model_data['is_trained']
        print(f"Model loaded from {filepath}")


class LSTMAutoencoder:
    """
    LSTM Autoencoder for vibration anomaly detection.
    """
    
    def __init__(self,
                 sequence_length: int = 100,
                 n_features: int = 1,
                 latent_dim: int = 32,
                 lstm_units: int = 64):
        """
        Initialize LSTM Autoencoder.
        
        Args:
            sequence_length: Length of input sequences
            n_features: Number of features per timestep
            latent_dim: Dimension of latent representation
            lstm_units: Number of LSTM units
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.lstm_units = lstm_units
        
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_trained = False
        self.threshold = None
        
    def build_model(self):
        """Build the LSTM Autoencoder architecture."""
        # Encoder
        input_layer = layers.Input(shape=(self.sequence_length, self.n_features))
        
        # Encoder LSTM layers
        x = layers.LSTM(self.lstm_units, return_sequences=True)(input_layer)
        x = layers.Dropout(0.2)(x)
        x = layers.LSTM(self.latent_dim, return_sequences=False)(x)
        encoded = layers.Dropout(0.2)(x)
        
        # Decoder
        x = layers.RepeatVector(self.sequence_length)(encoded)
        x = layers.LSTM(self.latent_dim, return_sequences=True)(x)
        x = layers.Dropout(0.2)(x)
        x = layers.LSTM(self.lstm_units, return_sequences=True)(x)
        x = layers.Dropout(0.2)(x)
        
        # Output layer
        decoded = layers.TimeDistributed(layers.Dense(self.n_features))(x)
        
        # Create model
        self.model = keras.Model(input_layer, decoded)
        self.model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return self.model
    
    def prepare_sequences(self, data: np.ndarray) -> np.ndarray:
        """
        Convert time series data into sequences for LSTM.
        
        Args:
            data: Time series data (n_samples, signal_length)
            
        Returns:
            Sequences (n_sequences, sequence_length, n_features)
        """
        sequences = []
        
        for signal in data:
            # Normalize signal
            signal_norm = self.scaler.fit_transform(signal.reshape(-1, 1)).flatten()
            
            # Create overlapping sequences
            for i in range(len(signal_norm) - self.sequence_length + 1):
                sequence = signal_norm[i:i + self.sequence_length]
                sequences.append(sequence)
        
        return np.array(sequences).reshape(-1, self.sequence_length, self.n_features)
    
    def train(self, X: np.ndarray, y: np.ndarray = None, 
              epochs: int = 50, batch_size: int = 32, validation_split: float = 0.2) -> Dict:
        """
        Train the LSTM Autoencoder.
        
        Args:
            X: Training data (n_samples, signal_length)
            y: Labels (ignored for autoencoder training)
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            
        Returns:
            Training metrics dictionary
        """
        # Filter to normal data only for training
        if y is not None:
            normal_indices = y == 0
            X_normal = X[normal_indices]
        else:
            X_normal = X
            
        # Prepare sequences
        print(f"Preparing sequences from {X_normal.shape[0]} normal samples...")
        X_sequences = self.prepare_sequences(X_normal)
        print(f"Created {X_sequences.shape[0]} sequences of length {self.sequence_length}")
        
        # Build model
        if self.model is None:
            self.build_model()
            
        # Train autoencoder (input = output for reconstruction)
        print("Training LSTM Autoencoder...")
        history = self.model.fit(
            X_sequences, X_sequences,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1,
            shuffle=True
        )
        
        self.is_trained = True
        
        # Calculate reconstruction threshold on training data
        train_predictions = self.model.predict(X_sequences, verbose=0)
        train_mae = np.mean(np.abs(X_sequences - train_predictions), axis=(1, 2))
        self.threshold = np.percentile(train_mae, 95)  # 95th percentile as threshold
        
        training_metrics = {
            'model_type': 'lstm_autoencoder',
            'n_sequences': X_sequences.shape[0],
            'sequence_length': self.sequence_length,
            'final_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1]),
            'reconstruction_threshold': float(self.threshold),
            'epochs_trained': epochs
        }
        
        return training_metrics
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies using reconstruction error.
        
        Args:
            X: Input data (n_samples, signal_length)
            
        Returns:
            Tuple of (predictions, reconstruction_errors)
            predictions: Binary labels (0=normal, 1=anomaly)
            reconstruction_errors: MAE reconstruction errors per sample
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
            
        # Prepare sequences
        X_sequences = self.prepare_sequences(X)
        
        # Get reconstructions
        reconstructions = self.model.predict(X_sequences, verbose=0)
        
        # Calculate reconstruction errors (MAE per sequence)
        mae_errors = np.mean(np.abs(X_sequences - reconstructions), axis=(1, 2))
        
        # Aggregate errors per original sample
        # Note: Each sample creates multiple overlapping sequences
        sequences_per_sample = X.shape[1] - self.sequence_length + 1
        sample_errors = []
        
        for i in range(X.shape[0]):
            start_idx = i * sequences_per_sample
            end_idx = start_idx + sequences_per_sample
            sample_error = np.mean(mae_errors[start_idx:end_idx])
            sample_errors.append(sample_error)
        
        sample_errors = np.array(sample_errors)
        
        # Make predictions based on threshold
        predictions = (sample_errors > self.threshold).astype(int)
        
        return predictions, sample_errors
    
    def plot_training_history(self, history):
        """Plot training history."""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('lstm_training_history.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_reconstruction_errors(self, X: np.ndarray, y: np.ndarray = None):
        """Plot reconstruction error distribution."""
        _, errors = self.predict(X)
        
        plt.figure(figsize=(12, 6))
        
        if y is not None:
            # Separate errors by class
            normal_errors = errors[y == 0]
            anomaly_errors = errors[y == 1]
            
            plt.subplot(1, 2, 1)
            plt.hist([normal_errors, anomaly_errors], bins=50, alpha=0.7, 
                    label=['Normal', 'Anomaly'], color=['blue', 'red'])
            plt.axvline(self.threshold, color='green', linestyle='--', 
                       label=f'Threshold ({self.threshold:.4f})')
            plt.xlabel('Reconstruction Error (MAE)')
            plt.ylabel('Frequency')
            plt.title('Error Distribution by Class')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.scatter(range(len(errors)), errors, c=y, cmap='coolwarm', alpha=0.6)
            plt.axhline(self.threshold, color='green', linestyle='--', 
                       label=f'Threshold ({self.threshold:.4f})')
            plt.xlabel('Sample Index')
            plt.ylabel('Reconstruction Error')
            plt.title('Reconstruction Errors Over Time')
            plt.colorbar(label='Class (0=Normal, 1=Anomaly)')
            plt.legend()
        else:
            plt.hist(errors, bins=50, alpha=0.7, color='blue')
            plt.axvline(self.threshold, color='green', linestyle='--', 
                       label=f'Threshold ({self.threshold:.4f})')
            plt.xlabel('Reconstruction Error (MAE)')
            plt.ylabel('Frequency')
            plt.title('Reconstruction Error Distribution')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('lstm_reconstruction_errors.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
            
        # Save Keras model
        self.model.save(f"{filepath}.keras")
        
        # Save additional components
        model_data = {
            'sequence_length': self.sequence_length,
            'n_features': self.n_features,
            'latent_dim': self.latent_dim,
            'lstm_units': self.lstm_units,
            'scaler': self.scaler,
            'threshold': self.threshold,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, f"{filepath}_metadata.pkl")
        
        print(f"Model saved to {filepath}.keras and {filepath}_metadata.pkl")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        # Load Keras model
        self.model = keras.models.load_model(f"{filepath}.keras")
        
        # Load additional components
        model_data = joblib.load(f"{filepath}_metadata.pkl")
        self.sequence_length = model_data['sequence_length']
        self.n_features = model_data['n_features']
        self.latent_dim = model_data['latent_dim']
        self.lstm_units = model_data['lstm_units']
        self.scaler = model_data['scaler']
        self.threshold = model_data['threshold']
        self.is_trained = model_data['is_trained']
        
        print(f"Model loaded from {filepath}.keras and {filepath}_metadata.pkl")


class ModelTrainer:
    """
    Unified interface for training different anomaly detection models.
    """
    
    def __init__(self, model_type: str = 'isolation_forest', **model_kwargs):
        """
        Initialize model trainer.
        
        Args:
            model_type: Type of model ('isolation_forest' or 'lstm_autoencoder')
            **model_kwargs: Additional arguments for model initialization
        """
        self.model_type = model_type.lower()
        
        if self.model_type == 'isolation_forest':
            self.model = IsolationForestAnomalyDetector(**model_kwargs)
        elif self.model_type == 'lstm_autoencoder':
            self.model = LSTMAutoencoder(**model_kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train_and_evaluate(self, X: np.ndarray, y: np.ndarray, 
                          test_size: float = 0.3, **kwargs) -> Dict:
        """
        Train and evaluate the model.
        
        Args:
            X: Feature data (n_samples, n_features)
            y: Labels (0=normal, 1=anomaly)
            test_size: Fraction of data to use for testing
            **kwargs: Additional arguments for training
            
        Returns:
            Evaluation metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Train model
        train_metrics = self.model.train(X_train, y_train, **kwargs)
        
        # Test model
        y_pred, scores = self.model.predict(X_test)
        
        # Calculate metrics
        test_metrics = {
            'accuracy': np.mean(y_pred == y_test),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'roc_auc': roc_auc_score(y_test, scores) if len(np.unique(y_test)) > 1 else 0.0
        }
        
        # Plot confusion matrix
        self.plot_confusion_matrix(y_test, y_pred)
        
        return {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'model_type': self.model_type
        }
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {self.model_type.title()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'confusion_matrix_{self.model_type}.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        self.model.save_model(filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        self.model.load_model(filepath)


def main():
    """
    Main function to demonstrate model training.
    """
    from data_simulator import VibrationDataSimulator
    
    # Generate sample data
    simulator = VibrationDataSimulator(random_seed=42)
    df, labels = simulator.generate_dataset(n_normal=100, n_anomalies=20, signal_duration=0.5)
    
    # Get feature data (raw signals)
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    X = df[feature_cols].values
    y = labels
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Normal samples: {sum(y == 0)}, Anomalous samples: {sum(y == 1)}")
    
    # Train Isolation Forest
    print("\n" + "="*50)
    print("Training Isolation Forest")
    print("="*50)
    
    if_trainer = ModelTrainer('isolation_forest')
    if_results = if_trainer.train_and_evaluate(X, y)
    if_trainer.save_model('models/isolation_forest_model.pkl')
    
    # Train LSTM Autoencoder
    print("\n" + "="*50)
    print("Training LSTM Autoencoder")
    print("="*50)
    
    lstm_trainer = ModelTrainer('lstm_autoencoder', sequence_length=50)
    lstm_results = lstm_trainer.train_and_evaluate(X, y, epochs=5, batch_size=8)
    lstm_trainer.save_model('models/lstm_autoencoder_model')
    
    # Print results
    print("\n" + "="*50)
    print("TRAINING RESULTS SUMMARY")
    print("="*50)
    
    print(f"\nIsolation Forest Test Accuracy: {if_results['test_metrics']['accuracy']:.3f}")
    print(f"Isolation Forest ROC-AUC: {if_results['test_metrics']['roc_auc']:.3f}")
    
    print(f"\nLSTM Autoencoder Test Accuracy: {lstm_results['test_metrics']['accuracy']:.3f}")
    print(f"LSTM Autoencoder ROC-AUC: {lstm_results['test_metrics']['roc_auc']:.3f}")


if __name__ == "__main__":
    main() 