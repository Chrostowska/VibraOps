"""
Vibration Data Simulator for MLOps Anomaly Detection Project

This module generates realistic vibration time-series data with:
- Normal vibration patterns (sine waves with noise)
- Anomalous patterns (outliers, frequency shifts, amplitude changes)
- Configurable parameters for different scenarios
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional
import json
from datetime import datetime, timedelta


class VibrationDataSimulator:
    """
    Simulates vibration data for machinery monitoring with normal and anomalous patterns.
    """
    
    def __init__(self, 
                 sampling_rate: int = 1000,
                 base_frequency: float = 50.0,
                 noise_level: float = 0.1,
                 random_seed: Optional[int] = 42):
        """
        Initialize the vibration data simulator.
        
        Args:
            sampling_rate: Samples per second
            base_frequency: Base vibration frequency in Hz
            noise_level: Noise amplitude relative to signal
            random_seed: Random seed for reproducibility
        """
        self.sampling_rate = sampling_rate
        self.base_frequency = base_frequency
        self.noise_level = noise_level
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def generate_normal_vibration(self, 
                                duration: float,
                                amplitude: float = 1.0,
                                harmonics: List[float] = [1.0, 0.3, 0.1]) -> np.ndarray:
        """
        Generate normal vibration signal with harmonics and noise.
        
        Args:
            duration: Duration in seconds
            amplitude: Signal amplitude
            harmonics: List of harmonic amplitudes (relative to fundamental)
            
        Returns:
            Generated vibration signal
        """
        n_samples = int(duration * self.sampling_rate)
        t = np.linspace(0, duration, n_samples)
        
        # Generate signal with harmonics
        signal = np.zeros(n_samples)
        for i, harmonic_amp in enumerate(harmonics, 1):
            signal += harmonic_amp * amplitude * np.sin(
                2 * np.pi * self.base_frequency * i * t
            )
        
        # Add Gaussian noise
        noise = self.noise_level * amplitude * np.random.normal(0, 1, n_samples)
        signal += noise
        
        return signal
    
    def generate_anomaly_spike(self, 
                              duration: float,
                              spike_amplitude: float = 5.0,
                              spike_start: float = 0.3,
                              spike_width: float = 0.1) -> np.ndarray:
        """
        Generate anomalous vibration with amplitude spikes.
        
        Args:
            duration: Duration in seconds
            spike_amplitude: Amplitude multiplier for spike
            spike_start: Relative position of spike (0-1)
            spike_width: Relative width of spike (0-1)
            
        Returns:
            Anomalous vibration signal
        """
        signal = self.generate_normal_vibration(duration)
        n_samples = len(signal)
        
        # Add spike
        spike_start_idx = int(spike_start * n_samples)
        spike_end_idx = int((spike_start + spike_width) * n_samples)
        
        # Create spike envelope
        spike_length = spike_end_idx - spike_start_idx
        spike_envelope = np.exp(-((np.arange(spike_length) - spike_length/2) / (spike_length/6))**2)
        
        signal[spike_start_idx:spike_end_idx] += spike_amplitude * spike_envelope
        
        return signal
    
    def generate_frequency_anomaly(self, 
                                  duration: float,
                                  frequency_shift: float = 0.3,
                                  shift_start: float = 0.4,
                                  shift_duration: float = 0.3) -> np.ndarray:
        """
        Generate anomalous vibration with frequency shifts.
        
        Args:
            duration: Duration in seconds
            frequency_shift: Relative frequency change
            shift_start: Relative position where shift starts (0-1)
            shift_duration: Relative duration of shift (0-1)
            
        Returns:
            Anomalous vibration signal
        """
        n_samples = int(duration * self.sampling_rate)
        t = np.linspace(0, duration, n_samples)
        
        # Calculate frequency modulation
        shift_start_idx = int(shift_start * n_samples)
        shift_end_idx = int((shift_start + shift_duration) * n_samples)
        
        frequency = np.full(n_samples, self.base_frequency)
        frequency[shift_start_idx:shift_end_idx] *= (1 + frequency_shift)
        
        # Generate signal with varying frequency
        phase = 2 * np.pi * np.cumsum(frequency) / self.sampling_rate
        signal = np.sin(phase)
        
        # Add noise
        noise = self.noise_level * np.random.normal(0, 1, n_samples)
        signal += noise
        
        return signal
    
    def generate_bearing_fault(self, 
                              duration: float,
                              fault_frequency: float = 120.0,
                              fault_amplitude: float = 0.8) -> np.ndarray:
        """
        Generate bearing fault signature.
        
        Args:
            duration: Duration in seconds
            fault_frequency: Bearing fault frequency in Hz
            fault_amplitude: Amplitude of fault signature
            
        Returns:
            Bearing fault vibration signal
        """
        signal = self.generate_normal_vibration(duration)
        n_samples = len(signal)
        t = np.linspace(0, duration, n_samples)
        
        # Add bearing fault signature (impulses at fault frequency)
        fault_impulses = fault_amplitude * np.sin(2 * np.pi * fault_frequency * t)
        fault_impulses *= np.exp(-5 * (t % (1/fault_frequency)) / (1/fault_frequency))
        
        return signal + fault_impulses
    
    def generate_dataset(self, 
                        n_normal: int = 1000,
                        n_anomalies: int = 200,
                        signal_duration: float = 2.0,
                        save_path: Optional[str] = None) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Generate a complete dataset with normal and anomalous samples.
        
        Args:
            n_normal: Number of normal samples
            n_anomalies: Number of anomalous samples
            signal_duration: Duration of each signal in seconds
            save_path: Optional path to save the dataset
            
        Returns:
            Tuple of (features_dataframe, labels_array)
        """
        print(f"Generating dataset: {n_normal} normal + {n_anomalies} anomalous samples...")
        
        all_signals = []
        all_labels = []
        all_metadata = []
        
        # Generate normal samples
        for i in range(n_normal):
            signal = self.generate_normal_vibration(signal_duration)
            all_signals.append(signal)
            all_labels.append(0)  # Normal
            all_metadata.append({
                'sample_id': f'normal_{i:04d}',
                'type': 'normal',
                'timestamp': datetime.now() + timedelta(seconds=i*signal_duration)
            })
        
        # Generate anomalous samples (mixed types)
        anomaly_types = ['spike', 'frequency', 'bearing_fault']
        
        for i in range(n_anomalies):
            anomaly_type = np.random.choice(anomaly_types)
            
            if anomaly_type == 'spike':
                signal = self.generate_anomaly_spike(signal_duration)
            elif anomaly_type == 'frequency':
                signal = self.generate_frequency_anomaly(signal_duration)
            else:  # bearing_fault
                signal = self.generate_bearing_fault(signal_duration)
            
            all_signals.append(signal)
            all_labels.append(1)  # Anomaly
            all_metadata.append({
                'sample_id': f'anomaly_{i:04d}',
                'type': anomaly_type,
                'timestamp': datetime.now() + timedelta(seconds=(n_normal + i)*signal_duration)
            })
        
        # Convert to DataFrame
        feature_names = [f'feature_{i:04d}' for i in range(len(all_signals[0]))]
        df = pd.DataFrame(all_signals, columns=feature_names)
        
        # Add metadata
        for key in all_metadata[0].keys():
            df[key] = [meta[key] for meta in all_metadata]
        
        labels = np.array(all_labels)
        
        # Save if path provided
        if save_path:
            df.to_csv(f"{save_path}_features.csv", index=False)
            np.save(f"{save_path}_labels.npy", labels)
            print(f"Dataset saved to {save_path}_features.csv and {save_path}_labels.npy")
        
        print(f"Dataset generated successfully!")
        print(f"Normal samples: {n_normal}, Anomalous samples: {n_anomalies}")
        print(f"Signal duration: {signal_duration}s, Sampling rate: {self.sampling_rate}Hz")
        
        return df, labels
    
    def visualize_samples(self, df: pd.DataFrame, labels: np.ndarray, n_samples: int = 4):
        """
        Visualize sample signals from the dataset.
        
        Args:
            df: Features DataFrame
            labels: Labels array
            n_samples: Number of samples to visualize
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        # Get feature columns
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        
        # Plot normal samples
        normal_indices = np.where(labels == 0)[0][:n_samples//2]
        anomaly_indices = np.where(labels == 1)[0][:n_samples//2]
        
        for i, idx in enumerate(normal_indices):
            signal = df.iloc[idx][feature_cols].values
            t = np.linspace(0, len(signal)/self.sampling_rate, len(signal))
            axes[i].plot(t, signal, 'b-', alpha=0.8)
            axes[i].set_title(f'Normal Signal - {df.iloc[idx]["sample_id"]}')
            axes[i].set_xlabel('Time (s)')
            axes[i].set_ylabel('Amplitude')
            axes[i].grid(True, alpha=0.3)
        
        for i, idx in enumerate(anomaly_indices, len(normal_indices)):
            signal = df.iloc[idx][feature_cols].values
            t = np.linspace(0, len(signal)/self.sampling_rate, len(signal))
            axes[i].plot(t, signal, 'r-', alpha=0.8)
            axes[i].set_title(f'Anomaly Signal - {df.iloc[idx]["sample_id"]} ({df.iloc[idx]["type"]})')
            axes[i].set_xlabel('Time (s)')
            axes[i].set_ylabel('Amplitude')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('sample_vibrations.png', dpi=150, bbox_inches='tight')
        plt.show()


def main():
    """
    Main function to demonstrate the vibration data simulator.
    """
    # Initialize simulator
    simulator = VibrationDataSimulator(
        sampling_rate=1000,
        base_frequency=50.0,
        noise_level=0.1,
        random_seed=42
    )
    
    # Generate dataset
    df, labels = simulator.generate_dataset(
        n_normal=100,
        n_anomalies=25,
        signal_duration=1.0,
        save_path="data/vibration_dataset"
    )
    
    # Visualize samples
    simulator.visualize_samples(df, labels, n_samples=4)
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Normal samples: {sum(labels == 0)} ({sum(labels == 0)/len(labels)*100:.1f}%)")
    print(f"Anomalous samples: {sum(labels == 1)} ({sum(labels == 1)/len(labels)*100:.1f}%)")
    print(f"Features per sample: {len([col for col in df.columns if col.startswith('feature_')])}")


if __name__ == "__main__":
    main() 