"""
FastAPI Service for Vibration Anomaly Detection

This module provides a REST API for real-time anomaly detection using trained models.
It supports both Isolation Forest and LSTM Autoencoder models for vibration analysis.

Endpoints:
- POST /predict: Predict anomalies in vibration data
- GET /health: Health check
- GET /model-info: Get loaded model information
- POST /predict/batch: Batch prediction for multiple samples
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import List, Dict, Any, Optional, Union
import numpy as np
import json
import logging
import time
from datetime import datetime
import os
import asyncio
from contextlib import asynccontextmanager
import uvicorn

from models import IsolationForestAnomalyDetector, LSTMAutoencoder
from data_simulator import VibrationDataSimulator


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Pydantic models for request/response
class VibrationData(BaseModel):
    """Input model for vibration data."""
    signal: List[float]
    timestamp: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = {}
    
    @validator('signal')
    def validate_signal(cls, v):
        if len(v) < 10:
            raise ValueError('Signal must contain at least 10 data points')
        if len(v) > 10000:
            raise ValueError('Signal too long (max 10000 points)')
        return v


class BatchVibrationData(BaseModel):
    """Input model for batch vibration data."""
    signals: List[List[float]]
    timestamps: Optional[List[str]] = None
    metadata: Optional[List[Dict[str, Any]]] = None
    
    @validator('signals')
    def validate_signals(cls, v):
        if len(v) > 100:
            raise ValueError('Batch size too large (max 100 samples)')
        for i, signal in enumerate(v):
            if len(signal) < 10:
                raise ValueError(f'Signal {i} must contain at least 10 data points')
            if len(signal) > 10000:
                raise ValueError(f'Signal {i} too long (max 10000 points)')
        return v


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    is_anomaly: bool
    anomaly_score: float
    confidence: float
    timestamp: str
    processing_time_ms: float
    model_used: str
    metadata: Optional[Dict[str, Any]] = {}


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[PredictionResponse]
    batch_summary: Dict[str, Any]
    total_processing_time_ms: float


class ModelInfo(BaseModel):
    """Model information response."""
    model_type: str
    is_loaded: bool
    model_path: str
    load_timestamp: str
    version: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    uptime_seconds: float
    model_loaded: bool
    memory_usage_mb: float


# Global variables for model management
loaded_model = None
model_type = None
model_load_time = None
app_start_time = time.time()


# Model loading functions
def load_isolation_forest_model(model_path: str) -> IsolationForestAnomalyDetector:
    """Load Isolation Forest model."""
    model = IsolationForestAnomalyDetector()
    model.load_model(model_path)
    return model


def load_lstm_model(model_path: str) -> LSTMAutoencoder:
    """Load LSTM Autoencoder model."""
    model = LSTMAutoencoder()
    model.load_model(model_path)
    return model


def initialize_model():
    """Initialize and load the anomaly detection model."""
    global loaded_model, model_type, model_load_time
    
    # Try to load models in order of preference
    model_paths = [
        ('isolation_forest', 'models/isolation_forest_model.pkl'),
        ('lstm_autoencoder', 'models/lstm_autoencoder_model'),
    ]
    
    for mtype, path in model_paths:
        try:
            logger.info(f"Attempting to load {mtype} model from {path}")
            
            if mtype == 'isolation_forest' and os.path.exists(path):
                loaded_model = load_isolation_forest_model(path)
                model_type = mtype
                model_load_time = datetime.now()
                logger.info(f"Successfully loaded {mtype} model")
                return
            elif mtype == 'lstm_autoencoder':
                # Check for LSTM files
                if (os.path.exists(f"{path}_keras.h5") and 
                    os.path.exists(f"{path}_metadata.json") and 
                    os.path.exists(f"{path}_scaler.pkl")):
                    loaded_model = load_lstm_model(path)
                    model_type = mtype
                    model_load_time = datetime.now()
                    logger.info(f"Successfully loaded {mtype} model")
                    return
        except Exception as e:
            logger.warning(f"Failed to load {mtype} model: {str(e)}")
            continue
    
    # If no models found, create a simulator for demo purposes
    logger.warning("No trained models found. Using simulator for demo mode.")
    loaded_model = VibrationDataSimulator(random_seed=42)
    model_type = 'demo_simulator'
    model_load_time = datetime.now()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting VibraOps API service...")
    
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Initialize model
    initialize_model()
    
    logger.info("VibraOps API service started successfully")
    yield
    
    # Shutdown
    logger.info("Shutting down VibraOps API service...")


# Create FastAPI app
app = FastAPI(
    title="VibraOps Anomaly Detection API",
    description="Real-time vibration anomaly detection using machine learning",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Utility functions
def get_memory_usage() -> float:
    """Get memory usage in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


def log_prediction(prediction_data: Dict[str, Any]):
    """Log prediction for monitoring."""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'prediction': prediction_data,
        'model_type': model_type
    }
    
    # Write to prediction log
    with open('logs/predictions.jsonl', 'a') as f:
        f.write(json.dumps(log_entry) + '\n')


def make_prediction(signal: List[float], metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """Make a single prediction."""
    if loaded_model is None:
        raise HTTPException(status_code=503, detail="No model loaded")
    
    start_time = time.time()
    
    try:
        signal_array = np.array(signal).reshape(1, -1)
        
        if model_type in ['isolation_forest', 'lstm_autoencoder']:
            predictions, scores = loaded_model.predict(signal_array)
            is_anomaly = bool(predictions[0])
            anomaly_score = float(scores[0])
            confidence = 1.0 - abs(0.5 - anomaly_score) * 2  # Convert to confidence
            
        elif model_type == 'demo_simulator':
            # Demo mode: random predictions for demonstration
            anomaly_score = np.random.random()
            is_anomaly = anomaly_score > 0.7
            confidence = 0.8
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        processing_time = (time.time() - start_time) * 1000
        
        result = {
            'is_anomaly': is_anomaly,
            'anomaly_score': anomaly_score,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'processing_time_ms': processing_time,
            'model_used': model_type,
            'metadata': metadata or {}
        }
        
        # Log prediction
        log_prediction(result)
        
        return result
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# API Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with basic information."""
    return {
        "service": "VibraOps Anomaly Detection API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    uptime = time.time() - app_start_time
    memory_usage = get_memory_usage()
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        uptime_seconds=uptime,
        model_loaded=loaded_model is not None,
        memory_usage_mb=memory_usage
    )


@app.get("/model-info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the loaded model."""
    if loaded_model is None:
        raise HTTPException(status_code=503, detail="No model loaded")
    
    return ModelInfo(
        model_type=model_type,
        is_loaded=True,
        model_path=f"models/{model_type}_model",
        load_timestamp=model_load_time.isoformat() if model_load_time else "",
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_anomaly(data: VibrationData, background_tasks: BackgroundTasks):
    """
    Predict anomaly for a single vibration signal.
    
    Args:
        data: Vibration data containing signal and metadata
        
    Returns:
        Prediction result with anomaly score and classification
    """
    try:
        result = make_prediction(data.signal, data.metadata)
        
        # Add background task for additional logging if needed
        background_tasks.add_task(
            lambda: logger.info(f"Processed prediction: anomaly={result['is_anomaly']}, score={result['anomaly_score']:.3f}")
        )
        
        return PredictionResponse(**result)
        
    except Exception as e:
        logger.error(f"Prediction endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch_anomalies(data: BatchVibrationData):
    """
    Predict anomalies for multiple vibration signals in batch.
    
    Args:
        data: Batch of vibration signals
        
    Returns:
        Batch prediction results with summary statistics
    """
    if loaded_model is None:
        raise HTTPException(status_code=503, detail="No model loaded")
    
    start_time = time.time()
    
    try:
        predictions = []
        anomaly_count = 0
        total_score = 0.0
        
        for i, signal in enumerate(data.signals):
            metadata = data.metadata[i] if data.metadata and i < len(data.metadata) else {}
            result = make_prediction(signal, metadata)
            predictions.append(PredictionResponse(**result))
            
            if result['is_anomaly']:
                anomaly_count += 1
            total_score += result['anomaly_score']
        
        total_processing_time = (time.time() - start_time) * 1000
        
        batch_summary = {
            'total_samples': len(data.signals),
            'anomalies_detected': anomaly_count,
            'anomaly_rate': anomaly_count / len(data.signals),
            'average_score': total_score / len(data.signals),
            'model_used': model_type
        }
        
        return BatchPredictionResponse(
            predictions=predictions,
            batch_summary=batch_summary,
            total_processing_time_ms=total_processing_time
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/simulate-data")
async def simulate_vibration_data(
    duration: float = 1.0,
    anomaly_type: str = "normal",
    sampling_rate: int = 1000
):
    """
    Generate simulated vibration data for testing.
    
    Args:
        duration: Signal duration in seconds
        anomaly_type: Type of signal ('normal', 'spike', 'frequency', 'bearing_fault')
        sampling_rate: Sampling rate in Hz
        
    Returns:
        Simulated vibration signal
    """
    try:
        simulator = VibrationDataSimulator(sampling_rate=sampling_rate, random_seed=None)
        
        if anomaly_type == "normal":
            signal = simulator.generate_normal_vibration(duration)
        elif anomaly_type == "spike":
            signal = simulator.generate_anomaly_spike(duration)
        elif anomaly_type == "frequency":
            signal = simulator.generate_frequency_anomaly(duration)
        elif anomaly_type == "bearing_fault":
            signal = simulator.generate_bearing_fault(duration)
        else:
            raise ValueError(f"Unknown anomaly type: {anomaly_type}")
        
        return {
            "signal": signal.tolist(),
            "duration": duration,
            "sampling_rate": sampling_rate,
            "anomaly_type": anomaly_type,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 