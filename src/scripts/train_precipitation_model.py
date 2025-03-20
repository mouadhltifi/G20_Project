import os
import numpy as np
import pandas as pd
import rasterio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import joblib

# Configure TensorFlow for Metal GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        logging.info("Metal GPU is available and configured")
    except RuntimeError as e:
        logging.warning(f"Error configuring Metal GPU: {e}")
else:
    logging.warning("No Metal GPU found, using CPU")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('precipitation_training.log'),
        logging.StreamHandler()
    ]
)

def load_precipitation_data(data_dir):
    """Load precipitation data from TIF files"""
    logging.info("Loading precipitation data...")
    precipitation_data = {}
    
    # Get all TIF files in the directory
    files = [f for f in os.listdir(data_dir) if f.endswith('.tif') and 'R.' in f]
    
    for file in files:
        year = int(file.split('R')[0])
        if 2011 <= year <= 2023:
            file_path = os.path.join(data_dir, file)
            with rasterio.open(file_path) as src:
                data = src.read(1)
                # Mask no-data values (they are stored as -3.4028234663852886e+38)
                no_data_value = -3.4028234663852886e+38
                valid_data = data[data != no_data_value]
                
                if len(valid_data) > 0:
                    mean_precip = float(np.mean(valid_data))
                    precipitation_data[year] = mean_precip
                    logging.info(f"Loaded data for year {year}: Mean precipitation = {mean_precip:.2f} mm")
                    logging.info(f"  Valid pixels: {len(valid_data)}, Total pixels: {data.size}")
                else:
                    logging.warning(f"No valid data found for year {year}, skipping...")
    
    if not precipitation_data:
        raise ValueError("No valid precipitation data found!")
    
    logging.info(f"Successfully loaded data for {len(precipitation_data)} years")
    return precipitation_data

def prepare_data(precipitation_data):
    """Prepare data for training"""
    logging.info("Preparing data for training...")
    
    # Convert to numpy arrays
    years = np.array(list(precipitation_data.keys()))
    precip = np.array(list(precipitation_data.values()))
    
    # Log data statistics
    logging.info(f"Years covered: {min(years)} to {max(years)}")
    logging.info(f"Precipitation range: {np.min(precip):.2f} to {np.max(precip):.2f} mm")
    logging.info(f"Mean precipitation: {np.mean(precip):.2f} mm")
    logging.info(f"Standard deviation: {np.std(precip):.2f} mm")
    
    # Scale both features and target
    X = years.reshape(-1, 1)
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    X_scaled = X_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(precip.reshape(-1, 1)).ravel()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )
    
    logging.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
    return X_train, X_test, y_train, y_test, X_scaler, y_scaler

def create_model():
    """Create and compile the neural network model"""
    logging.info("Creating neural network model...")
    
    model = Sequential([
        Dense(64, activation='relu', input_shape=(1,), 
              kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dropout(0.3),
        Dense(32, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dropout(0.2),
        Dense(16, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dense(1)
    ])
    
    # Use mixed precision for better performance on Metal GPU
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')],
        jit_compile=True  # Enable XLA compilation for better performance
    )
    
    logging.info("Model created successfully")
    logging.info(model.summary())
    return model

def train_model(model, X_train, y_train, X_test, y_test):
    """Train the model with callbacks"""
    logging.info("Starting model training...")
    
    # Create callbacks
    checkpoint = ModelCheckpoint(
        'precipitation_model.h5',
        monitor='val_loss',
        save_best_only=True,
        mode='min'
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=50,
        restore_best_weights=True
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=10,
        min_lr=0.0001
    )
    
    # Train the model with optimized settings for Metal GPU
    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=8,  # Increased batch size for better GPU utilization
        validation_split=0.2,
        callbacks=[checkpoint, early_stopping, reduce_lr],
        verbose=1,
        use_multiprocessing=True,  # Enable multiprocessing for data loading
        workers=4  # Number of CPU cores to use for data loading
    )
    
    # Evaluate the model
    test_loss, test_mae, test_rmse = model.evaluate(X_test, y_test, verbose=0)
    logging.info(f"Test loss: {test_loss:.4f}")
    logging.info(f"Test MAE: {test_mae:.4f}")
    logging.info(f"Test RMSE: {test_rmse:.4f}")
    
    return history

def plot_training_history(history):
    """Plot and save training history"""
    logging.info("Plotting training history...")
    
    plt.figure(figsize=(15, 5))
    
    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot MAE
    plt.subplot(1, 3, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    # Plot RMSE
    plt.subplot(1, 3, 3)
    plt.plot(history.history['rmse'], label='Training RMSE')
    plt.plot(history.history['val_rmse'], label='Validation RMSE')
    plt.title('Model RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('precipitation_training_history.png')
    logging.info("Training history plot saved as 'precipitation_training_history.png'")

def main():
    """Main function to run the training process"""
    try:
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Load data
        data_dir = "Datasets_Hackathon/Climate_Precipitation_Data"
        precipitation_data = load_precipitation_data(data_dir)
        
        # Prepare data
        X_train, X_test, y_train, y_test, X_scaler, y_scaler = prepare_data(precipitation_data)
        
        # Create and train model
        model = create_model()
        history = train_model(model, X_train, y_train, X_test, y_test)
        
        # Plot training history
        plot_training_history(history)
        
        # Save scalers
        joblib.dump(X_scaler, 'precipitation_X_scaler.joblib')
        joblib.dump(y_scaler, 'precipitation_y_scaler.joblib')
        logging.info("Saved scalers to disk")
        
        logging.info("Training completed successfully!")
        
    except Exception as e:
        logging.error(f"An error occurred during training: {str(e)}")
        raise

if __name__ == "__main__":
    main() 