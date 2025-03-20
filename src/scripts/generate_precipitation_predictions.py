import numpy as np
import rasterio
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import os
from tqdm import tqdm
import calendar
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def load_historical_data(start_year=2011, end_year=2023):
    """Load historical precipitation data with spatial information."""
    logging.info(f'Loading historical data from {start_year} to {end_year}...')
    data = []
    years = []
    coordinates = None
    
    # Load first file to get spatial information
    first_file = f'data/Datasets_Hackathon/Climate_Precipitation_Data/{start_year}R.tif'
    with rasterio.open(first_file) as src:
        # Get coordinate information
        transform = src.transform
        crs = src.crs
        height, width = src.shape
        
        # Create coordinate grid
        rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        lons, lats = rasterio.transform.xy(transform, rows, cols)
        coordinates = np.stack([lons, lats], axis=-1)
    
    # Load data for all years
    for year in tqdm(range(start_year, end_year + 1)):
        file_path = f'data/Datasets_Hackathon/Climate_Precipitation_Data/{year}R.tif'
        with rasterio.open(file_path) as src:
            year_data = src.read(1).astype(np.float32)
            
            # Handle invalid values
            # Replace negative values with 0 (precipitation can't be negative)
            year_data = np.maximum(year_data, 0)
            
            # Replace NaN and inf values with the mean of valid values
            valid_mask = np.isfinite(year_data)
            if np.any(valid_mask):
                mean_value = np.mean(year_data[valid_mask])
                year_data[~valid_mask] = mean_value
            else:
                year_data.fill(0)  # If all values are invalid, use 0
            
            data.append(year_data)
    
    return np.array(data), np.array(range(start_year, end_year + 1)), coordinates, transform, crs

def prepare_features_for_location(year, coordinates):
    """Create features for a specific location and year."""
    # Temporal features
    features = [
        year,                    # Year
        np.sin(2 * np.pi * year / 10),  # Seasonal features
        np.cos(2 * np.pi * year / 10),
        year ** 2,              # Polynomial features
        year ** 3,
        calendar.isleap(year),  # Leap year indicator
    ]
    
    # Spatial features (normalized coordinates)
    features.extend([
        coordinates[0],  # Longitude
        coordinates[1],  # Latitude
        coordinates[0] * coordinates[1],  # Interaction term
        coordinates[0]**2,  # Quadratic terms
        coordinates[1]**2
    ])
    
    return np.array(features, dtype=np.float32)

def prepare_training_data(historical_data, historical_years, coordinates):
    """Prepare training data maintaining spatial relationships."""
    height, width = coordinates.shape[:2]
    n_years = len(historical_years)
    
    X = []  # Features
    y = []  # Target values
    
    # Normalize coordinates
    coord_scaler = MinMaxScaler()
    flat_coords = coordinates.reshape(-1, 2)
    coord_scaler.fit(flat_coords)
    normalized_coords = coord_scaler.transform(flat_coords).reshape(height, width, 2)
    
    # For each location
    for i in tqdm(range(height), desc="Preparing training data"):
        for j in range(width):
            loc_coords = normalized_coords[i, j]
            
            # For each year at this location
            for year_idx, year in enumerate(historical_years):
                features = prepare_features_for_location(year, loc_coords)
                X.append(features)
                y.append(historical_data[year_idx, i, j])
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    # Ensure all values are finite
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    y = np.nan_to_num(y, nan=0, posinf=0, neginf=0)
    
    return X, y, coord_scaler

def train_model(X, y):
    """Train a Random Forest model."""
    logging.info('Training model...')
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model.fit(X_scaled, y)
    return model, scaler

def generate_predictions(model, scaler, coordinates, transform, crs, coord_scaler, start_year=2024, end_year=2030):
    """Generate predictions for future years at each location."""
    logging.info(f'Generating predictions for {start_year}-{end_year}...')
    
    # Create future years array
    future_years = np.arange(start_year, end_year + 1)
    
    # Initialize predictions array
    height, width = coordinates.shape[:2]
    n_years = len(future_years)
    predictions = np.zeros((n_years, height, width), dtype=np.float32)
    
    # Normalize coordinates
    flat_coords = coordinates.reshape(-1, 2)
    normalized_coords = coord_scaler.transform(flat_coords).reshape(height, width, 2)
    
    # Generate predictions for each location
    for i in tqdm(range(height), desc="Generating predictions"):
        for j in range(width):
            # Create features for this location using normalized coordinates
            loc_coords = normalized_coords[i, j]
            
            # Generate predictions for each future year at this location
            for year_idx, year in enumerate(future_years):
                features = prepare_features_for_location(year, loc_coords)
                features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)
                features_scaled = scaler.transform(features.reshape(1, -1))
                pred = model.predict(features_scaled)[0]
                
                # Apply bounds checking
                pred = np.clip(pred, 0, 1000)  # Reasonable bounds for precipitation
                predictions[year_idx, i, j] = pred
    
    return future_years, predictions

def save_predictions(predictions, years, coordinates, transform, crs, output_dir='predictions'):
    """Save predictions as GeoTIFF files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each year's prediction as a GeoTIFF
    for year_idx, year in enumerate(years):
        output_file = os.path.join(output_dir, f'precipitation_prediction_{year}.tif')
        
        with rasterio.open(
            output_file,
            'w',
            driver='GTiff',
            height=predictions[year_idx].shape[0],
            width=predictions[year_idx].shape[1],
            count=1,
            dtype=predictions[year_idx].dtype,
            crs=crs,
            transform=transform
        ) as dst:
            dst.write(predictions[year_idx], 1)
        
        logging.info(f'Prediction for {year} saved to {output_file}')
    
    # Save summary statistics
    summary_file = os.path.join(output_dir, 'prediction_summary.csv')
    summary_data = []
    for year_idx, year in enumerate(years):
        year_pred = predictions[year_idx]
        summary_data.append({
            'Year': year,
            'Mean_Precipitation': np.mean(year_pred),
            'Std_Precipitation': np.std(year_pred),
            'Min_Precipitation': np.min(year_pred),
            'Max_Precipitation': np.max(year_pred)
        })
    
    pd.DataFrame(summary_data).to_csv(summary_file, index=False)
    logging.info(f'Summary statistics saved to {summary_file}')

def main():
    # Load historical data with spatial information
    historical_data, historical_years, coordinates, transform, crs = load_historical_data()
    
    # Print data statistics for validation
    logging.info("\nHistorical Data Statistics:")
    logging.info(f"Data shape: {historical_data.shape}")
    logging.info(f"Mean precipitation: {np.mean(historical_data):.2f} mm")
    logging.info(f"Std precipitation: {np.std(historical_data):.2f} mm")
    logging.info(f"Min precipitation: {np.min(historical_data):.2f} mm")
    logging.info(f"Max precipitation: {np.max(historical_data):.2f} mm")
    
    # Prepare training data maintaining spatial relationships
    X, y, coord_scaler = prepare_training_data(historical_data, historical_years, coordinates)
    
    # Print training data statistics
    logging.info("\nTraining Data Statistics:")
    logging.info(f"X shape: {X.shape}")
    logging.info(f"y shape: {y.shape}")
    logging.info(f"Mean target: {np.mean(y):.2f} mm")
    logging.info(f"Std target: {np.std(y):.2f} mm")
    
    # Train model
    model, scaler = train_model(X, y)
    
    # Generate predictions for 2024-2030
    future_years, predictions = generate_predictions(model, scaler, coordinates, transform, crs, coord_scaler)
    
    # Save predictions
    save_predictions(predictions, future_years, coordinates, transform, crs)
    
    # Print summary
    logging.info('\nPrediction Summary:')
    for year_idx, year in enumerate(future_years):
        year_pred = predictions[year_idx]
        logging.info(f'Year {year}:')
        logging.info(f'  Mean: {np.mean(year_pred):.2f} mm')
        logging.info(f'  Std: {np.std(year_pred):.2f} mm')
        logging.info(f'  Min: {np.min(year_pred):.2f} mm')
        logging.info(f'  Max: {np.max(year_pred):.2f} mm')
    
    # Save model and scalers for future use
    model_dir = 'models/precipitation'
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, 'precipitation_prediction_model.joblib'))
    joblib.dump(scaler, os.path.join(model_dir, 'precipitation_scaler.joblib'))
    joblib.dump(coord_scaler, os.path.join(model_dir, 'coordinate_scaler.joblib'))
    logging.info('Model and scalers saved for future use')

if __name__ == '__main__':
    main() 