import numpy as np
import rasterio
import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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

# Check if GPU is available
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
logging.info(f'Using device: {device}')

class GPPDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features).to(device)
        self.targets = torch.FloatTensor(targets).to(device)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class GPPModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.model(x)

def load_historical_data(start_year=2011, end_year=2023):
    """Load historical GPP data with spatial information."""
    logging.info(f'Loading historical data from {start_year} to {end_year}...')
    data = []
    coordinates = None
    
    # Load first file to get spatial information
    first_file = f'data/Datasets_Hackathon/MODIS_Gross_Primary_Production_GPP/{start_year}_GP.tif'
    with rasterio.open(first_file) as src:
        transform = src.transform
        crs = src.crs
        height, width = src.shape
        
        # Create coordinate grid
        rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        lons, lats = rasterio.transform.xy(transform, rows, cols)
        coordinates = np.stack([lons, lats], axis=-1)
    
    # Load data for all years
    for year in tqdm(range(start_year, end_year + 1)):
        file_path = f'data/Datasets_Hackathon/MODIS_Gross_Primary_Production_GPP/{year}_GP.tif'
        with rasterio.open(file_path) as src:
            year_data = src.read(1).astype(np.float32)
            
            # Handle invalid values
            year_data = np.maximum(year_data, 0)
            valid_mask = np.isfinite(year_data)
            if np.any(valid_mask):
                mean_value = np.mean(year_data[valid_mask])
                year_data[~valid_mask] = mean_value
            else:
                year_data.fill(0)
            
            data.append(year_data)
    
    return np.array(data), np.array(range(start_year, end_year + 1)), coordinates, transform, crs

def prepare_features_for_location(year, coordinates):
    """Create features for a specific location and year."""
    features = [
        year,
        np.sin(2 * np.pi * year / 10),
        np.cos(2 * np.pi * year / 10),
        year ** 2,
        year ** 3,
        calendar.isleap(year),
        coordinates[0],
        coordinates[1],
        coordinates[0] * coordinates[1],
        coordinates[0]**2,
        coordinates[1]**2
    ]
    return np.array(features, dtype=np.float32)

def prepare_training_data(historical_data, historical_years, coordinates):
    """Prepare training data maintaining spatial relationships."""
    height, width = coordinates.shape[:2]
    
    # Normalize coordinates
    coord_scaler = MinMaxScaler()
    flat_coords = coordinates.reshape(-1, 2)
    coord_scaler.fit(flat_coords)
    normalized_coords = coord_scaler.transform(flat_coords).reshape(height, width, 2)
    
    # Prepare features and targets in batches
    batch_size = 10000
    X_batches = []
    y_batches = []
    
    for i in tqdm(range(0, height, batch_size), desc="Preparing training data"):
        i_end = min(i + batch_size, height)
        X_batch = []
        y_batch = []
        
        for j in range(width):
            for k in range(i, i_end):
                loc_coords = normalized_coords[k, j]
                for year_idx, year in enumerate(historical_years):
                    features = prepare_features_for_location(year, loc_coords)
                    X_batch.append(features)
                    y_batch.append(historical_data[year_idx, k, j])
        
        X_batches.append(np.array(X_batch, dtype=np.float32))
        y_batches.append(np.array(y_batch, dtype=np.float32))
    
    X = np.concatenate(X_batches)
    y = np.concatenate(y_batches)
    
    # Normalize features and targets
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    
    X = feature_scaler.fit_transform(X)
    y = target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
    
    # Ensure all values are finite
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    y = np.nan_to_num(y, nan=0, posinf=0, neginf=0)
    
    return X, y, coord_scaler, feature_scaler, target_scaler

def train_model(X, y, input_size):
    """Train a PyTorch model."""
    logging.info('Training model...')
    
    # Create dataset and dataloader
    dataset = GPPDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=2048, shuffle=True)
    
    # Initialize model
    model = GPPModel(input_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Training loop
    n_epochs = 100
    best_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    for epoch in tqdm(range(n_epochs), desc="Training epochs"):
        model.train()
        total_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            logging.info(f'Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.4f}')
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'models/gpp/best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f'Early stopping triggered after {epoch + 1} epochs')
                break
    
    # Load best model
    model.load_state_dict(torch.load('models/gpp/best_model.pth'))
    return model

def generate_predictions(model, coordinates, transform, crs, coord_scaler, feature_scaler, target_scaler, start_year=2024, end_year=2030):
    """Generate predictions for future years at each location."""
    logging.info(f'Generating predictions for {start_year}-{end_year}...')
    
    future_years = np.arange(start_year, end_year + 1)
    height, width = coordinates.shape[:2]
    n_years = len(future_years)
    predictions = np.zeros((n_years, height, width), dtype=np.float32)
    
    # Normalize coordinates
    flat_coords = coordinates.reshape(-1, 2)
    normalized_coords = coord_scaler.transform(flat_coords).reshape(height, width, 2)
    
    # Generate predictions in batches
    batch_size = 10000
    model.eval()
    
    with torch.no_grad():
        for i in tqdm(range(0, height, batch_size), desc="Generating predictions"):
            i_end = min(i + batch_size, height)
            for j in range(width):
                for k in range(i, i_end):
                    loc_coords = normalized_coords[k, j]
                    batch_features = []
                    
                    for year in future_years:
                        features = prepare_features_for_location(year, loc_coords)
                        batch_features.append(features)
                    
                    # Scale features
                    batch_features = feature_scaler.transform(batch_features)
                    batch_features = torch.FloatTensor(batch_features).to(device)
                    batch_pred = model(batch_features).cpu().numpy().flatten()
                    
                    # Inverse transform predictions
                    batch_pred = target_scaler.inverse_transform(batch_pred.reshape(-1, 1)).flatten()
                    
                    # Apply bounds checking based on historical data
                    batch_pred = np.clip(batch_pred, 0, 65535)
                    predictions[:, k, j] = batch_pred
    
    return future_years, predictions

def save_predictions(predictions, years, coordinates, transform, crs, output_dir='predictions'):
    """Save predictions as GeoTIFF files."""
    os.makedirs(output_dir, exist_ok=True)
    
    for year_idx, year in enumerate(years):
        output_file = os.path.join(output_dir, f'gpp_prediction_{year}.tif')
        
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
    
    summary_file = os.path.join(output_dir, 'gpp_prediction_summary.csv')
    summary_data = []
    for year_idx, year in enumerate(years):
        year_pred = predictions[year_idx]
        summary_data.append({
            'Year': year,
            'Mean_GPP': np.mean(year_pred),
            'Std_GPP': np.std(year_pred),
            'Min_GPP': np.min(year_pred),
            'Max_GPP': np.max(year_pred)
        })
    
    pd.DataFrame(summary_data).to_csv(summary_file, index=False)
    logging.info(f'Summary statistics saved to {summary_file}')

def main():
    # Load historical data
    historical_data, historical_years, coordinates, transform, crs = load_historical_data()
    
    # Print data statistics
    logging.info("\nHistorical Data Statistics:")
    logging.info(f"Data shape: {historical_data.shape}")
    logging.info(f"Mean GPP: {np.mean(historical_data):.2f} gC/m²/year")
    logging.info(f"Std GPP: {np.std(historical_data):.2f} gC/m²/year")
    logging.info(f"Min GPP: {np.min(historical_data):.2f} gC/m²/year")
    logging.info(f"Max GPP: {np.max(historical_data):.2f} gC/m²/year")
    
    # Prepare training data
    X, y, coord_scaler, feature_scaler, target_scaler = prepare_training_data(historical_data, historical_years, coordinates)
    
    # Print training data statistics
    logging.info("\nTraining Data Statistics:")
    logging.info(f"X shape: {X.shape}")
    logging.info(f"y shape: {y.shape}")
    logging.info(f"Mean target: {np.mean(y):.2f}")
    logging.info(f"Std target: {np.std(y):.2f}")
    
    # Train model
    model = train_model(X, y, input_size=X.shape[1])
    
    # Generate predictions
    future_years, predictions = generate_predictions(model, coordinates, transform, crs, coord_scaler, feature_scaler, target_scaler)
    
    # Save predictions
    save_predictions(predictions, future_years, coordinates, transform, crs)
    
    # Print summary
    logging.info('\nPrediction Summary:')
    for year_idx, year in enumerate(future_years):
        year_pred = predictions[year_idx]
        logging.info(f'Year {year}:')
        logging.info(f'  Mean: {np.mean(year_pred):.2f} gC/m²/year')
        logging.info(f'  Std: {np.std(year_pred):.2f} gC/m²/year')
        logging.info(f'  Min: {np.min(year_pred):.2f} gC/m²/year')
        logging.info(f'  Max: {np.max(year_pred):.2f} gC/m²/year')
    
    # Save model and scalers
    model_dir = 'models/gpp'
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, 'gpp_prediction_model.pth'))
    joblib.dump(coord_scaler, os.path.join(model_dir, 'gpp_coordinate_scaler.joblib'))
    joblib.dump(feature_scaler, os.path.join(model_dir, 'gpp_feature_scaler.joblib'))
    joblib.dump(target_scaler, os.path.join(model_dir, 'gpp_target_scaler.joblib'))
    logging.info('Model and scalers saved for future use')

if __name__ == '__main__':
    main() 