import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import logging
import rasterio
import pickle
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import xarray as xr
from scipy.ndimage import gaussian_filter
import calendar

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Log system information
logging.info('Starting precipitation model training...')
logging.info(f'Using {os.cpu_count()} CPU cores for parallel processing')

def load_tif_file(file_path):
    """Load a single TIF file and return its data with spatial information."""
    with rasterio.open(file_path) as src:
        data = src.read(1).astype(np.float32)  # Convert to float32
        # Handle NaN and infinite values
        data = np.nan_to_num(data, nan=np.nanmean(data), posinf=np.nanmax(data), neginf=np.nanmin(data))
        transform = src.transform
        crs = src.crs
        logging.debug(f'Loaded {file_path}: shape {data.shape}')
        return data, transform, crs

def create_spatial_features(data, transform):
    """Create spatial features from the precipitation data."""
    # Scale the data first to avoid overflow
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.reshape(-1, 1)).reshape(data.shape)
    
    # Calculate spatial statistics
    mean_precip = np.mean(data_scaled, dtype=np.float32)
    std_precip = np.std(data_scaled, dtype=np.float32)
    max_precip = np.max(data_scaled)
    min_precip = np.min(data_scaled)
    
    # Calculate terrain features using precipitation gradients
    gradient_y, gradient_x = np.gradient(data_scaled.astype(np.float32))
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    
    # Calculate smoothed precipitation
    smoothed = gaussian_filter(data_scaled.astype(np.float32), sigma=2)
    
    # Stack features and handle any remaining invalid values
    features = np.stack([
        data_scaled,  # Original precipitation
        gradient_magnitude,  # Terrain features
        smoothed,  # Smoothed precipitation
        np.full_like(data_scaled, mean_precip),  # Mean precipitation
        np.full_like(data_scaled, std_precip),  # Standard deviation
        np.full_like(data_scaled, max_precip),  # Maximum precipitation
        np.full_like(data_scaled, min_precip)   # Minimum precipitation
    ], axis=-1)
    
    # Handle any remaining invalid values
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features

def create_temporal_features(year, month):
    """Create temporal features from the year and month."""
    features = np.array([
        year,  # Original year
        month,  # Month
        np.sin(2 * np.pi * month / 12),  # Seasonal features
        np.cos(2 * np.pi * month / 12),
        np.sin(2 * np.pi * month / 6),
        np.cos(2 * np.pi * month / 6),
        year ** 2,  # Polynomial features
        year ** 3,
        calendar.monthrange(year, month)[1],  # Days in month
        month ** 2,  # Month squared
        month ** 3   # Month cubed
    ], dtype=np.float32)
    return features

# Load the precipitation dataset
logging.info('Loading the precipitation dataset...')
precipitation_data = []
transforms = []
crs_list = []
years = []
months = []

# Use ThreadPoolExecutor for parallel loading of TIF files
with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    # Create list of file paths and years
    file_paths = [f'data/Datasets_Hackathon/Climate_Precipitation_Data/{year}R.tif' 
                  for year in range(2011, 2024)]
    
    logging.info(f'Found {len(file_paths)} TIF files to process')
    
    # Load files in parallel with progress bar
    futures = [executor.submit(load_tif_file, path) for path in file_paths]
    for year, future in enumerate(tqdm(futures, desc="Loading TIF files", total=len(file_paths))):
        data, transform, crs = future.result()
        precipitation_data.append(data)
        transforms.append(transform)
        crs_list.append(crs)
        years.extend([year + 2011] * data.size)
        months.extend([1] * data.size)  # Assuming annual data

# Create features
logging.info('Creating spatial and temporal features...')
X_spatial = []
X_temporal = []
y = []

for i, (data, transform) in enumerate(zip(precipitation_data, transforms)):
    # Create spatial features
    spatial_features = create_spatial_features(data, transform)
    
    # Create temporal features for each pixel
    year = years[i * data.size]
    month = months[i * data.size]
    temporal_features = create_temporal_features(year, month)
    
    # Flatten spatial features while preserving pixel relationships
    X_spatial.append(spatial_features.reshape(-1, spatial_features.shape[-1]))
    X_temporal.extend([temporal_features] * data.size)
    y.extend(data.flatten())

X_spatial = np.vstack(X_spatial)
X_temporal = np.array(X_temporal)
y = np.array(y, dtype=np.float32)

# Combine features
X = np.hstack([X_spatial, X_temporal])

logging.info(f'Total data points loaded: {len(X)}')
logging.info(f'Feature shape: {X.shape}, Target shape: {y.shape}')

# Handle NaNs and clip extreme values
logging.info('Handling NaNs and clipping extreme values...')
y = np.nan_to_num(y, nan=np.nanmean(y))  # Replace NaNs with the mean
max_value = np.percentile(y, 99)  # Get the 99th percentile value
y = np.clip(y, None, max_value)  # Clip values above the 99th percentile

# Remove outliers using IQR method
logging.info('Removing outliers...')
Q1 = np.percentile(y, 25)
Q3 = np.percentile(y, 75)
IQR = Q3 - Q1
lower_bound = Q1 - 3 * IQR
upper_bound = Q3 + 3 * IQR
mask = (y >= lower_bound) & (y <= upper_bound)
y = y[mask]
X = X[mask]
logging.info(f'Data points after removing outliers: {len(X)}')

# Split the data into training and testing sets
logging.info('Splitting the data into training and testing sets...')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logging.info(f'Training set size: {len(X_train)}, Test set size: {len(X_test)}')

# Create preprocessing pipeline
logging.info('Setting up preprocessing pipeline...')
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), list(range(X.shape[1])))
    ])

# Define the model with hyperparameters to tune
model = RandomForestRegressor(
    random_state=42,
    n_estimators=100,  # Reduced number of trees for faster training
    max_depth=10,      # Reduced max depth
    min_samples_split=5,
    min_samples_leaf=2,
    n_jobs=-1
)

# Create pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', model)
])

# Define parameter grid for GridSearchCV
param_grid = {
    'regressor__n_estimators': [50, 100, 200],
    'regressor__max_depth': [5, 10, 15],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4]
}

# Perform GridSearchCV
logging.info('Performing GridSearchCV for hyperparameter tuning...')
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,  # Reduced number of folds for faster training
    n_jobs=-1,
    scoring='r2',
    verbose=1
)

grid_search.fit(X_train, y_train)
logging.info(f'Best parameters: {grid_search.best_params_}')
logging.info(f'Best cross-validation score: {grid_search.best_score_:.3f}')

# Get the best model
best_model = grid_search.best_estimator_

# Make predictions on the test set
logging.info('Making predictions on the test set...')
y_pred = best_model.predict(X_test)
logging.info('Predictions completed')

# Calculate the model's performance
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
logging.info('Model Performance Metrics:')
logging.info(f'- Mean Absolute Error: {mae:.3f}')
logging.info(f'- Root Mean Square Error: {rmse:.3f}')
logging.info(f'- R² Score: {r2:.3f}')

# Plot the results
logging.info('Generating visualization...')
plt.figure(figsize=(15, 10))

# Plot 1: Actual vs Predicted
plt.subplot(2, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Precipitation')
plt.ylabel('Predicted Precipitation')
plt.title('Actual vs Predicted Precipitation')

# Plot 2: Feature Importance
plt.subplot(2, 2, 2)
feature_importance = best_model.named_steps['regressor'].feature_importances_
feature_names = [f'Feature {i+1}' for i in range(len(feature_importance))]
plt.bar(feature_names, feature_importance)
plt.xticks(rotation=45)
plt.title('Feature Importance')
plt.tight_layout()

# Plot 3: Residuals
plt.subplot(2, 2, 3)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Precipitation')
plt.ylabel('Residuals')
plt.title('Residual Plot')

# Plot 4: Distribution of Errors
plt.subplot(2, 2, 4)
plt.hist(residuals, bins=50)
plt.xlabel('Residuals')
plt.ylabel('Count')
plt.title('Distribution of Errors')

plt.tight_layout()
plt.show()
logging.info('Visualization completed')

# Save the trained model and preprocessing pipeline
logging.info('Saving model and preprocessing pipeline...')
os.makedirs('models/precipitation', exist_ok=True)
joblib.dump(best_model, 'models/precipitation/precipitation_model.joblib')
logging.info('Model and pipeline saved successfully')
logging.info('Training process completed') 