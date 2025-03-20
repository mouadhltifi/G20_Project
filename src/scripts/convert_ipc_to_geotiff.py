import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
import warnings

# Input and output paths
ipc_csv_path = 'cadre_harmonise_caf_ipc_assaba_monthly_grouped.csv'
admin_layers_dir = 'data/Datasets_Hackathon/Admin_layers'
output_dir = 'data/Datasets_Hackathon/IPC_Data'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load IPC data
ipc_data = pd.read_csv(ipc_csv_path)

# Load district boundaries
districts = gpd.read_file(os.path.join(admin_layers_dir, 'Assaba_Districts_layer.shp'))

# Create a mapping between CSV district names and shapefile district names
district_name_mapping = {
    'Barkeol': 'Barkéol',
    'Guerrou': 'Guerou',
    'Kankossa': 'Kankoussa',
    'Kiffa': 'Kiffa',
    'Boumdeid': 'Boumdeid'
}

# Get reference raster information from precipitation data
with rasterio.open('data/Datasets_Hackathon/Climate_Precipitation_Data/2023R.tif') as ref_src:
    ref_transform = ref_src.transform
    ref_crs = ref_src.crs
    ref_shape = ref_src.shape
    ref_bounds = ref_src.bounds

# Process data year by year
for year in range(2014, 2024):
    print(f"Processing year {year}...")
    
    # Filter data for the current year and valid district names
    year_data = ipc_data[
        (pd.to_datetime(ipc_data['reference_date']).dt.year == year) &
        (ipc_data['adm2_name'] != 'unknown') &
        (ipc_data['adm2_name'].notna())
    ]
    
    if len(year_data) == 0:
        print(f"No data available for {year}, skipping...")
        continue
    
    # Create empty raster with nodata value
    raster = np.full(ref_shape, -9999, dtype=np.float32)
    
    # For each district in the data
    for district_csv in year_data['adm2_name'].unique():
        if district_csv not in district_name_mapping:
            warnings.warn(f"District {district_csv} not found in mapping, skipping...")
            continue
            
        district_shp = district_name_mapping[district_csv]
        
        # Get district geometry
        district_geom = districts[districts['ADM2_EN'] == district_shp]
        if len(district_geom) == 0:
            warnings.warn(f"District {district_shp} not found in shapefile, skipping...")
            continue
            
        # Get the most recent data for the district in this year
        district_data = year_data[year_data['adm2_name'] == district_csv]
        if len(district_data) == 0:
            continue
            
        latest_data = district_data.sort_values('reference_date').iloc[-1]
        
        # Get population in each phase
        phase_populations = [
            latest_data['phase1'],  # Phase 1
            latest_data['phase2'],  # Phase 2
            latest_data['phase3'],  # Phase 3
            latest_data['phase4']   # Phase 4
        ]
        
        # Find the phase with the highest population
        max_phase_idx = np.argmax(phase_populations)
        # Add 1 because phases are 1-based
        dominant_phase = max_phase_idx + 1
        
        # Print debug information
        print(f"District: {district_csv}")
        print(f"Populations by phase: {phase_populations}")
        print(f"Dominant phase: {dominant_phase}")
            
        # Burn the district into the raster with the IPC phase value
        shapes = [(geom, dominant_phase) for geom in district_geom.geometry]
        raster = features.rasterize(
            shapes=shapes,
            out_shape=ref_shape,
            transform=ref_transform,
            fill=-9999,
            out=raster,
            dtype=np.float32
        )
    
    # Save the raster
    output_path = os.path.join(output_dir, f"{year}IPC.tif")
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=ref_shape[0],
        width=ref_shape[1],
        count=1,
        dtype=np.float32,
        crs=ref_crs,
        transform=ref_transform,
        nodata=-9999
    ) as dst:
        dst.write(raster, 1)
    print(f"Saved {output_path}")

print("Processing complete!") 