import rasterio
import numpy as np
import os

def examine_tif_file(file_path):
    print(f"\nExamining file: {file_path}")
    with rasterio.open(file_path) as src:
        data = src.read(1)
        print(f"Data shape: {data.shape}")
        print(f"Data type: {data.dtype}")
        print(f"Metadata: {src.meta}")
        print(f"Min value: {np.min(data)}")
        print(f"Max value: {np.max(data)}")
        print(f"Mean value: {np.mean(data)}")
        print(f"Number of NaN: {np.isnan(data).sum()}")
        print(f"Number of inf: {np.isinf(data).sum()}")
        print(f"Unique values: {np.unique(data)}")
        
        # Print value distribution
        valid_data = data[~np.isnan(data) & ~np.isinf(data)]
        if len(valid_data) > 0:
            print("\nValue distribution:")
            percentiles = [0, 25, 50, 75, 100]
            for p in percentiles:
                print(f"{p}th percentile: {np.percentile(valid_data, p)}")

def main():
    data_dir = "Datasets_Hackathon/Climate_Precipitation_Data"
    files = [f for f in os.listdir(data_dir) if f.endswith('.tif') and 'R.' in f]
    
    for file in sorted(files):
        file_path = os.path.join(data_dir, file)
        examine_tif_file(file_path)

if __name__ == "__main__":
    main() 