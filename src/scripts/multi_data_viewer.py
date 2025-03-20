import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from matplotlib.widgets import Slider, RadioButtons

class MultiDataViewer:
    def __init__(self):
        # Set up the data directories
        self.base_dir = 'Datasets_Hackathon'
        self.datasets = {
            'Precipitation': {
                'dir': 'Climate_Precipitation_Data',
                'cmap': 'Blues',
                'unit': 'mm',
                'scale_factor': 1.0,
                'type': 'continuous',
                'vmin': 0,
                'vmax': 700  # Based on max observed value around 690
            },
            'GPP': {
                'dir': 'MODIS_Gross_Primary_Production_GPP',
                'cmap': 'viridis',  # Changed to viridis for better distinction of values
                'unit': 'gC/m²/year',
                'scale_factor': 0.0001,  # MODIS GPP scaling
                'type': 'continuous',
                'vmin': 0,
                'vmax': 3.5,  # Adjusted to show more detail in the common range
                'use_percentile': True  # New flag to use percentile-based scaling
            },
            'Land Cover': {
                'dir': 'Modis_Land_Cover_Data',
                'cmap': 'tab20',
                'unit': 'class',
                'scale_factor': 1.0,
                'type': 'categorical',
                'vmin': 7,  # Minimum observed class
                'vmax': 16,  # Maximum observed class
                'class_names': {
                    7: 'Open Shrublands',
                    10: 'Grasslands',
                    12: 'Croplands',
                    13: 'Urban and Built-up',
                    16: 'Barren or Sparsely Vegetated'
                }
            },
            'Population': {
                'dir': 'Gridded_Population_Density_Data',
                'cmap': 'YlOrRd',
                'unit': 'people/km²',
                'scale_factor': 1.0,
                'type': 'continuous',
                'vmin': 0,
                'vmax': None,  # Will be set dynamically
                'use_percentile': True,  # Use percentile-based scaling
                'percentile_min': 1,  # 1st percentile
                'percentile_max': 95  # 95th percentile
            }
        }
        
        # Create the figure and axis
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        
        # Initialize data
        self.current_year_idx = 0
        self.current_dataset = 'Precipitation'
        self.all_data = {}
        self.years = {}
        
        # Load all datasets
        self.load_all_data()
        
        # Create the slider
        self.slider_ax = plt.axes([0.2, 0.02, 0.6, 0.03])
        self.slider = Slider(
            self.slider_ax,
            'Year',
            0,
            len(self.years[self.current_dataset]) - 1,
            valinit=0,
            valstep=1
        )
        
        # Create radio buttons for dataset selection
        self.radio_ax = plt.axes([0.02, 0.02, 0.15, 0.15])
        self.radio = RadioButtons(
            self.radio_ax,
            list(self.datasets.keys()),
            active=0
        )
        
        # Create the colorbar
        self.im = self.ax.imshow(self.all_data[self.current_dataset][0], 
                                cmap=self.datasets[self.current_dataset]['cmap'],
                                norm='log' if self.current_dataset == 'Population' else None)
        self.cbar = plt.colorbar(self.im, ax=self.ax, 
                                label=f"{self.current_dataset} ({self.datasets[self.current_dataset]['unit']})")
        
        # Connect the events
        self.slider.on_changed(self.update)
        self.radio.on_clicked(self.change_dataset)
        
        # Initialize the plot
        self.update(0)
        
    def load_all_data(self):
        """Load and clean the data from all datasets"""
        print("Loading and cleaning data...")
        for dataset_name, dataset_info in self.datasets.items():
            data_dir = os.path.join(self.base_dir, dataset_info['dir'])
            print(f"\nChecking directory: {data_dir}")
            
            tif_files = sorted(glob.glob(os.path.join(data_dir, '*.tif')))
            print(f"Found {len(tif_files)} TIF files")
            
            if not tif_files:
                print(f"Warning: No TIF files found in {data_dir}")
                continue
            
            # Parse years based on dataset type
            if dataset_name == 'Land Cover':
                self.years[dataset_name] = [os.path.basename(f).split('LCT')[0] for f in tif_files]
            elif dataset_name == 'Population':
                # Extract year from filename, handling both Assaba and mrt formats
                self.years[dataset_name] = []
                for f in tif_files:
                    if 'Assaba' in f:
                        year = os.path.basename(f).split('_')[-1].split('.')[0]
                    else:
                        year = os.path.basename(f).split('_')[-2]
                    self.years[dataset_name].append(year)
            else:
                self.years[dataset_name] = [os.path.basename(f).split('R')[0].split('_')[0] for f in tif_files]
            
            print(f"Years found: {self.years[dataset_name]}")
            self.all_data[dataset_name] = []
            
            for tif_file in tif_files:
                print(f"Loading file: {os.path.basename(tif_file)}")
                with rasterio.open(tif_file) as src:
                    data = src.read(1)
                    print(f"Data shape: {data.shape}, Data type: {data.dtype}")
                    
                    # Apply scale factor
                    data = data * dataset_info['scale_factor']
                    
                    # Clean the data based on dataset type
                    if dataset_info['type'] == 'continuous':
                        # For continuous data, mask negative values and NoData
                        data = np.ma.masked_where(data < 0, data)
                        if hasattr(src, 'nodata') and src.nodata is not None:
                            data = np.ma.masked_where(data == src.nodata * dataset_info['scale_factor'], data)
                        
                        # For Population, handle zeros separately
                        if dataset_name == 'Population':
                            data = np.ma.masked_where(data == 0, data)
                    
                    elif dataset_info['type'] == 'categorical':
                        # For categorical data, mask values not in the class dictionary
                        valid_classes = list(dataset_info['class_names'].keys())
                        data = np.ma.masked_where(~np.isin(data, valid_classes), data)
                    
                    print(f"Valid data points: {np.sum(~data.mask)}")
                    self.all_data[dataset_name].append(data)
            
            print(f"Successfully loaded {len(tif_files)} files for {dataset_name}")
        
        print("\nAll data loaded successfully!")
        
    def update(self, val):
        """Update the visualization"""
        self.current_year_idx = int(self.slider.val)
        
        # Get the current year's data
        data = self.all_data[self.current_dataset][self.current_year_idx]
        
        # Update the image data
        self.im.set_data(data)
        
        # Set the color scale limits based on dataset type
        vmin = self.datasets[self.current_dataset]['vmin']
        vmax = self.datasets[self.current_dataset]['vmax']
        
        if self.current_dataset == 'Population':
            # For population, use percentile-based scaling
            valid_data = data[~data.mask] if isinstance(data, np.ma.MaskedArray) else data
            vmin = np.percentile(valid_data, self.datasets['Population']['percentile_min'])
            vmax = np.percentile(valid_data, self.datasets['Population']['percentile_max'])
            # Use linear scale instead of log
            self.im.set_norm(None)
        elif self.current_dataset == 'GPP' and self.datasets['GPP']['use_percentile']:
            # For GPP, use percentile-based scaling
            valid_data = data[~data.mask] if isinstance(data, np.ma.MaskedArray) else data
            vmin = np.percentile(valid_data, 1)  # 1st percentile to avoid outliers
            vmax = np.percentile(valid_data, 99)  # 99th percentile to avoid outliers
        
        self.im.set_clim(vmin=vmin, vmax=vmax)
        
        # Update the title
        year = self.years[self.current_dataset][self.current_year_idx]
        self.ax.set_title(f'{self.current_dataset} Map - {year}')
        
        # Update statistics (only for valid data)
        valid_data = data[~data.mask] if isinstance(data, np.ma.MaskedArray) else data
        if self.datasets[self.current_dataset]['type'] == 'categorical':
            # For land cover, show class distribution
            unique, counts = np.unique(valid_data, return_counts=True)
            stats_text = "Class Distribution:\n"
            for u, c in zip(unique, counts):
                class_name = self.datasets[self.current_dataset]['class_names'].get(int(u), f'Unknown Class {u}')
                stats_text += f"{class_name}: {c}\n"
        else:
            if self.current_dataset == 'Population':
                stats_text = f'Mean: {np.mean(valid_data):.1f}\n'
                stats_text += f'Median: {np.median(valid_data):.1f}\n'
                stats_text += f'Min: {np.min(valid_data):.1f}\n'
                stats_text += f'Max: {np.max(valid_data):.1f}\n'
                stats_text += f'1st percentile: {np.percentile(valid_data, 1):.1f}\n'
                stats_text += f'95th percentile: {np.percentile(valid_data, 95):.1f}'
            elif self.current_dataset == 'GPP':
                stats_text = f'Mean: {np.mean(valid_data):.3f}\n'
                stats_text += f'Median: {np.median(valid_data):.3f}\n'
                stats_text += f'1st percentile: {np.percentile(valid_data, 1):.3f}\n'
                stats_text += f'99th percentile: {np.percentile(valid_data, 99):.3f}\n'
                stats_text += f'Max: {np.max(valid_data):.3f}'
            else:
                stats_text = f'Mean: {np.mean(valid_data):.2f}\nStd: {np.std(valid_data):.2f}\n'
                stats_text += f'Min: {np.min(valid_data):.2f}\nMax: {np.max(valid_data):.2f}'
        
        # Remove old stats text if it exists
        for text in self.ax.texts:
            text.remove()
            
        # Add new stats text
        self.ax.text(0.02, 0.98, stats_text, 
                    transform=self.ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.draw()
        
    def change_dataset(self, label):
        """Change the current dataset"""
        self.current_dataset = label
        self.current_year_idx = 0
        self.slider.valmin = 0
        self.slider.valmax = len(self.years[label]) - 1
        self.slider.val = 0
        
        # Update the colorbar
        self.im.set_cmap(self.datasets[label]['cmap'])
        if label == 'Population':
            self.im.set_norm('log')
        else:
            self.im.set_norm(None)
        self.cbar.set_label(f"{label} ({self.datasets[label]['unit']})")
        
        # Update the plot
        self.update(0)

def main():
    viewer = MultiDataViewer()
    plt.show()

if __name__ == "__main__":
    main() 