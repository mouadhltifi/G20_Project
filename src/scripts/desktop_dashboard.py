import sys
import os
import numpy as np
import pandas as pd
import rasterio
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QHBoxLayout, QComboBox, QLabel, QPushButton,
                            QTextEdit, QSplitter, QFrame, QScrollArea, QTabWidget,
                            QSlider, QGridLayout, QSizePolicy)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QPalette, QColor, QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
import folium
from folium import plugins
import rasterio.mask
import geopandas as gpd
from datetime import datetime
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from shapely.geometry import box
from matplotlib.transforms import Affine2D
import google.generativeai as genai

class DashboardWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mauritania Environmental Dashboard")
        self.setGeometry(100, 100, 1600, 900)
        
        # Read API key from external file
        with open('api_key.txt', 'r') as file:
            self.gemini_api_key = file.read().strip()
        genai.configure(api_key=self.gemini_api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        self.chat = self.model.start_chat(history=[])
        
        # Set up the initial context for the AI
        self.setup_ai_context()
        
        # Initialize data
        self.precipitation_dir = "data/Datasets_Hackathon/Climate_Precipitation_Data"
        self.land_cover_dir = "data/Datasets_Hackathon/Modis_Land_Cover_Data"
        self.gpp_dir = "data/Datasets_Hackathon/MODIS_Gross_Primary_Production_GPP"
        self.admin_dir = "data/Datasets_Hackathon/Admin_layers"
        
        # Load admin layers
        self.region_layer = gpd.read_file(os.path.join(self.admin_dir, "Assaba_Region_layer.shp"))
        self.districts_layer = gpd.read_file(os.path.join(self.admin_dir, "Assaba_Districts_layer.shp"))
        
        # Get available years from all datasets
        precip_years = set([int(f.split('R')[0]) for f in os.listdir(self.precipitation_dir) 
                          if f.endswith('.tif') and 'R.' in f and 2011 <= int(f.split('R')[0]) <= 2023])
        lc_years = set([int(f.split('LCT')[0]) for f in os.listdir(self.land_cover_dir) 
                       if f.endswith('.tif') and 'LCT' in f and 2011 <= int(f.split('LCT')[0]) <= 2023])
        gpp_years = set([int(f.split('_GP')[0]) for f in os.listdir(self.gpp_dir) 
                        if f.endswith('.tif') and '_GP.' in f and 2011 <= int(f.split('_GP')[0]) <= 2023])
        
        # Use years that are available in all datasets
        self.years = sorted(list(precip_years & lc_years & gpp_years))
        print(f"Available years in all datasets: {self.years}")
        
        if not self.years:
            raise ValueError("No common years found across datasets!")
            
        self.districts = sorted(self.districts_layer['ADM3_EN'].unique().tolist())
        
        # Initialize data storage
        self.precipitation_data = {}
        self.land_cover_data = {}
        self.gpp_data = {}
        self.current_year = self.years[0]
        self.current_layer = "Precipitation"
        self.current_district = "All Districts"
        self.colorbar = None
        
        # Load data
        self.load_data()
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # Create splitter for main content and side panel
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Create main content area (left side)
        main_content = QWidget()
        main_content_layout = QVBoxLayout(main_content)
        
        # Create controls panel
        controls_panel = QFrame()
        controls_panel.setStyleSheet("""
            QFrame {
                background-color: #3b3b3b;
                border-radius: 10px;
                padding: 10px;
            }
            QLabel {
                color: white;
                font-weight: bold;
                font-size: 12px;
            }
            QComboBox {
                background-color: #4b4b4b;
                color: white;
                border: 1px solid #5b5b5b;
                border-radius: 5px;
                padding: 5px;
                min-width: 150px;
            }
            QComboBox:hover {
                background-color: #5b5b5b;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: none;
            }
            QSlider::groove:horizontal {
                border: 1px solid #5b5b5b;
                height: 8px;
                background: #4b4b4b;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #0d47a1;
                border: none;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #1565c0;
            }
            QSlider::sub-page:horizontal {
                background: #0d47a1;
                border-radius: 4px;
            }
        """)
        
        controls_layout = QHBoxLayout(controls_panel)
        
        # Year slider
        year_label = QLabel("Year:")
        self.year_slider = QSlider(Qt.Orientation.Horizontal)
        self.year_slider.setMinimum(self.years[0])
        self.year_slider.setMaximum(self.years[-1])
        self.year_slider.setValue(self.current_year)
        self.year_slider.valueChanged.connect(self.update_year)
        self.year_label = QLabel(str(self.current_year))
        self.year_label.setStyleSheet("color: white; min-width: 50px;")
        
        # Layer selection
        layer_label = QLabel("Layer:")
        self.layer_combo = QComboBox()
        self.layer_combo.addItems(["Precipitation", "Land Cover", "GPP"])
        self.layer_combo.currentTextChanged.connect(self.update_layer)
        
        # District selection
        district_label = QLabel("District:")
        self.district_combo = QComboBox()
        self.district_combo.addItem("All Districts")
        self.district_combo.addItems(self.districts)
        self.district_combo.currentTextChanged.connect(self.update_district)
        
        # Add controls to layout
        controls_layout.addWidget(year_label)
        controls_layout.addWidget(self.year_slider)
        controls_layout.addWidget(self.year_label)
        controls_layout.addWidget(layer_label)
        controls_layout.addWidget(self.layer_combo)
        controls_layout.addWidget(district_label)
        controls_layout.addWidget(self.district_combo)
        controls_layout.addStretch()
        
        main_content_layout.addWidget(controls_panel)
        
        # Create scrollable area for plots
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #2b2b2b;
            }
        """)
        
        # Create grid layout for plots
        plots_widget = QWidget()
        plots_layout = QHBoxLayout(plots_widget)  # Changed to horizontal layout
        
        # Create left side (map) and right side (plots) containers
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        
        # Create frames for each plot
        self.map_frame = self.create_plot_frame("Map Visualization")
        self.trend_frame = self.create_plot_frame("Trend Analysis")
        self.stats_frame = self.create_plot_frame("Statistics")
        
        # Add map to left container
        left_layout.addWidget(self.map_frame)
        
        # Add other plots to right container
        right_layout.addWidget(self.trend_frame)
        right_layout.addWidget(self.stats_frame)
        
        # Add containers to main layout
        plots_layout.addWidget(left_container)
        plots_layout.addWidget(right_container)
        
        # Set stretch factors (map takes 60%, plots take 40%)
        plots_layout.setStretch(0, 60)  # Left side (map)
        plots_layout.setStretch(1, 40)  # Right side (plots)
        
        scroll.setWidget(plots_widget)
        main_content_layout.addWidget(scroll)
        
        # Create side panel (right side)
        side_panel = QWidget()
        side_panel.setStyleSheet("""
            QWidget {
                background-color: #3b3b3b;
                border-radius: 10px;
            }
        """)
        side_layout = QVBoxLayout(side_panel)
        
        # Chat interface
        chat_frame = QFrame()
        chat_frame.setStyleSheet("""
            QFrame {
                background-color: #2b2b2b;
                border-radius: 10px;
                padding: 10px;
            }
            QLabel {
                color: white;
                font-weight: bold;
                font-size: 12px;
            }
            QTextEdit {
                background-color: #4b4b4b;
                color: white;
                border: 1px solid #5b5b5b;
                border-radius: 5px;
                padding: 5px;
            }
            QPushButton {
                background-color: #0d47a1;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 5px 10px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #1565c0;
            }
        """)
        chat_layout = QVBoxLayout(chat_frame)
        
        chat_label = QLabel("Chat Interface")
        chat_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_input = QTextEdit()
        self.chat_input.setMaximumHeight(60)
        self.chat_input.setPlaceholderText("Type your message here...")
        send_button = QPushButton("Send")
        send_button.clicked.connect(self.send_message)
        
        chat_layout.addWidget(chat_label)
        chat_layout.addWidget(self.chat_display)
        chat_layout.addWidget(self.chat_input)
        chat_layout.addWidget(send_button)
        
        # Statistics panel
        stats_frame = QFrame()
        stats_frame.setStyleSheet("""
            QFrame {
                background-color: #2b2b2b;
                border-radius: 10px;
                padding: 10px;
            }
            QLabel {
                color: white;
                font-weight: bold;
                font-size: 12px;
            }
            QTextEdit {
                background-color: #4b4b4b;
                color: white;
                border: 1px solid #5b5b5b;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        stats_layout = QVBoxLayout(stats_frame)
        
        stats_label = QLabel("Statistics")
        stats_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stats_display = QTextEdit()
        self.stats_display.setReadOnly(True)
        
        stats_layout.addWidget(stats_label)
        stats_layout.addWidget(self.stats_display)
        
        # Add panels to side layout
        side_layout.addWidget(chat_frame)
        side_layout.addWidget(stats_frame)
        
        # Add main content and side panel to splitter
        splitter.addWidget(main_content)
        splitter.addWidget(side_panel)
        
        # Set initial splitter sizes (70% main content, 30% side panel)
        splitter.setSizes([1120, 480])
        
        # Set dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
        """)
        
        # Initial updates
        self.update_visualizations()
        self.update_statistics()
        
    def create_plot_frame(self, title):
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background-color: #3b3b3b;
                border-radius: 10px;
                padding: 10px;
            }
            QLabel {
                color: white;
                font-weight: bold;
                font-size: 12px;
            }
        """)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(5, 5, 5, 5)  # Reduce margins
        layout.setSpacing(5)  # Reduce spacing
        
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        if title == "Map Visualization":
            # Create figure with larger size for the map
            canvas = FigureCanvas(plt.figure(figsize=(12, 8)))
            # Add navigation toolbar for map
            toolbar = NavigationToolbar2QT(canvas, frame)
            layout.addWidget(toolbar)
            canvas.setStyleSheet("background-color: transparent;")
            layout.addWidget(canvas)
        else:
            # Create figure that will adjust to container size
            canvas = FigureCanvas(plt.figure(figsize=(6, 4)))
            canvas.setStyleSheet("background-color: transparent;")
            layout.addWidget(canvas)
            
            # Make the canvas expand to fill available space
            canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        return frame
        
    def load_data(self):
        # Load precipitation data and get the reference extent
        self.reference_bounds = None
        for year in self.years:
            file = f"{year}R.tif"
            if os.path.exists(os.path.join(self.precipitation_dir, file)):
                with rasterio.open(os.path.join(self.precipitation_dir, file)) as src:
                    self.precipitation_data[year] = src.read(1)
                    if year == self.years[0]:  # Store the transform and CRS from the first file
                        self.raster_transform = src.transform
                        self.raster_crs = src.crs
                        # Store the reference bounds
                        self.reference_bounds = src.bounds
        
        # Load land cover data
        for year in self.years:
            file = f"{year}LCT.tif"
            if os.path.exists(os.path.join(self.land_cover_dir, file)):
                with rasterio.open(os.path.join(self.land_cover_dir, file)) as src:
                    self.land_cover_data[year] = src.read(1)
        
        # Load GPP data
        for year in self.years:
            file = f"{year}_GP.tif"
            if os.path.exists(os.path.join(self.gpp_dir, file)):
                with rasterio.open(os.path.join(self.gpp_dir, file)) as src:
                    self.gpp_data[year] = src.read(1)
        
        # Print loaded data for debugging
        print(f"Loaded precipitation years: {sorted(self.precipitation_data.keys())}")
        print(f"Loaded land cover years: {sorted(self.land_cover_data.keys())}")
        print(f"Loaded GPP years: {sorted(self.gpp_data.keys())}")
        
        # Reproject admin layers to match raster CRS
        self.region_layer = self.region_layer.to_crs(self.raster_crs)
        self.districts_layer = self.districts_layer.to_crs(self.raster_crs)
    
    def update_year(self, year):
        self.current_year = year
        self.year_label.setText(str(year))
        self.update_visualizations()
        self.update_statistics()
    
    def update_layer(self, layer):
        self.current_layer = layer
        self.update_visualizations()
        self.update_statistics()
    
    def update_district(self, district):
        self.current_district = district
        self.update_visualizations()
        self.update_statistics()
    
    def update_visualizations(self):
        # Clear all plots
        for frame in [self.map_frame, self.trend_frame, self.stats_frame]:
            layout = frame.layout()
            # Find the canvas widget
            for i in range(layout.count()):
                widget = layout.itemAt(i).widget()
                if isinstance(widget, FigureCanvas):
                    widget.figure.clear()
                    break
        
        # Map visualization
        map_canvas = None
        for i in range(self.map_frame.layout().count()):
            widget = self.map_frame.layout().itemAt(i).widget()
            if isinstance(widget, FigureCanvas):
                map_canvas = widget
                break
        
        if map_canvas is None:
            return
            
        map_fig = map_canvas.figure
        map_ax = map_fig.add_subplot(111)
        
        # Get the data based on current layer
        if self.current_layer == "Precipitation":
            data = self.precipitation_data[self.current_year]
            cmap = plt.cm.Blues
            title = f"Precipitation (mm) - {self.current_year}"
            # Set fixed scale limits for precipitation (0-1000mm)
            vmin = 0
            vmax = 1000
        elif self.current_layer == "Land Cover":
            data = self.land_cover_data[self.current_year]
            cmap = plt.cm.tab20
            title = f"Land Cover - {self.current_year}"
            vmin = 0
            vmax = 20  # Assuming 20 land cover classes
        else:  # GPP
            data = self.gpp_data[self.current_year]
            # Create custom colormap for GPP (brown -> yellow -> green)
            colors = ['#8B4513', '#DAA520', '#90EE90', '#228B22']  # brown -> yellow -> light green -> forest green
            n_bins = 256
            custom_cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
            cmap = custom_cmap
            title = f"GPP (gC/m²/year) - {self.current_year}"
            # Set fixed scale limits for GPP (0-2000 gC/m²/year)
            vmin = 0
            vmax = 2000
        
        # Apply mask for nodata values
        data = np.ma.masked_invalid(data)
        
        # Plot the data
        im = map_ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, extent=[
            self.reference_bounds.left,
            self.reference_bounds.right,
            self.reference_bounds.bottom,
            self.reference_bounds.top
        ])
        
        # Add colorbar
        plt.colorbar(im, ax=map_ax, label=title.split(' - ')[0])
        
        # Transform the admin layers to match the raster's coordinate system
        region_layer_plot = self.region_layer.copy()
        districts_layer_plot = self.districts_layer.copy()
        
        # Add region boundary first (as background)
        region_layer_plot.boundary.plot(
            ax=map_ax,
            color='yellow',
            linewidth=2,
            label='Region Boundary',
            zorder=2
        )
        
        # Add district boundaries with better visibility
        if self.current_district != "All Districts":
            # Plot all districts in light gray first
            districts_layer_plot.boundary.plot(
                ax=map_ax,
                color='lightgray',
                linewidth=1,
                alpha=0.3,
                zorder=3
            )
            # Then highlight the selected district
            district_mask = districts_layer_plot['ADM3_EN'] == self.current_district
            if district_mask.any():
                districts_layer_plot[district_mask].boundary.plot(
                    ax=map_ax,
                    color='red',
                    linewidth=2,
                    label=self.current_district,
                    zorder=4
                )
                # Fill the selected district with semi-transparent red
                districts_layer_plot[district_mask].plot(
                    ax=map_ax,
                    color='red',
                    alpha=0.1,
                    zorder=3
                )
        else:
            # Plot all districts with better visibility
            districts_layer_plot.boundary.plot(
                ax=map_ax,
                color='white',
                linewidth=1.5,
                alpha=0.7,
                zorder=3
            )
        
        # Customize the plot
        map_ax.set_title(title, color='white', pad=20)
        map_ax.set_axis_off()
        map_ax.legend(loc='upper right', facecolor='#3b3b3b', edgecolor='white')
        
        # Set the extent to match the reference bounds
        map_ax.set_xlim(self.reference_bounds.left, self.reference_bounds.right)
        map_ax.set_ylim(self.reference_bounds.bottom, self.reference_bounds.top)
        
        # Store the current data for statistics
        self.current_data = data
        
        # Update other visualizations
        self.update_trend_plot(map_canvas)
        self.update_stats_plot(map_canvas)
    
    def update_trend_plot(self, map_canvas):
        trend_canvas = self.trend_frame.layout().itemAt(1).widget()
        trend_fig = trend_canvas.figure
        trend_ax = trend_fig.add_subplot(111)
        
        if self.current_layer == "Precipitation":
            data_dict = self.precipitation_data
            ylabel = "Precipitation (mm)"
        elif self.current_layer == "Land Cover":
            data_dict = self.land_cover_data
            ylabel = "Land Cover Class"
        else:  # GPP
            data_dict = self.gpp_data
            ylabel = "GPP (gC/m²/year)"
        
        years = sorted(data_dict.keys())
        means = [np.nanmean(data_dict[year]) for year in years]
        
        trend_ax.plot(years, means, 'o-', color='#0d47a1')
        trend_ax.set_title(f"Mean {self.current_layer} Over Time", color='white', fontsize=12, pad=15)
        trend_ax.set_xlabel("Year", color='white', fontsize=10)
        trend_ax.set_ylabel(ylabel, color='white', fontsize=10)
        trend_ax.tick_params(colors='white', labelsize=9)
        trend_ax.grid(True, color='#4b4b4b')
        trend_fig.patch.set_facecolor('#3b3b3b')
        
        # Adjust layout to fit container
        trend_fig.tight_layout()
        trend_canvas.draw()
    
    def update_stats_plot(self, map_canvas):
        stats_canvas = self.stats_frame.layout().itemAt(1).widget()
        stats_fig = stats_canvas.figure
        stats_ax = stats_fig.add_subplot(111)
        
        if self.current_layer == "Land Cover":
            data = self.current_data
            # Get unique classes and their counts
            unique_classes = np.unique(data[~data.mask])  # Exclude masked values
            counts = [np.sum(data == c) for c in unique_classes]
            total = np.sum(counts)
            percentages = (np.array(counts) / total) * 100
            
            # Sort by frequency
            sorted_idx = np.argsort(counts)[::-1]
            unique_classes = unique_classes[sorted_idx]
            counts = np.array(counts)[sorted_idx]
            percentages = percentages[sorted_idx]
            
            # Create bar plot
            bars = stats_ax.bar(range(len(unique_classes)), counts)
            stats_ax.set_title(f"Land Cover Class Distribution - {self.current_year}", color='white', fontsize=12, pad=15)
            stats_ax.set_xlabel("Land Cover Class", color='white', fontsize=10)
            stats_ax.set_ylabel("Count", color='white', fontsize=10)
            stats_ax.tick_params(colors='white', labelsize=9)
            
            # Add percentage labels on top of bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                stats_ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{percentages[i]:.1f}%',
                            ha='center', va='bottom', color='white', fontsize=8)
            
            # Set x-axis labels
            stats_ax.set_xticks(range(len(unique_classes)))
            stats_ax.set_xticklabels([f'Class {int(c)}' for c in unique_classes], rotation=45, color='white', fontsize=9)
        else:
            data = self.current_data.compressed()  # Get non-masked values
            
            # Create histogram
            stats_ax.hist(data, bins=50, color='#0d47a1', alpha=0.7)
            stats_ax.axvline(np.nanmean(data), color='red', linestyle='dashed', linewidth=2, label='Mean')
            stats_ax.axvline(np.nanmedian(data), color='green', linestyle='dashed', linewidth=2, label='Median')
            stats_ax.set_title(f"{self.current_layer} Distribution - {self.current_year}", color='white', fontsize=12, pad=15)
            stats_ax.set_xlabel(self.current_layer, color='white', fontsize=10)
            stats_ax.set_ylabel("Frequency", color='white', fontsize=10)
            stats_ax.tick_params(colors='white', labelsize=9)
            stats_ax.legend(fontsize=9)
        
        stats_fig.patch.set_facecolor('#3b3b3b')
        
        # Adjust layout to fit container
        stats_fig.tight_layout()
        stats_canvas.draw()
    
    def update_statistics(self):
        if self.current_layer == "Precipitation":
            data = self.current_data
            unit = "mm"
        elif self.current_layer == "Land Cover":
            data = self.current_data
            unit = "class"
        else:  # GPP
            data = self.current_data
            unit = "gC/m²/year"
        
        stats_text = f"Statistics for {self.current_layer} ({self.current_year})\n"
        stats_text += f"Selected District: {self.current_district}\n\n"
        
        if self.current_layer == "Land Cover":
            unique_classes, counts = np.unique(data, return_counts=True)
            total = np.sum(counts)
            stats_text += "Class Distribution:\n"
            for cls, count in zip(unique_classes, counts):
                percentage = (count / total) * 100
                stats_text += f"Class {int(cls)}: {count} pixels ({percentage:.1f}%)\n"
        else:
            stats_text += f"Mean: {np.mean(data):.2f} {unit}\n"
            stats_text += f"Median: {np.median(data):.2f} {unit}\n"
            stats_text += f"Standard Deviation: {np.std(data):.2f} {unit}\n"
            stats_text += f"Minimum: {np.min(data):.2f} {unit}\n"
            stats_text += f"Maximum: {np.max(data):.2f} {unit}\n"
            stats_text += f"25th Percentile: {np.percentile(data, 25):.2f} {unit}\n"
            stats_text += f"75th Percentile: {np.percentile(data, 75):.2f} {unit}\n"
        
        self.stats_display.setText(stats_text)
    
    def setup_ai_context(self):
        """Set up the initial context for the AI assistant"""
        context = """You are an AI assistant specialized in environmental data analysis and visualization. 
        You have access to data about:
        1. Precipitation patterns
        2. Land cover changes
        3. Gross Primary Production (GPP)
        
        The data is from the Assaba region in Mauritania, spanning from 2011 to 2023.
        
        Your role is to:
        1. Help users understand the environmental data being displayed
        2. Provide insights about trends and patterns
        3. Answer questions about specific districts or time periods
        4. Explain the relationships between different environmental factors
        
        Always be concise and focused on the environmental aspects of the data."""
        
        self.chat.send_message(context)
    
    def send_message(self):
        message = self.chat_input.toPlainText()
        if message:
            # Display user message
            self.chat_display.append(f"You: {message}")
            self.chat_input.clear()
            
            try:
                # Prepare the prompt with context about current view
                current_context = f"""
                Current view context:
                - Layer: {self.current_layer}
                - Year: {self.current_year}
                - District: {self.current_district}
                
                User question: {message}
                
                Please provide a response focused on the environmental data and current view context.
                """
                
                # Get response from Gemini
                response = self.chat.send_message(current_context)
                
                # Display AI response
                self.chat_display.append(f"Assistant: {response.text}")
                
            except Exception as e:
                self.chat_display.append(f"Assistant: I apologize, but I encountered an error: {str(e)}")
                # Reset chat if there's an error
                self.chat = self.model.start_chat(history=[])
                self.setup_ai_context()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DashboardWindow()
    window.show()
    sys.exit(app.exec()) 