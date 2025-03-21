import sys
import os
import numpy as np
import pandas as pd
import rasterio
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QHBoxLayout, QComboBox, QLabel, QPushButton,
                            QTextEdit, QSplitter, QFrame, QScrollArea, QTabWidget,
                            QSlider, QGridLayout, QSizePolicy, QCheckBox)
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
import logging
from matplotlib.patches import Patch

from fpdf import FPDF

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
        self.infrastructure_dir = "data/Datasets_Hackathon/Streamwater_Line_Road_Network"
        self.ipc_dir = "data/Datasets_Hackathon/IPC_Data"
        self.prediction_dir = "predictions"  # Add prediction directory
        
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
        ipc_years = set([int(f.split('IPC')[0]) for f in os.listdir(self.ipc_dir)
                        if f.endswith('.tif') and 'IPC' in f])
        
        # Get prediction years
        self.prediction_years = []
        self.gpp_prediction_years = []
        self.precip_prediction_years = []
        if os.path.exists(self.prediction_dir):
            # Get GPP prediction years
            gpp_pred_years = set([int(f.split('_')[2].split('.')[0]) for f in os.listdir(self.prediction_dir)
                                if f.startswith('gpp_prediction_') and f.endswith('.tif')])
            self.gpp_prediction_years = sorted(list(gpp_pred_years))
            
            # Get precipitation prediction years
            precip_pred_years = set([int(f.split('_')[2].split('.')[0]) for f in os.listdir(self.prediction_dir)
                                   if f.startswith('precipitation_prediction_') and f.endswith('.tif')])
            self.precip_prediction_years = sorted(list(precip_pred_years))
            
            # Combined prediction years
            self.prediction_years = sorted(list(gpp_pred_years | precip_pred_years))
        
        # Use years that are available in all datasets
        self.years = sorted(list(precip_years & lc_years & gpp_years))
        self.ipc_years = sorted(list(ipc_years))  # Keep separate list for IPC years
        print(f"Available years in all datasets: {self.years}")
        print(f"Available IPC years: {self.ipc_years}")
        print(f"Available GPP prediction years: {self.gpp_prediction_years}")
        print(f"Available precipitation prediction years: {self.precip_prediction_years}")
        
        if not self.years:
            raise ValueError("No common years found across datasets!")
            
        self.districts = sorted(self.districts_layer['ADM3_EN'].unique().tolist())
        
        # Initialize data storage
        self.precipitation_data = {}
        self.land_cover_data = {}
        self.gpp_data = {}
        self.ipc_data = {}
        self.gpp_predictions = {}
        self.precipitation_predictions = {}  # Add precipitation predictions storage
        self.current_year = self.years[0]
        self.current_layer = "Precipitation"
        self.current_district = "All Districts"
        self.current_prediction_layer = "GPP"  # Add current prediction layer
        self.current_prediction_district = "All Districts"  # Add current prediction district
        self.colorbar = None
        
        # Load data
        self.load_data()
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # Create tab widget
        tab_widget = QTabWidget()
        tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: none;
                background-color: #2b2b2b;
            }
            QTabBar::tab {
                background-color: #3b3b3b;
                color: white;
                padding: 8px 16px;
                border: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #0d47a1;
            }
            QTabBar::tab:hover:!selected {
                background-color: #4b4b4b;
            }
        """)
        main_layout.addWidget(tab_widget)
        
        # Create Analysis tab
        analysis_tab = QWidget()
        analysis_layout = QHBoxLayout(analysis_tab)
        
        # Create splitter for main content and side panel in Analysis tab
        analysis_splitter = QSplitter(Qt.Orientation.Horizontal)
        analysis_layout.addWidget(analysis_splitter)
        
        # Create main content area (left side) for Analysis tab
        analysis_main_content = QWidget()
        analysis_main_content_layout = QVBoxLayout(analysis_main_content)
        
        # Create controls panel for Analysis tab
        analysis_controls_panel = QFrame()
        analysis_controls_panel.setStyleSheet("""
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
        
        analysis_controls_layout = QHBoxLayout(analysis_controls_panel)
        
        # Year slider for Analysis tab
        year_label = QLabel("Year:")
        self.year_slider = QSlider(Qt.Orientation.Horizontal)
        self.year_slider.setMinimum(self.years[0])
        self.year_slider.setMaximum(self.years[-1])
        self.year_slider.setValue(self.current_year)
        self.year_slider.valueChanged.connect(self.update_year)
        self.year_label = QLabel(str(self.current_year))
        self.year_label.setStyleSheet("color: white; min-width: 50px;")
        
        # Layer selection for Analysis tab
        layer_label = QLabel("Layer:")
        self.layer_combo = QComboBox()
        self.layer_combo.addItems(["Precipitation", "Land Cover", "GPP", "IPC"])
        self.layer_combo.currentTextChanged.connect(self.update_layer)
        
        # District selection for Analysis tab
        district_label = QLabel("District:")
        self.district_combo = QComboBox()
        self.district_combo.addItem("All Districts")
        self.district_combo.addItems(self.districts)
        self.district_combo.currentTextChanged.connect(self.update_district)
        
        # Add admin layers toggle checkboxes for Analysis tab
        admin_label = QLabel("Admin Layers:")
        self.show_region_check = QCheckBox("Region Boundary")
        self.show_districts_check = QCheckBox("District Boundaries")
        self.show_region_check.setChecked(True)
        self.show_districts_check.setChecked(True)
        self.show_region_check.stateChanged.connect(self.update_visualizations)
        self.show_districts_check.stateChanged.connect(self.update_visualizations)
        
        # Style the checkboxes for Analysis tab
        self.show_region_check.setStyleSheet("""
            QCheckBox {
                color: white;
                font-size: 12px;
            }
            QCheckBox::indicator {
                width: 15px;
                height: 15px;
            }
            QCheckBox::indicator:unchecked {
                border: 1px solid #5b5b5b;
                background-color: #4b4b4b;
                border-radius: 3px;
            }
            QCheckBox::indicator:checked {
                border: 1px solid #0d47a1;
                background-color: #0d47a1;
                border-radius: 3px;
            }
        """)
        self.show_districts_check.setStyleSheet(self.show_region_check.styleSheet())
        
        # Add infrastructure layers toggle checkboxes for Analysis tab
        infra_label = QLabel("Infrastructure:")
        self.show_streams_check = QCheckBox("Stream Water")
        self.show_roads_check = QCheckBox("Road Network")
        self.show_streams_check.setChecked(True)
        self.show_roads_check.setChecked(True)
        self.show_streams_check.stateChanged.connect(self.update_visualizations)
        self.show_roads_check.stateChanged.connect(self.update_visualizations)
        
        # Style the infrastructure checkboxes for Analysis tab
        infra_style = """
            QCheckBox {
                color: white;
                font-size: 12px;
            }
            QCheckBox::indicator {
                width: 15px;
                height: 15px;
            }
            QCheckBox::indicator:unchecked {
                border: 1px solid #5b5b5b;
                background-color: #4b4b4b;
                border-radius: 3px;
            }
            QCheckBox::indicator:checked {
                border: 1px solid #0d47a1;
                background-color: #0d47a1;
                border-radius: 3px;
            }
        """
        self.show_streams_check.setStyleSheet(infra_style)
        self.show_roads_check.setStyleSheet(infra_style)
        
        # Add controls to layout for Analysis tab
        analysis_controls_layout.addWidget(year_label)
        analysis_controls_layout.addWidget(self.year_slider)
        analysis_controls_layout.addWidget(self.year_label)
        analysis_controls_layout.addWidget(layer_label)
        analysis_controls_layout.addWidget(self.layer_combo)
        analysis_controls_layout.addWidget(district_label)
        analysis_controls_layout.addWidget(self.district_combo)
        analysis_controls_layout.addWidget(admin_label)
        analysis_controls_layout.addWidget(self.show_region_check)
        analysis_controls_layout.addWidget(self.show_districts_check)
        analysis_controls_layout.addWidget(infra_label)
        analysis_controls_layout.addWidget(self.show_streams_check)
        analysis_controls_layout.addWidget(self.show_roads_check)
        analysis_controls_layout.addStretch()
        
        analysis_main_content_layout.addWidget(analysis_controls_panel)
        
        # Create scrollable area for plots in Analysis tab
        analysis_scroll = QScrollArea()
        analysis_scroll.setWidgetResizable(True)
        analysis_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        analysis_scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #2b2b2b;
            }
        """)
        
        # Create grid layout for plots in Analysis tab
        analysis_plots_widget = QWidget()
        analysis_plots_layout = QHBoxLayout(analysis_plots_widget)
        
        # Create left side (map) and right side (plots) containers for Analysis tab
        analysis_left_container = QWidget()
        analysis_left_layout = QVBoxLayout(analysis_left_container)
        analysis_right_container = QWidget()
        analysis_right_layout = QVBoxLayout(analysis_right_container)
        
        # Create frames for each plot in Analysis tab
        self.map_frame = self.create_plot_frame("Map Visualization")
        self.trend_frame = self.create_plot_frame("Trend Analysis")
        self.stats_frame = self.create_plot_frame("Statistics")
        
        # Add map to left container in Analysis tab
        analysis_left_layout.addWidget(self.map_frame)
        
        # Add other plots to right container in Analysis tab
        analysis_right_layout.addWidget(self.trend_frame)
        analysis_right_layout.addWidget(self.stats_frame)
        
        # Add containers to main layout in Analysis tab
        analysis_plots_layout.addWidget(analysis_left_container)
        analysis_plots_layout.addWidget(analysis_right_container)
        
        # Set stretch factors (map takes 60%, plots take 40%) in Analysis tab
        analysis_plots_layout.setStretch(0, 60)  # Left side (map)
        analysis_plots_layout.setStretch(1, 40)  # Right side (plots)
        
        analysis_scroll.setWidget(analysis_plots_widget)
        analysis_main_content_layout.addWidget(analysis_scroll)
        
        # Create side panel (right side) for Analysis tab
        analysis_side_panel = QWidget()
        analysis_side_panel.setStyleSheet("""
            QWidget {
                background-color: #3b3b3b;
                border-radius: 10px;
            }
        """)
        analysis_side_layout = QVBoxLayout(analysis_side_panel)
        
        # Chat interface for Analysis tab
        analysis_chat_frame = QFrame()
        analysis_chat_frame.setStyleSheet("""
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
        analysis_chat_layout = QVBoxLayout(analysis_chat_frame)
        
        analysis_chat_label = QLabel("Chat Interface")
        analysis_chat_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_input = QTextEdit()
        self.chat_input.setMaximumHeight(60)
        self.chat_input.setPlaceholderText("Type your message here...")
        send_button = QPushButton("Send")
        send_button.clicked.connect(self.send_message)
        
        
        analysis_chat_layout.addWidget(analysis_chat_label)
        analysis_chat_layout.addWidget(self.chat_display)
        analysis_chat_layout.addWidget(self.chat_input)
        analysis_chat_layout.addWidget(send_button)

        
        # Statistics panel for Analysis tab
        analysis_stats_frame = QFrame()
        analysis_stats_frame.setStyleSheet("""
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
        analysis_stats_layout = QVBoxLayout(analysis_stats_frame)
        
        analysis_stats_label = QLabel("Statistics")
        analysis_stats_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stats_display = QTextEdit()
        self.stats_display.setReadOnly(True)
        
        analysis_stats_layout.addWidget(analysis_stats_label)
        analysis_stats_layout.addWidget(self.stats_display)
        
        # Add panels to side layout in Analysis tab
        analysis_side_layout.addWidget(analysis_chat_frame)
        analysis_side_layout.addWidget(analysis_stats_frame)
        
        # Add main content and side panel to splitter in Analysis tab
        analysis_splitter.addWidget(analysis_main_content)
        analysis_splitter.addWidget(analysis_side_panel)

        # Create report button for Analysis tab
        report_button = QPushButton("Generate Report")
        report_button.clicked.connect(self.generate_report)

        # Add report button to main layout in Analysis tab
        analysis_side_layout.addWidget(report_button)
        
        # Set initial splitter sizes (70% main content, 30% side panel) in Analysis tab
        analysis_splitter.setSizes([1120, 480])
        
        # Create Prediction tab (similar structure to Analysis tab)
        prediction_tab = QWidget()
        prediction_layout = QHBoxLayout(prediction_tab)
        
        # Create splitter for main content and side panel in Prediction tab
        prediction_splitter = QSplitter(Qt.Orientation.Horizontal)
        prediction_layout.addWidget(prediction_splitter)
        
        # Create main content area (left side) for Prediction tab
        prediction_main_content = QWidget()
        prediction_main_content_layout = QVBoxLayout(prediction_main_content)
        
        # Create controls panel for Prediction tab
        prediction_controls_panel = QFrame()
        prediction_controls_panel.setStyleSheet(analysis_controls_panel.styleSheet())
        prediction_controls_layout = QHBoxLayout(prediction_controls_panel)
        
        # Layer selection for Prediction tab
        prediction_layer_label = QLabel("Layer:")
        self.prediction_layer_combo = QComboBox()
        self.prediction_layer_combo.addItems(["GPP", "Precipitation"])
        self.prediction_layer_combo.currentTextChanged.connect(self.update_prediction_layer)
        
        # Year slider for Prediction tab
        prediction_year_label = QLabel("Year:")
        self.prediction_year_slider = QSlider(Qt.Orientation.Horizontal)
        if self.prediction_years:
            self.prediction_year_slider.setMinimum(min(self.prediction_years))
            self.prediction_year_slider.setMaximum(max(self.prediction_years))
            self.prediction_year_slider.setValue(min(self.prediction_years))
            self.prediction_year_slider.valueChanged.connect(self.update_prediction_year)
        self.prediction_year_label = QLabel(str(min(self.prediction_years)) if self.prediction_years else "No predictions")
        self.prediction_year_label.setStyleSheet("color: white; min-width: 50px;")
        
        # District selection for Prediction tab
        prediction_district_label = QLabel("District:")
        self.prediction_district_combo = QComboBox()
        self.prediction_district_combo.addItem("All Districts")
        self.prediction_district_combo.addItems(self.districts)
        self.prediction_district_combo.currentTextChanged.connect(self.update_prediction_district)
        
        # Add admin layers toggle checkboxes for Prediction tab
        prediction_admin_label = QLabel("Admin Layers:")
        self.prediction_show_region_check = QCheckBox("Region Boundary")
        self.prediction_show_districts_check = QCheckBox("District Boundaries")
        self.prediction_show_region_check.setChecked(True)
        self.prediction_show_districts_check.setChecked(True)
        self.prediction_show_region_check.stateChanged.connect(self.update_prediction_visualization)
        self.prediction_show_districts_check.stateChanged.connect(self.update_prediction_visualization)
        
        # Style the checkboxes for Prediction tab
        self.prediction_show_region_check.setStyleSheet(self.show_region_check.styleSheet())
        self.prediction_show_districts_check.setStyleSheet(self.show_districts_check.styleSheet())
        
        # Add infrastructure layers toggle checkboxes for Prediction tab
        prediction_infra_label = QLabel("Infrastructure:")
        self.prediction_show_streams_check = QCheckBox("Stream Water")
        self.prediction_show_roads_check = QCheckBox("Road Network")
        self.prediction_show_streams_check.setChecked(True)
        self.prediction_show_roads_check.setChecked(True)
        self.prediction_show_streams_check.stateChanged.connect(self.update_prediction_visualization)
        self.prediction_show_roads_check.stateChanged.connect(self.update_prediction_visualization)
        
        # Style the infrastructure checkboxes for Prediction tab
        self.prediction_show_streams_check.setStyleSheet(infra_style)
        self.prediction_show_roads_check.setStyleSheet(infra_style)
        
        # Add controls to layout for Prediction tab
        prediction_controls_layout.addWidget(prediction_layer_label)
        prediction_controls_layout.addWidget(self.prediction_layer_combo)
        prediction_controls_layout.addWidget(prediction_year_label)
        prediction_controls_layout.addWidget(self.prediction_year_slider)
        prediction_controls_layout.addWidget(self.prediction_year_label)
        prediction_controls_layout.addWidget(prediction_district_label)
        prediction_controls_layout.addWidget(self.prediction_district_combo)
        prediction_controls_layout.addWidget(prediction_admin_label)
        prediction_controls_layout.addWidget(self.prediction_show_region_check)
        prediction_controls_layout.addWidget(self.prediction_show_districts_check)
        prediction_controls_layout.addWidget(prediction_infra_label)
        prediction_controls_layout.addWidget(self.prediction_show_streams_check)
        prediction_controls_layout.addWidget(self.prediction_show_roads_check)
        prediction_controls_layout.addStretch()
        
        prediction_main_content_layout.addWidget(prediction_controls_panel)
        
        # Create scrollable area for plots in Prediction tab
        prediction_scroll = QScrollArea()
        prediction_scroll.setWidgetResizable(True)
        prediction_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        prediction_scroll.setStyleSheet(analysis_scroll.styleSheet())
        
        # Create grid layout for plots in Prediction tab
        prediction_plots_widget = QWidget()
        prediction_plots_layout = QHBoxLayout(prediction_plots_widget)
        
        # Create left side (map) and right side (plots) containers for Prediction tab
        prediction_left_container = QWidget()
        prediction_left_layout = QVBoxLayout(prediction_left_container)
        prediction_right_container = QWidget()
        prediction_right_layout = QVBoxLayout(prediction_right_container)
        
        # Create frames for each plot in Prediction tab
        self.prediction_map_frame = self.create_plot_frame("Prediction Map")
        self.prediction_trend_frame = self.create_plot_frame("Prediction Trends")
        self.prediction_stats_frame = self.create_plot_frame("Prediction Statistics")
        
        # Add map to left container in Prediction tab
        prediction_left_layout.addWidget(self.prediction_map_frame)
        
        # Add other plots to right container in Prediction tab
        prediction_right_layout.addWidget(self.prediction_trend_frame)
        prediction_right_layout.addWidget(self.prediction_stats_frame)
        
        # Add containers to main layout in Prediction tab
        prediction_plots_layout.addWidget(prediction_left_container)
        prediction_plots_layout.addWidget(prediction_right_container)
        
        # Set stretch factors (map takes 60%, plots take 40%) in Prediction tab
        prediction_plots_layout.setStretch(0, 60)  # Left side (map)
        prediction_plots_layout.setStretch(1, 40)  # Right side (plots)
        
        prediction_scroll.setWidget(prediction_plots_widget)
        prediction_main_content_layout.addWidget(prediction_scroll)
        
        # Create side panel (right side) for Prediction tab
        prediction_side_panel = QWidget()
        prediction_side_panel.setStyleSheet(analysis_side_panel.styleSheet())
        prediction_side_layout = QVBoxLayout(prediction_side_panel)
        
        # Add placeholder panels for Prediction tab
        prediction_chat_frame = QFrame()
        prediction_chat_frame.setStyleSheet(analysis_chat_frame.styleSheet())
        prediction_chat_layout = QVBoxLayout(prediction_chat_frame)
        
        prediction_chat_label = QLabel("Prediction Chat Interface")
        prediction_chat_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        prediction_chat_layout.addWidget(prediction_chat_label)
        
        prediction_stats_frame = QFrame()
        prediction_stats_frame.setStyleSheet(analysis_stats_frame.styleSheet())
        prediction_stats_layout = QVBoxLayout(prediction_stats_frame)
        
        prediction_stats_label = QLabel("Prediction Statistics")
        prediction_stats_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        prediction_stats_layout.addWidget(prediction_stats_label)
        
        # Add panels to side layout in Prediction tab
        prediction_side_layout.addWidget(prediction_chat_frame)
        prediction_side_layout.addWidget(prediction_stats_frame)
        
        # Add main content and side panel to splitter in Prediction tab
        prediction_splitter.addWidget(prediction_main_content)
        prediction_splitter.addWidget(prediction_side_panel)
        
        # Set initial splitter sizes (70% main content, 30% side panel) in Prediction tab
        prediction_splitter.setSizes([1120, 480])
        
        # Add tabs to tab widget
        tab_widget.addTab(analysis_tab, "Analysis")
        tab_widget.addTab(prediction_tab, "Prediction")
        
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
        
        if title == "Map Visualization":
            # Create figure with larger size for the map
            canvas = FigureCanvas(plt.figure(figsize=(12, 8)))
            # Add navigation toolbar for map
            toolbar = NavigationToolbar2QT(canvas, frame)
            layout.addWidget(toolbar)
            canvas.setStyleSheet("background-color: transparent;")
            layout.addWidget(canvas)
            # Set the canvas to expand in both directions
            canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        else:
            # Add title for non-map frames
            title_label = QLabel(title)
            title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(title_label)
            
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
        
        # Load IPC data
        for year in self.ipc_years:
            file = f"{year}IPC.tif"
            if os.path.exists(os.path.join(self.ipc_dir, file)):
                with rasterio.open(os.path.join(self.ipc_dir, file)) as src:
                    self.ipc_data[year] = src.read(1)
        
        # Load GPP predictions
        for year in self.gpp_prediction_years:
            file = f"gpp_prediction_{year}.tif"
            if os.path.exists(os.path.join(self.prediction_dir, file)):
                with rasterio.open(os.path.join(self.prediction_dir, file)) as src:
                    self.gpp_predictions[year] = src.read(1)
        
        # Load precipitation predictions
        for year in self.precip_prediction_years:
            file = f"precipitation_prediction_{year}.tif"
            if os.path.exists(os.path.join(self.prediction_dir, file)):
                with rasterio.open(os.path.join(self.prediction_dir, file)) as src:
                    self.precipitation_predictions[year] = src.read(1)
        
        # Load infrastructure layers
        try:
            # Load and print CRS information first
            print(f"Raster CRS: {self.raster_crs}")
            
            # Load infrastructure layers
            self.streams_layer = gpd.read_file(os.path.join(self.infrastructure_dir, "Streamwater.shp"))
            self.roads_layer = gpd.read_file(os.path.join(self.infrastructure_dir, "Main_Road.shp"))
            
            print(f"Original roads CRS: {self.roads_layer.crs}")
            print(f"Original streams CRS: {self.streams_layer.crs}")
            
            # Ensure we're using the same CRS as the raster
            if self.roads_layer.crs != self.raster_crs:
                self.roads_layer = self.roads_layer.to_crs(self.raster_crs)
            if self.streams_layer.crs != self.raster_crs:
                self.streams_layer = self.streams_layer.to_crs(self.raster_crs)
            
            print(f"Final roads CRS: {self.roads_layer.crs}")
            print(f"Final streams CRS: {self.streams_layer.crs}")
            
            # Print bounds information
            print(f"Raster bounds: {self.reference_bounds}")
            print(f"Roads bounds: {self.roads_layer.total_bounds}")
            print(f"Streams bounds: {self.streams_layer.total_bounds}")
            
            logging.info("Successfully loaded infrastructure layers")
        except Exception as e:
            logging.error(f"Error loading infrastructure layers: {str(e)}")
            # Create empty GeoDataFrames as fallback
            self.streams_layer = gpd.GeoDataFrame()
            self.roads_layer = gpd.GeoDataFrame()
        
        # Reproject admin layers to match raster CRS
        self.region_layer = self.region_layer.to_crs(self.raster_crs)
        self.districts_layer = self.districts_layer.to_crs(self.raster_crs)
        
        # Print loaded data for debugging
        print(f"Loaded precipitation years: {sorted(self.precipitation_data.keys())}")
        print(f"Loaded land cover years: {sorted(self.land_cover_data.keys())}")
        print(f"Loaded GPP years: {sorted(self.gpp_data.keys())}")
    
    def update_year(self, year):
        self.current_year = year
        self.year_label.setText(str(year))
        self.update_visualizations()
        self.update_statistics()
    
    def update_layer(self, layer):
        self.current_layer = layer
        # Update year slider range based on selected layer
        if layer == "IPC":
            self.year_slider.setMinimum(min(self.ipc_years))
            self.year_slider.setMaximum(max(self.ipc_years))
            if self.current_year not in self.ipc_years:
                self.current_year = min(self.ipc_years)
                self.year_slider.setValue(self.current_year)
                self.year_label.setText(str(self.current_year))
        else:
            self.year_slider.setMinimum(min(self.years))
            self.year_slider.setMaximum(max(self.years))
            if self.current_year not in self.years:
                self.current_year = min(self.years)
                self.year_slider.setValue(self.current_year)
                self.year_label.setText(str(self.current_year))
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
            vmin = 0
            vmax = 600
        elif self.current_layer == "Land Cover":
            data = self.land_cover_data[self.current_year]
            cmap = plt.cm.tab10
            title = f"Land Cover - {self.current_year}"
            vmin = 7
            vmax = 16
        elif self.current_layer == "GPP":
            data = self.gpp_data[self.current_year]
            colors = ['#8B4513', '#DAA520', '#90EE90', '#228B22']
            n_bins = 256
            custom_cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
            cmap = custom_cmap
            title = f"GPP (gC/m²/year) - {self.current_year}"
            vmin = 200
            vmax = 60000
        else:  # IPC
            data = self.ipc_data[self.current_year]
            # Create custom colormap for IPC phases with more distinct colors
            colors = ['#FFFFFF',  # white for no data
                     '#92D050',  # green for Phase 1 (Minimal)
                     '#FFEB3B',  # yellow for Phase 2 (Stressed)
                     '#FF9800',  # orange for Phase 3 (Crisis)
                     '#FF0000']  # red for Phase 4 (Emergency)
            custom_cmap = LinearSegmentedColormap.from_list('ipc', colors, N=5)
            cmap = custom_cmap
            title = f"IPC Phase Classification - {self.current_year}"
            vmin = 0  # 0 for no data
            vmax = 4  # max phase is 4
            
            # Create custom legend for IPC phases with better visibility
            legend_elements = [
                Patch(facecolor='#92D050', edgecolor='black', label='Phase 1: Minimal'),
                Patch(facecolor='#FFEB3B', edgecolor='black', label='Phase 2: Stressed'),
                Patch(facecolor='#FF9800', edgecolor='black', label='Phase 3: Crisis'),
                Patch(facecolor='#FF0000', edgecolor='black', label='Phase 4: Emergency')
            ]
        
        # Apply mask for nodata values
        data = np.ma.masked_where(data == -9999, data)
        
        # Plot the data
        im = map_ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, extent=[
            self.reference_bounds.left,
            self.reference_bounds.right,
            self.reference_bounds.bottom,
            self.reference_bounds.top
        ])
        
        # Transform the admin layers to match the raster's coordinate system
        region_layer_plot = self.region_layer.copy()
        districts_layer_plot = self.districts_layer.copy()
        
        # Add stream water if enabled
        if self.show_streams_check.isChecked() and not self.streams_layer.empty:
            self.streams_layer.plot(
                ax=map_ax,
                color='blue',
                linewidth=1,
                alpha=0.6,
                zorder=2
            )

        # Add road network if enabled
        if self.show_roads_check.isChecked() and not self.roads_layer.empty:
            self.roads_layer.plot(
                ax=map_ax,
                color='orange',
                linewidth=1.5,
                alpha=0.8,
                zorder=2
            )
            
            # Add road network statistics to the plot
            road_lengths = self.roads_layer.geometry.length
            stats_text = f"Road Network Stats:\n"
            stats_text += f"Total Length: {road_lengths.sum()/1000:.1f} km\n"
            stats_text += f"Segments: {len(self.roads_layer)}"
            map_ax.text(0.02, 0.02, stats_text,
                       transform=map_ax.transAxes,
                       color='white',
                       fontsize=8,
                       bbox=dict(facecolor='#3b3b3b', alpha=0.7, edgecolor='none'),
                       zorder=6)
        
        # Add region boundary if enabled
        if self.show_region_check.isChecked():
            region_layer_plot.boundary.plot(
                ax=map_ax,
                color='yellow',
                linewidth=2,
                zorder=3
            )
        
        # Add district boundaries if enabled
        if self.show_districts_check.isChecked():
            if self.current_district != "All Districts":
                # Plot all districts in light gray first
                districts_layer_plot.boundary.plot(
                    ax=map_ax,
                    color='lightgray',
                    linewidth=1,
                    alpha=0.3,
                    zorder=4
                )
                # Then highlight the selected district
                district_mask = districts_layer_plot['ADM3_EN'] == self.current_district
                if district_mask.any():
                    districts_layer_plot[district_mask].boundary.plot(
                        ax=map_ax,
                        color='red',
                        linewidth=2,
                        zorder=5
                    )
                    # Fill the selected district with semi-transparent red
                    districts_layer_plot[district_mask].plot(
                        ax=map_ax,
                        color='red',
                        alpha=0.1,
                        zorder=4
                    )
            else:
                # Plot all districts with better visibility
                districts_layer_plot.boundary.plot(
                    ax=map_ax,
                    color='white',
                    linewidth=1.5,
                    alpha=0.7,
                    zorder=4
                )
        
        # Customize the plot
        map_ax.set_title(title, color='white', pad=20)
        map_ax.set_axis_off()
        
        # Add legend based on layer type
        if self.current_layer == "IPC":
            # Add custom legend for IPC phases with better positioning and style
            legend = map_ax.legend(
                handles=legend_elements,
                loc='upper left',  # Changed to upper left to avoid overlap
                bbox_to_anchor=(0.02, 0.98),  # Adjusted position
                title='IPC Classification',
                title_fontsize=12,
                fontsize=10,
                framealpha=0.8,
                edgecolor='white',
                facecolor='#3b3b3b'
            )
            legend.get_title().set_color('white')
            for text in legend.get_texts():
                text.set_color('white')
        else:
            # Add colorbar for non-IPC layers
            plt.colorbar(im, ax=map_ax, label=title.split(' - ')[0])
            
            # Add infrastructure and admin boundaries legend
            handles = []
            labels = []
            if self.show_streams_check.isChecked() and not self.streams_layer.empty:
                handles.append(plt.Line2D([0], [0], color='blue', linewidth=1))
                labels.append('Stream Water')
            if self.show_roads_check.isChecked() and not self.roads_layer.empty:
                handles.append(plt.Line2D([0], [0], color='orange', linewidth=1.5))
                labels.append('Road Network')
            if self.show_region_check.isChecked():
                handles.append(plt.Line2D([0], [0], color='yellow', linewidth=2))
                labels.append('Region Boundary')
            if self.show_districts_check.isChecked():
                handles.append(plt.Line2D([0], [0], color='white', linewidth=1.5))
                labels.append('District Boundaries')
            
            if handles:
                legend = map_ax.legend(
                    handles=handles,
                    labels=labels,
                    loc='upper right',
                    bbox_to_anchor=(0.98, 0.98),
                    title='Map Elements',
                    title_fontsize=10,
                    fontsize=8,
                    framealpha=0.8,
                    edgecolor='white',
                    facecolor='#3b3b3b'
                )
                legend.get_title().set_color('white')
                for text in legend.get_texts():
                    text.set_color('white')
        
        # Set the extent to match the reference bounds
        map_ax.set_xlim(self.reference_bounds.left, self.reference_bounds.right)
        map_ax.set_ylim(self.reference_bounds.bottom, self.reference_bounds.top)
        
        # Store the current data for statistics
        self.current_data = data
        
        # Update other visualizations
        self.update_trend_plot(map_canvas)
        self.update_stats_plot(map_canvas)
    
    def get_district_mask(self, data):
        """Get a mask for the selected district."""
        if self.current_district == "All Districts":
            return np.ones_like(data, dtype=bool)
        
        # Get the selected district geometry
        district_geom = self.districts_layer[self.districts_layer['ADM3_EN'] == self.current_district].geometry.iloc[0]
        
        # Create an affine transform for the data
        transform = self.raster_transform
        
        # Create a raster mask using rasterio.mask
        height, width = data.shape
        mask = rasterio.features.geometry_mask(
            [district_geom],
            out_shape=data.shape,
            transform=transform,
            invert=True
        )
        
        return mask

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
        elif self.current_layer == "GPP":
            data_dict = self.gpp_data
            ylabel = "GPP (gC/m²/year)"
        else:  # IPC
            data_dict = self.ipc_data
            ylabel = "IPC Phase"
        
        years = sorted(data_dict.keys())
        means = []
        
        for year in years:
            data = data_dict[year]
            # Create masked array for no-data values (-9999)
            masked_array = np.ma.masked_where((data == -9999) | (data < 0), data)
            
            if self.current_district != "All Districts":
                # Get district mask and combine with no-data mask
                district_mask = self.get_district_mask(data)
                masked_array = np.ma.array(masked_array, mask=~district_mask | masked_array.mask)
            
            # Calculate mean of valid values only
            mean_value = masked_array.mean()
            if mean_value is not np.ma.masked:
                means.append(float(mean_value))
            else:
                means.append(np.nan)
        
        # Remove any NaN values
        valid_data = [(year, mean) for year, mean in zip(years, means) if not np.isnan(mean)]
        if valid_data:
            plot_years, plot_means = zip(*valid_data)
            trend_ax.plot(plot_years, plot_means, 'o-', color='#0d47a1')
            
        trend_ax.set_title(f"Mean {self.current_layer} Over Time - {self.current_district}", color='white', fontsize=12, pad=15)
        trend_ax.set_xlabel("Year", color='white', fontsize=10)
        trend_ax.set_ylabel(ylabel, color='white', fontsize=10)
        trend_ax.tick_params(colors='white', labelsize=9)
        trend_fig.patch.set_facecolor('#3b3b3b')
        
        # Adjust layout to fit container
        trend_fig.tight_layout()
        trend_canvas.draw()
    
    def update_stats_plot(self, map_canvas):
        stats_canvas = self.stats_frame.layout().itemAt(1).widget()
        stats_fig = stats_canvas.figure
        stats_ax = stats_fig.add_subplot(111)
        
        # Get district mask
        district_mask = self.get_district_mask(self.current_data)
        # Apply district mask to data
        masked_data = self.current_data[district_mask]
        
        if self.current_layer == "Land Cover":
            # Get unique classes and their counts for the masked data
            unique_classes = np.unique(masked_data[~np.ma.getmask(masked_data)])  # Exclude masked values
            counts = [np.sum(masked_data == c) for c in unique_classes]
            total = np.sum(counts)
            percentages = (np.array(counts) / total) * 100
            
            # Sort by frequency
            sorted_idx = np.argsort(counts)[::-1]
            unique_classes = unique_classes[sorted_idx]
            counts = np.array(counts)[sorted_idx]
            percentages = percentages[sorted_idx]
            
            # Create bar plot
            bars = stats_ax.bar(range(len(unique_classes)), counts)
            stats_ax.set_title(f"Land Cover Class Distribution - {self.current_year}\n{self.current_district}", 
                             color='white', fontsize=12, pad=15)
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
            # Get non-masked values for the selected district
            data = masked_data.compressed()  # Get non-masked values
            data = data[data > -1e+30]
            
            # Create histogram
            stats_ax.hist(data, bins=50, color='#0d47a1', alpha=0.7)
            stats_ax.axvline(np.nanmean(data), color='red', linestyle='dashed', linewidth=2, label='Mean')
            stats_ax.axvline(np.nanmedian(data), color='green', linestyle='dashed', linewidth=2, label='Median')
            stats_ax.set_title(f"{self.current_layer} Distribution - {self.current_year}\n{self.current_district}", 
                             color='white', fontsize=12, pad=15)
            stats_ax.set_xlabel(self.current_layer, color='white', fontsize=10)
            stats_ax.set_ylabel("Frequency", color='white', fontsize=10)
            stats_ax.tick_params(colors='white', labelsize=9)
            stats_ax.legend(fontsize=9)
        
        stats_fig.patch.set_facecolor('#3b3b3b')
        
        # Adjust layout to fit container
        stats_fig.tight_layout()
        stats_canvas.draw()
    
    def update_statistics(self):
        # Get district mask
        district_mask = self.get_district_mask(self.current_data)
        # Apply district mask to data
        masked_data = self.current_data[district_mask]
        
        if self.current_layer == "Precipitation":
            unit = "mm"
            stats_text = f"Statistics for {self.current_layer} ({self.current_year})\n"
            stats_text += f"Selected District: {self.current_district}\n\n"
            stats_text += f"Mean: {np.mean(masked_data):.2f} {unit}\n"
            stats_text += f"Median: {np.median(masked_data):.2f} {unit}\n"
            stats_text += f"Standard Deviation: {np.std(masked_data):.2f} {unit}\n"
            stats_text += f"Minimum: {np.min(masked_data):.2f} {unit}\n"
            stats_text += f"Maximum: {np.max(masked_data):.2f} {unit}\n"
            stats_text += f"25th Percentile: {np.percentile(masked_data, 25):.2f} {unit}\n"
            stats_text += f"75th Percentile: {np.percentile(masked_data, 75):.2f} {unit}\n"
        elif self.current_layer == "Land Cover":
            unit = "class"
            stats_text = f"Statistics for {self.current_layer} ({self.current_year})\n"
            stats_text += f"Selected District: {self.current_district}\n\n"
            unique_classes, counts = np.unique(masked_data[~np.ma.getmask(masked_data)], return_counts=True)
            total = np.sum(counts)
            stats_text += "Class Distribution:\n"
            for cls, count in zip(unique_classes, counts):
                percentage = (count / total) * 100
                stats_text += f"Class {int(cls)}: {count} pixels ({percentage:.1f}%)\n"
        elif self.current_layer == "GPP":
            unit = "gC/m²/year"
            stats_text = f"Statistics for {self.current_layer} ({self.current_year})\n"
            stats_text += f"Selected District: {self.current_district}\n\n"
            stats_text += f"Mean: {np.mean(masked_data):.2f} {unit}\n"
            stats_text += f"Median: {np.median(masked_data):.2f} {unit}\n"
            stats_text += f"Standard Deviation: {np.std(masked_data):.2f} {unit}\n"
            stats_text += f"Minimum: {np.min(masked_data):.2f} {unit}\n"
            stats_text += f"Maximum: {np.max(masked_data):.2f} {unit}\n"
            stats_text += f"25th Percentile: {np.percentile(masked_data, 25):.2f} {unit}\n"
            stats_text += f"75th Percentile: {np.percentile(masked_data, 75):.2f} {unit}\n"
        else:  # IPC
            stats_text = f"Statistics for {self.current_layer} ({self.current_year})\n"
            stats_text += f"Selected District: {self.current_district}\n\n"
            
            # Count pixels in each IPC phase for the selected district
            unique_phases, counts = np.unique(masked_data[~np.ma.getmask(masked_data)], return_counts=True)
            total = np.sum(counts)
            
            stats_text += "IPC Phase Distribution:\n"
            phase_names = {
                1: "Minimal",
                2: "Stressed",
                3: "Crisis",
                4: "Emergency"
            }
            
            for phase, count in zip(unique_phases, counts):
                if phase in phase_names:
                    percentage = (count / total) * 100
                    stats_text += f"Phase {int(phase)} ({phase_names[phase]}): {count} pixels ({percentage:.1f}%)\n"
        
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
    
    def update_prediction_layer(self, layer):
        """Update the prediction visualization when the layer changes."""
        self.current_prediction_layer = layer
        
        # Update year slider range based on selected layer
        if layer == "GPP" and self.gpp_prediction_years:
            self.prediction_year_slider.setMinimum(min(self.gpp_prediction_years))
            self.prediction_year_slider.setMaximum(max(self.gpp_prediction_years))
            if self.prediction_year_slider.value() not in self.gpp_prediction_years:
                self.prediction_year_slider.setValue(min(self.gpp_prediction_years))
        elif layer == "Precipitation" and self.precip_prediction_years:
            self.prediction_year_slider.setMinimum(min(self.precip_prediction_years))
            self.prediction_year_slider.setMaximum(max(self.precip_prediction_years))
            if self.prediction_year_slider.value() not in self.precip_prediction_years:
                self.prediction_year_slider.setValue(min(self.precip_prediction_years))
        
        self.update_prediction_visualization()
    
    def update_prediction_year(self, year):
        """Update the prediction visualization when the year changes."""
        if year in self.prediction_years:
            self.prediction_year_label.setText(str(year))
            self.update_prediction_visualization()
    
    def update_prediction_district(self, district):
        """Update the prediction visualization when the district changes."""
        self.current_prediction_district = district
        self.update_prediction_visualization()
    
    def update_prediction_visualization(self):
        """Update the prediction map and statistics."""
        # Clear prediction plots
        for frame in [self.prediction_map_frame, self.prediction_trend_frame, self.prediction_stats_frame]:
            layout = frame.layout()
            for i in range(layout.count()):
                widget = layout.itemAt(i).widget()
                if isinstance(widget, FigureCanvas):
                    widget.figure.clear()
                    break
        
        # Get prediction map canvas
        map_canvas = None
        for i in range(self.prediction_map_frame.layout().count()):
            widget = self.prediction_map_frame.layout().itemAt(i).widget()
            if isinstance(widget, FigureCanvas):
                map_canvas = widget
                break
        
        if map_canvas is None:
            return
        
        # Get current prediction year
        year = self.prediction_year_slider.value()
        
        # Create the prediction map
        map_fig = map_canvas.figure
        map_ax = map_fig.add_subplot(111)
        
        if self.current_prediction_layer == "GPP" and year in self.gpp_predictions:
            data = self.gpp_predictions[year]
            colors = ['#8B4513', '#DAA520', '#90EE90', '#228B22']
            n_bins = 256
            custom_cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
            cmap = custom_cmap
            title = f"Predicted GPP (gC/m²/year) - {year}"
            vmin = 200
            vmax = 60000
        elif self.current_prediction_layer == "Precipitation" and year in self.precipitation_predictions:
            data = self.precipitation_predictions[year]
            cmap = plt.cm.Blues
            title = f"Predicted Precipitation (mm) - {year}"
            vmin = 0
            vmax = 600
        else:
            return
        
        # Apply mask for nodata values
        data = np.ma.masked_where(data == -9999, data)
        
        # Plot the prediction data
        im = map_ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, extent=[
            self.reference_bounds.left,
            self.reference_bounds.right,
            self.reference_bounds.bottom,
            self.reference_bounds.top
        ])
        
        # Add colorbar
        plt.colorbar(im, ax=map_ax, label=title.split(' - ')[0])
        
        # Add admin boundaries
        if self.prediction_show_region_check.isChecked():
            self.region_layer.boundary.plot(
                ax=map_ax,
                color='yellow',
                linewidth=2,
                zorder=3
            )
        
        if self.prediction_show_districts_check.isChecked():
            if self.current_prediction_district != "All Districts":
                # Plot all districts in light gray first
                self.districts_layer.boundary.plot(
                    ax=map_ax,
                    color='lightgray',
                    linewidth=1,
                    alpha=0.3,
                    zorder=4
                )
                # Then highlight the selected district
                district_mask = self.districts_layer['ADM3_EN'] == self.current_prediction_district
                if district_mask.any():
                    self.districts_layer[district_mask].boundary.plot(
                        ax=map_ax,
                        color='red',
                        linewidth=2,
                        zorder=5
                    )
                    # Fill the selected district with semi-transparent red
                    self.districts_layer[district_mask].plot(
                        ax=map_ax,
                        color='red',
                        alpha=0.1,
                        zorder=4
                    )
            else:
                # Plot all districts with better visibility
                self.districts_layer.boundary.plot(
                    ax=map_ax,
                    color='white',
                    linewidth=1.5,
                    alpha=0.7,
                    zorder=4
                )
        
        # Add infrastructure layers
        if self.prediction_show_streams_check.isChecked() and not self.streams_layer.empty:
            self.streams_layer.plot(
                ax=map_ax,
                color='blue',
                linewidth=1,
                alpha=0.6,
                zorder=2
            )
        
        if self.prediction_show_roads_check.isChecked() and not self.roads_layer.empty:
            self.roads_layer.plot(
                ax=map_ax,
                color='orange',
                linewidth=1.5,
                alpha=0.8,
                zorder=2
            )
        
        # Customize the plot
        map_ax.set_title(title, color='white', pad=20)
        map_ax.set_axis_off()
        
        # Set the extent to match the reference bounds
        map_ax.set_xlim(self.reference_bounds.left, self.reference_bounds.right)
        map_ax.set_ylim(self.reference_bounds.bottom, self.reference_bounds.top)
        
        # Update trend plot
        self.update_prediction_trend_plot(map_canvas)
        
        # Update statistics plot
        self.update_prediction_stats_plot(map_canvas)
        
        map_canvas.draw()
    
    def update_prediction_trend_plot(self, map_canvas):
        """Update the prediction trend plot."""
        trend_canvas = self.prediction_trend_frame.layout().itemAt(1).widget()
        trend_fig = trend_canvas.figure
        trend_ax = trend_fig.add_subplot(111)
        
        if self.current_prediction_layer == "GPP":
            # Get historical and predicted means for GPP
            historical_years = sorted(self.gpp_data.keys())
            historical_means = [np.mean(self.gpp_data[year]) for year in historical_years]
            prediction_years = sorted(self.gpp_predictions.keys())
            prediction_means = [np.mean(self.gpp_predictions[year]) for year in prediction_years]
            ylabel = "Mean GPP (gC/m²/year)"
        else:  # Precipitation
            # Get historical and predicted means for precipitation
            historical_years = sorted(self.precipitation_data.keys())
            historical_means = [np.mean(self.precipitation_data[year]) for year in historical_years]
            prediction_years = sorted(self.precipitation_predictions.keys())
            prediction_means = [np.mean(self.precipitation_predictions[year]) for year in prediction_years]
            ylabel = "Mean Precipitation (mm)"
        
        # Plot historical data
        trend_ax.plot(historical_years, historical_means, 'o-', color='#0d47a1', label='Historical')
        
        # Plot predictions
        trend_ax.plot(prediction_years, prediction_means, 'o--', color='#ff9800', label='Predicted')
        
        trend_ax.set_title(f"{self.current_prediction_layer} Trend Analysis", color='white', fontsize=12, pad=15)
        trend_ax.set_xlabel("Year", color='white', fontsize=10)
        trend_ax.set_ylabel(ylabel, color='white', fontsize=10)
        trend_ax.tick_params(colors='white', labelsize=9)
        trend_fig.patch.set_facecolor('#3b3b3b')
        
        trend_fig.tight_layout()
        trend_canvas.draw()
    
    def update_prediction_stats_plot(self, map_canvas):
        """Update the prediction statistics plot."""
        stats_canvas = self.prediction_stats_frame.layout().itemAt(1).widget()
        stats_fig = stats_canvas.figure
        stats_ax = stats_fig.add_subplot(111)
        
        year = self.prediction_year_slider.value()
        if (self.current_prediction_layer == "GPP" and year in self.gpp_predictions) or \
           (self.current_prediction_layer == "Precipitation" and year in self.precipitation_predictions):
            
            if self.current_prediction_layer == "GPP":
                data = self.gpp_predictions[year]
                xlabel = "GPP (gC/m²/year)"
            else:
                data = self.precipitation_predictions[year]
                xlabel = "Precipitation (mm)"
            
            data = data[~np.isnan(data)]  # Remove NaN values
            
            # Create histogram
            stats_ax.hist(data, bins=50, color='#ff9800', alpha=0.7)
            stats_ax.axvline(np.mean(data), color='red', linestyle='dashed', linewidth=2, label='Mean')
            stats_ax.axvline(np.median(data), color='green', linestyle='dashed', linewidth=2, label='Median')
            
            stats_ax.set_title(f"{self.current_prediction_layer} Distribution - {year}", color='white', fontsize=12, pad=15)
            stats_ax.set_xlabel(xlabel, color='white', fontsize=10)
            stats_ax.set_ylabel("Frequency", color='white', fontsize=10)
            stats_ax.tick_params(colors='white', labelsize=9)
            stats_ax.legend(fontsize=9)
            
            stats_fig.patch.set_facecolor('#3b3b3b')
            stats_fig.tight_layout()
            stats_canvas.draw()

    def generate_report(self):

        # Save current plots as images to include in the report
        map_canvas = self.map_frame.layout().itemAt(1).widget()
        map_canvas.figure.savefig("map_plot.png", dpi=150)

        trend_canvas = self.trend_frame.layout().itemAt(1).widget()
        trend_canvas.figure.savefig("trend_plot.png", dpi=150)

        stats_canvas = self.stats_frame.layout().itemAt(1).widget()
        stats_canvas.figure.savefig("stats_plot.png", dpi=150)
        
        try:
            #Set Up context for the report
            context = f"""You are an AI assistant specialized in environmental data analysis and visualization. 
                You have access to data about:
                1. Precipitation patterns
                2. Land cover changes
                3. Gross Primary Production (GPP)
                
                The data is from the Assaba region in Mauritania, spanning from 2011 to 2023.

                Current view context:
                - Layer: {self.current_layer}
                - Year: {self.current_year}
                - District: {self.current_district}

                I'll also have the plots of this data as map_plot.png, trend_plot.png and stats_plot.png.

                Generate a page long report that highlights key trends, behaviours and patterns observed regarding the layer, year and district selected,
                providing critical insights into the extent and nature of changes and what is displayed 
                on the plots, reference the plots as needed.

                Respond only with the report text
                
                """

            # Get response from Gemini
            response = self.chat.send_message(context)

            # Write response in a file 
            with open("report.txt ", "w") as file:
                file.write(response.text)

            # Transform the text to a pdf file
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size = 12)
            with open("report.txt", "r") as file:
                for line in file:
                    pdf.cell(200, 10, txt = line, ln = True, align = 'L')
            pdf.output("report.pdf")

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