import os
import glob
import shutil
import random
import inspect
import itertools
import matplotlib
import threading
import matplotlib.pyplot as plt
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.configs import ChatGPTConfig
from single_plot_generation_pipeline import *
from concurrent.futures import ThreadPoolExecutor, as_completed

matplotlib.use('Agg')

####################################################
# You need modify this part of your own API information
os.environ["OPENAI_BASE_URL"] = "base_url_of_openai"
os.environ["API_VERSION"] = "api_version"
os.environ["OPENAI_API_KEY"] = "your_openai_key"
os.environ["DEPLOYMENT_NAME"] = "gpt-4o"
####################################################

save_lock = threading.Lock()

chart_type_list = [
    'line', 'bar', 'pie', 'area', 'error_point', 'treemap', 'node', 'density',
    'histogram', 'box', 'bubble', 'candlestick', 'heatmap', 'radar', 'rose',
    'funnel', 'quiver', '3d', 'error_bar', 'scatter', 'violin', 'contour'
]

example_usages = {
    "plot_line_chart": [
        f"""
    plot_line_chart(
            data=[
                (['2020', '2021', '2022', '2023', '2024'], [1, 3, 2, 4, 5]),   # Line 1
                (['2020', '2021', '2022', '2023', '2024'], [2, 2, 3, 5, 6]),   # Line 2
                (['2020', '2021', '2022', '2023', '2024'], [3, 1, 4, 2, 5]),   # Line 3
                (['2020', '2021', '2022', '2023', '2024'], [4, 3, 2, 5, 7]),   # Line 4
                (['2020', '2021', '2022', '2023', '2024'], [1, 4, 3, 2, 1])    # Line 5
            ],
            title='Sample Line Chart with Multiple Lines',
            xlabel='Years',
            ylabel='Numbers',
            colors=['blue', 'red', 'green', 'orange', 'purple'],
            linestyles=['-', '--', '-.', ':', '-'],
            markers=['o', 's', '^', 'D', 'x'],
            linewidths=[2, 2, 2, 2, 2],
            grid=False,
            legend_labels=['Series A', 'Series B', 'Series C', 'Series D', 'Series E'],
            rotation=45,
            annotate_values=True,
            figsize=(12, 6)
        )
    """
    ],
    "plot_bar_chart": [
        f"""
    plot_bar_chart(
        data=[[3.4, 5.0, 2.1, 3.2], [4.3, 2.5, 5.4, 6.3], [3.6, 6.1, 4.0, 5.1],  [4.2, 3.0, 2.4, 1.5]], 
        orientation='horizontal',
        stacked=False,
        title='Employee Skill Levels',
        xlabel='Skills',
        ylabel='Rating (out of 10)',
        colors=['green', 'orange', 'blue'],
        linewidths=[1, 1, 1],
        grid=False,
        legend_labels=['Marketing Team', 'Development Team', 'Sales Team'], 
        label_rotation=0,
        group_label_rotation=0,
        group_labels=['Creativity', 'Problem Solving', 'Communication', 'Passion'],
        annotate_values=True,
        figsize=(8, 6)
    )
    """
    ],
    "plot_pie_chart": [
        f"""plot_pie_chart(
            data=[30, 20, 50],
            title='Sample Pie Chart',
            labels=['Category A', 'Category B', 'Category C'],
            colors=['blue', 'red', 'green'],
            explode=[0.3, 0, 0],
            startangle=90,
            shadow=False,
            autopct='%1.1f%%',
            ring=False,
            ring_width=0.3,
            show_legend=False,
            figsize=(8, 8)
        )"""
    ],
    "plot_area_chart": [
        f"""plot_area_chart(
            data=[[5, 7, 8, 6, 5],
                  [3, 5, 2, 4, 3]],
            title='Sample Stacked Area Chart',
            x_ticks=['A', 'B', 'C', 'D', 'E'],
            colors=['blue', 'orange'],
            alpha=0.4,
            linestyle='solid',  # Line style
            linewidth=2,        # Line width
            marker='o',         # Marker style
            figsize=(10, 6),
            rotation=45,
            y_label='Value',    # Y-axis label
            x_label='Categories', # X-axis label
            legend_labels=['Series A', 'Series B']  # Custom legend labels
        )
    """
    ],
    "plot_error_point_chart": [
        f"""
    plot_error_point_chart( 
        data=[[22.5, 23.5, 24.0], [55.0, 57.0, 60.0]], 
        pos_errors=[[0.5, 0.3, 0.4], [2.0, 1.5, 3.0]], 
        neg_errors=[[0.2, 0.3, 0.2], [1.0, 0.5, 2.0]],
        error_direction='vertical',
        color=['blue', 'orange'],
        marker='o',
        title='Temperature and Humidity Measurements with Errors',
        xlabel='Measurement Points',
        ylabel='Values',
        annotate=True,
        label_rotation=45,
        grid=True,
        legend_labels=['Temperature (°C)', 'Humidity (%)'],
        ticks=['Point 1', 'Point 2', 'Point 3'],
        figsize=(10, 6)
    )
    """
    ],
    "plot_treemap_chart": [
        f"""plot_treemap_chart(
        data=[300, 100, 200, 400],
        labels=['A', 'B', 'C', 'D'],
        colors=['red', 'blue', 'green', 'orange'],
        title='Sample Treemap Chart',
        show_values=True,
        border=True,
        pad=True,  # Adjust the padding for visibility
        font_size=12,  # Set the font size for labels
        font_color='white',  # Set the font color for labels
        figsize=(12, 8)
    )
    """
    ],
    "plot_funnel_chart": [
        f"""
    plot_funnel_chart(
                y=["Leads", "Prospects", "Qualified Leads", "Sales Meetings", "Closed Deals"],
                x=[1500, 800, 400, 250, 100],
                title="Sales Funnel Analysis for Q1 2024",
                textinfo="value+percent previous",
                textposition="inside",
                textfont=dict(size=12, color='#FFFFFF'),
                marker_color=['#ff5733', '#33ff57', '#3357ff', '#ff33a1'],
                marker_line=dict(color='#c70039', width=1),
                opacity=0.85,
                font=dict(family="Arial, sans-serif", size=14, color="#333333"),
                paper_bgcolor='rgba(255, 255, 255, 1)',
                plot_bgcolor='rgba(255, 255, 255, 1)',
                showgrid=True,
                gridwidth=1,
                gridcolor='LightGray',
                figsize=(1000, 800)
            )
    """
    ],
    "plot_node_chart": [
        f"""
    plot_node_chart(
            edges= [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (1, 6),
                (9, 0),  
                (0, 0),  
                (1, 1),   
                (5, 5),     
                (9, 9), 
            ],
            node_colors=['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'pink', 'brown'],  
            edge_colors=['black', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black'], 
            edge_labels=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], 
            node_labels=None, 
            edge_width=2,
            layout='circular', 
            title='Circular Graph with 10 Nodes of Different Colors',
            with_arrows=True,  
            self_loops=True,  
            figsize=(8, 8)
        )"""
    ],
    "plot_density_chart": [
        f"""
    plot_density_chart(data=np.concatenate([
                np.random.normal(loc=4, scale=1, size=1000), 
                np.random.normal(loc=10, scale=1.5, size=1000]
            ),
            labels=['Distribution 1'],
            colors=["purple"],  
            title='Bimodal Density Distribution',
            xlabel='Value',
            ylabel='Density',
            vertical=False,
            overlay=True,
            is_bimodal=True,
            grid=True,
            figsize=(10, 6))""", f"""
    plot_density_chart(
                data=[
                    np.random.normal(loc=2, scale=0.5, size=1000),  
                    np.random.normal(loc=5, scale=0.8, size=1000),  
                    np.random.normal(loc=8, scale=1.0, size=1000)
                ],
                labels=['Distribution 1', 'Distribution 2', 'Distribution 3'],
                colors=['blue', 'green', 'orange'],
                title='Multiple Unimodal Density Distributions',
                xlabel='Value',
                ylabel='Density',
                vertical=False,
                overlay=True,
                is_bimodal=False,
                grid=True,
                figsize=(10, 6)
            ) """
    ],
    "plot_histogram_chart": [
        f"""
    plot_histogram_chart(
            data_list=[
                np.random.normal(loc=0, scale=1, size=1000),  
                np.random.normal(loc=5, scale=1.5, size=1000), 
                np.random.normal(loc=10, scale=2, size=1000)  
            ],
            bins=30,
            labels=['Normal 1', 'Normal 2', 'Normal 3'],
            title='Comparison of Normal Distributions',
            xlabel='Frequency',
            ylabel='Value',
            colors=['blue', 'red', 'green'],
            border_color='black',
            alpha=0.7,
            grid=False,
            direction='horizontal',
            rotation=45,
            figsize=(10, 6)
        )
    """
    ],
    "plot_box_chart": [
        f"""
    plot_box_chart(
        data_list=[[1.5, 2.5, 3.5, 2, 2.8, 3.6, 3, 3.5, 4.0],
                    [5, 6, 7, 6.5, 7.5, 8.5, 7, 8, 8.5],
                    [3, 4, 2.5, 3.5, 4.5, 4.8, 5]],
        labels=['Math Scores', 'Science Scores', 'English Scores'],
        title='Distribution of Student Scores in Different Subjects',
        xlabel='Subjects',
        ylabel='Scores',
        colors=['lightblue', 'lightcoral', 'lightgreen'],
        median_color='darkred',
        hatch_styles=None,  
        widths=0.5,
        grid=False,
        orientation='horizontal',
        rotation=45,
        annotate_medians=True,
        outlier_settings={{  
            'points': [(1, 5.5), (2, 9), (3, 2), (1, 10), (2, 6.5), (3, 1)],
            'color': 'blue',
            'marker': 'o',
            'fill': False
        }},
        figsize=(10, 8)
    )"""
    ],
    "plot_bubble_chart": [
        f"""
    plot_bubble_chart(
            data=np.array([['A', 2.2], ['B', 3.1], ['C', 1.5], ['D', 4.5], ['E', 3.7], 
                        ['A', 5], ['B', 4], ['C', 4], ['D', 1], ['E', 2],
                        ['A', 1], ['B', 3], ['C', 5], ['D', 4], ['E', 5]]),
            sizes=np.array([204, 330, 900, 2900, 4000, 
                            1600, 112, 329, 1430, 146,
                            67, 83, 126, 5000, 67]),
            colors=None,
            labels=['Product A', 'Product B', 'Product C', 'Product D', 
                    'Product E', 'Product F', 'Product G', 'Product H', 
                    'Product I', 'Product J', 'Product K', 'Product L', 
                    'Product M', 'Product N', 'Product O'],
            title='Sales Volume by Product and Region',
            xlabel='Product Category',
            ylabel='Region',
            x_tick_angle=45,
            y_tick_angle=0,
            alpha=0.8,
            edge_color='black',
            color_map='viridis',  # Set to None since colors are provided
            show_legend=False,
            grid=True,
            legend_title='Products',  # Example legend title
            colorbar_label='Size Scale',  # Example colorbar label
            annotate=True,
            figsize=(12, 8)
        )
    """
    ],
    "plot_candlestick_chart": [
        f"""
    plot_candlestick_chart(
        dates=[
            '2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05',
            '2024-01-06', '2024-01-07', '2024-01-08', '2024-01-09', '2024-01-10'
        ],
        opens=[100, 102, 101, 103, 105, 104, 107, 109, 110, 108],
        highs=[102, 104, 103, 105, 106, 108, 109, 111, 111, 109],
        lows=[99, 100, 100, 101, 103, 102, 106, 107, 108, 107],
        closes=[101, 103, 102, 104, 105, 107, 108, 110, 109, 108],
        volumes= [1200, 1500, 1300, 1400, 1600, 1800, 1700, 1900, 2000, 1800],
        title='Sample Candlestick Chart',
        ylabel='Price',
        xlabel='Date',
        style='classic',
        enable_volume=True,
        x_ticks=['Jan 1', 'Jan 2', 'Jan 3', 'Jan 4', 'Jan 5', 
                  'Jan 6', 'Jan 7', 'Jan 8', 'Jan 9', 'Jan 10'],
        x_tick_rotation=45,
        colors={{'up': '#1f77b4', 'down': '#ff7f0e'}},  
        yaxis_range=[95, 115],  
        margin={{'l': 0.1, 'r': 0.9, 't': 0.9, 'b': 0.1}},
        grid=True, 
        figsize=(12, 8)
    )
    """
    ],
    "plot_heatmap_chart": [
        f"""
    plot_heatmap(
            data=np.array([[30.5, 45.2, 12.0, 20.5, 35.4, 50.3],
                        [25.3, 60.1, 5.0, 15.3, 40.2, 30.7],
                        [20.0, 35.5, 8.1, 25.8, 45.6, 60.0],
                        [75.4, 10.2, 20.5, 35.0, 50.1, 5.6],
                        [15.2, 25.5, 35.6, 50.2, 60.3, 12.1]]),
            title='Survey Results Heatmap',
            xlabel='Survey Options',
            ylabel='Respondent Groups',
            xticks = ["Option A", "Option B", "Option C", "Option D", "Option E", "Option F"],
            yticks = ["Group 1", "Group 2", "Group 3", "Group 4", "Group 5"],
            xtickangle=45,
            ytickangle=0,
            cmap='Blues',
            annot=False,
            fmt='.1f',
            linewidths=0.5,
            linecolor='black',
            figsize=(10, 8),
            colorbar=True,
            use_circles=False
        )
    """
    ],
    "plot_radar_chart": [
        f"""
    plot_radar_chart(
        data_list=[[58.2, 85.1, 50.7, 60.3], [60.5, 10.6, 55.1, 58.2],
                [62.4, 40.8, 52.9, 42.0]],
        labels=[
            'Fundraising Efforts (Million $)', 'Community Outreach (Score)',
            'Program Expenses (Million $)', 'Administrative Costs (Million $)'],  # length of labels should be 4, because the length of sublist in data_list is 4;
        title='Charity and Nonprofit Organizations Performance Analysis',
        colors=['blue', 'red', 'green'],
        alpha=0.5,
        fill=True,
        fill_colors=['lightblue', 'lightcoral', 'lightgreen'],
        border_styles=['solid', 'dashed', 'dotted'],
        border_shape='rectangle',
        legend_labels=['Hope Foundation', 'Helping Hands', 'Future Leaders'], # length of legend_labels should be 3, because the length of data_list is 3; 
        show_data_points=True,  
        figsize=(10, 10),
        show_grid=True
    )
    """
    ],
    "plot_rose_chart": [
        f"""
    plot_rose_chart(
        data_labels=['Electricity', 'Gasoline', 'Diesel', 'Jet Fuel', 'Coal'],
        data=[40, 25, 20, 10, 5],  
        title='Energy Consumption by Source', 
        colors=['#FF9999', '#66B3FF', '#99FF99', '#FFCC99', '#FFD700'],  
        edge_color='black',
        alpha=0.7,  
        figsize=(10, 8),
        save_path=None 
    )"""
    ],
    "plot_quiver_chart": [
        f"""   
    plot_quiver_chart(
            X=np.meshgrid(np.linspace(-2.0, 2.0, 20), np.linspace(-2.0, 2.0, 20))[0], 
            Y=np.meshgrid(np.linspace(-2.0, 2.0, 20), np.linspace(-2.0, 2.0, 20))[1],
            U=-np.meshgrid(np.linspace(-2.0, 2.0, 20), np.linspace(-2.0, 2.0, 20))[1], 
            V=np.meshgrid(np.linspace(-2.0, 2.0, 20), np.linspace(-2.0, 2.0, 20))[0],
            title="Wind Velocity Field",
            xlabel="Longitude",
            ylabel="Latitude",
            color_map="Blues",
            show_streamlines=False,
            streamline_color='black',
            legend_labels=None,
            grid=True,
            show_colorbar=True,
            colorbar_label='Velocity Magnitude',
            figsize=(10, 6)
        )""", f"""
    plot_quiver_chart(
            X=np.meshgrid(np.linspace(-2.0, 2.0, 20), np.linspace(-2.0, 2.0, 20))[0], 
            Y=np.meshgrid(np.linspace(-2.0, 2.0, 20), np.linspace(-2.0, 2.0, 20))[1],
            U=np.sin(np.meshgrid(np.linspace(-2.0, 2.0, 20), np.linspace(-2.0, 2.0, 20))[0]) * np.cos(np.meshgrid(np.linspace(-2.0, 2.0, 20), np.linspace(-2.0, 2.0, 20))[1]),
            V=-np.cos(np.meshgrid(np.linspace(-2.0, 2.0, 20), np.linspace(-2.0, 2.0, 20))[0]) * np.sin(np.meshgrid(np.linspace(-2.0, 2.0, 20), np.linspace(-2.0, 2.0, 20))[1]),
            title="Wind Speed and Direction", 
            xlabel="Longitude (degrees)", 
            ylabel="Latitude (degrees)" 
            color_map="coolwarm",
            show_streamlines=True,
            streamline_color='blue', 
            legend_labels=None,
            grid=False,
            show_colorbar=True,
            colorbar_label='Wind Speed',
            figsize=(10, 8)
        )""", f"""
    plot_quiver_chart(
            X=np.array([0, 0.5, 1, 1.5, 2, 2.5]), 
            Y=np.array([0, 1, 2, 1, 0, -1]), 
            U=np.array([1, 0.5, -0.5, 0.3, 0, -1]), 
            V=np.array([1, -1, 0.5, 0, -1, 1]),
            title="Quiver with Multiple Arrows and Adjusted Labels", 
            xlabel="X", 
            ylabel="Y", 
            color_map="plasma", 
            show_streamlines=False,  # can be False
            streamline_color=None,
            legend_labels=None, 
            show_colorbar=False,
            colorbar_label=False,    
            grid=True,               
            figsize=(8, 6)
        )"""
    ],
    "plot_3d_chart": [
        f"""
        plot_3d_chart(
        data=([['Product A', 'Product B', 'Product C', 'Product D', 'Product E'],   #**Must ensure the length of data_x should be equal to the length of data_z**
            ['2019', '2020', '2021', '2022'], #**Must ensure the length of data_y should be equal to the length of sublist in data_z**
            [[1200, 1500, 1800, 1300],
                [900, 1000, 1500, 1400],
                [500, 600, 700, 650],
                [2000, 2500, 2700, 2200],
                [800, 950, 1200, 1100]]]),  # Sales data for products over years
        plot_type='bar',
        labels=['Products', 'Years', 'Sales ($K)'],  # Updated axis labels
        title='Product Sales Over Years',
        colors=['lightblue', 'orange', 'green', 'red', 'purple'],
        bar_width=0.5,
        bar_alpha=0.8,  # Set transparency for bars
        elev=30,
        azim=45,
        tick_angle_x=0, 
        tick_angle_y=0,  
        figsize=(9, 8)
    )""", f"""
    plot_3d_chart(
        data=(
            [15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
            [45, 50, 55, 60, 65, 70, 75, 80, 85, 90],
            [1015, 1012, 1010, 1008, 1005, 1002, 1000, 999, 998, 997]   # can use random to generate
        ),
        plot_type='scatter',
        labels=['Temperature (°C)', 'Humidity (%)', 'Pressure (hPa)'],
        title='3D Scatter Plot of Environmental Factors Over Time',
        colors=['orange', 'green', 'blue', 'red', 'purple', 'cyan', 'magenta', 'brown', 'grey', 'yellow'],
        bar_width=0.6,
        elev=30,
        azim=45,
        tick_angle_x=45,
        tick_angle_y=45,
        scatter_marker='^',  # Specify marker style for scatter plot
        figsize=(10, 8)
    )""", f"""
   plot_3d_chart(
        data=(
            [0, 1, 2, 3, 4],
            [[5, 10, 15, 20, 25], 
            [4, 8, 12, 16, 20],
            [3, 6, 9, 12, 15]],
            [[2, 4, 6, 8, 10],
            [1, 2, 3, 4, 5],
            [3, 5, 7, 9, 11]]
        ),
        plot_type='line',
        labels=['Time (s)', 'Amplitude (m)', 'Height (m)'],
        title='3D Line Plot of Amplitude vs Time',
        colors=['red', 'green', 'blue'],
        bar_width=0.6,
        elev=30,
        azim=45,
        tick_angle_x=0,
        tick_angle_y=0,
        figsize=(9, 8)
    )""", f"""
    plot_3d_chart(
        data=(np.linspace(0, 5, 10), np.linspace(0, 5, 10), np.random.rand(10, 10)), 
        plot_type='surface',
        labels=['Temperature (°C)', 'Time (s)', 'Pressure (Pa)'],
        title='Temperature vs Time vs Pressure',
        colors=['viridis'], 
        bar_width=0.6,
        elev=30,
        azim=45,
        tick_angle_x=45,
        tick_angle_y=30,
        figsize=(9, 12)
    )""", f"""
    plot_3d_chart(
        data=(
            [0, 1, 2, 3, 4, 5],
            [0, 1, 2, 1, 2, 1],  
            [0, 1, 2, 1, 2, 3], 
            [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]  
        ),
        plot_type='skeleton',
        labels=['Time (s)', 'Temperature (°C)', 'Height (m)'],  
        title='3D Skeleton Plot of Temperature vs Time at Different Heights',
        colors=['blue'],  
        bar_width=0.6,
        elev=30,
        azim=45,
        tick_angle_x=60,
        tick_angle_y=0,
        figsize=(9, 8)
    )""", f"""
    plot_3d_chart(
        data=(np.linspace(-3, 3, 30), np.linspace(-3, 3, 30), 
            np.exp(-0.5 * (np.meshgrid(np.linspace(-3, 3, 30), 
            np.linspace(-3, 3, 30))[0]**2 + 
            np.meshgrid(np.linspace(-3, 3, 30), 
            np.linspace(-3, 3, 30))[1]**2))),
        plot_type='density',
        labels=['X Axis', 'Y Axis', 'Z Axis'],
        title='3D Gaussian Density Plot',
        colors=['viridis'],
        bar_width=0.6,
        elev=30,
        azim=45,
        tick_angle_x=0,
        tick_angle_y=60,
        figsize=(9, 8)
    )
    """
    ],
    "plot_error_bar_chart": [
        f"""
    plot_error_bar_chart(
        data_x=["ResNet50", "InceptionV3", "VGG16", "MobileNetV2", "EfficientNetB0"],
        data_y=[
            [60.3, 70.2, 72.4, 80.8, 90.1], 
        ],
        errors=[
            [4.5, 2.2, 6.1, 1.7, 1.2],  # Error for the accuracy metric
        ],
        bar_width=0.4,
        colors=["#add8e6", "#add8e6", "#add8e6", "#add8e6", "#add8e6"],  
        title="Comparison of Model Accuracies (Single Bar per Group)",
        xlabel="Models", 
        ylabel="Accuracy (%)",
        ylim=[0, 100],
        ticks=np.arange(0, 101, 10),
        xtick_angle=0,
        ytick_angle=0,
        grid=False,
        legend_labels="Ground-truth labels",  # Single legend label as a string
        figsize=(10, 5),
        orientation='horizontal' 
    )"""
    ],
    "plot_scatter_chart": [
        f"""
    plot_scatter_chart(
        data_list=[
            ([0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.6, 0.7, 0.8, 0.9]), 
            ([0.2, 0.3, 0.4, 0.5, 0.6], [1.5, 1.6, 1.7, 1.8, 1.9]), 
            ([0.3, 0.4, 0.5, 0.6, 0.7], [2.5, 2.6, 2.7, 2.8, 2.9])  
        ],
        labels=["Experiment Group 1", "Experiment Group 2", "Experiment Group 3"],
        xlabel="Measurement Frequency (Hz)", 
        ylabel="Response Amplitude (dB)",     
        title="Response Analysis Across Different Experiment Groups",  
        colors=["#FF6347", "#4682B4", "#6A5ACD"],
        markers=['o', 's', '^'],
        sizes=[50, 100, 75],
        alpha=0.8,
        legend=True,
        grid=True,
        x_tick_rotation=45,  # Set the rotation angle for x-axis tick labels
        point_labels=[
            ['A1', None, 'A3', None, 'A5'],  # Labels for points in Group 1
            [None, 'B2', None, 'B4', None],  # Labels for points in Group 2
            ['C1', 'C2', None, 'C4', None]   # Labels for points in Group 3
        ],
        figsize=(10, 6)
    )""", f"""
    plot_scatter_chart(
            data_list=[
                ([0.1], [1.0]),  
                ([0.2], [1.5]),  
                ([0.3], [1.2]),  
                ([0.4], [2.0]),  
                ([0.5], [2.5]),  
                ([0.6], [1.8]),  
                ([0.7], [2.2])   
            ],
            labels=None,
            xlabel="Feature Value",
            ylabel="Performance Score",
            title="Performance Analysis of Different Themes",
            colors=["#FF6347", "#4682B4", "#6A5ACD", "#32CD32", "#FFD700", "#FF69B4", "#8A2BE2"],
            markers=['o', 's', '^', 'D', 'x', 'P', 'v'],
            sizes=[100] * 7,
            alpha=0.8,
            legend=True,
            grid=True,
            x_tick_rotation=45,
            point_labels=None,
            figsize=(10, 6)
        )""", f"""
    plot_scatter_chart(
            data_list=[
                (np.random.normal(0.4, 0.05, 200), np.random.normal(0.3, 0.05, 200)),  
                (np.random.normal(-0.2, 0.05, 200), np.random.normal(0.1, 0.05, 200)), 
                (np.random.normal(-0.3, 0.05, 200), np.random.normal(-0.1, 0.05, 200)),  
                (np.random.normal(0.1, 0.05, 200), np.random.normal(0.2, 0.05, 200))  
            ],
            labels=["Group1", "Group2", "Group3", "Group4"],
            xlabel="PC1",
            ylabel="PC2",
            title="Clustered Scatter Plot",
            colors=["blue", "magenta", "yellow", "green"],
            markers=['o', 'o', 'o', 'o'],  # Same marker for all in clustered style
            sizes=[50, 50, 50, 50],  # Same size for all points
            alpha=0.5,
            legend=True,
            grid=True,
            x_tick_rotation=0,
            clustered=True,  # Enable clustered scatter style
            figsize=(8, 8)
        )
    """
    ],
    "plot_violin_chart": [
        f"""
    plot_violin_chart(
    data_list=[np.random.normal(loc=0, scale=1, size=100), np.random.normal(loc=1, scale=1.5, size=100), np.random.normal(loc=2, scale=1, size=100)],
    xticklabels=['Task 1', 'Task 2', 'Task 3'],
    xlabel="Tasks",
    ylabel='Values',
    ylim=[-3, 5],
    xticks=[1, 2, 3],
    title='Standard Violin Plot',
    colors=['#FF9999', '#66B3FF', '#99FF99'],
    edgecolor="#333333",
    linewidth=1.5,
    xlabel_rotation=0,
    show_grid=True,
    show_means=True,
    mean_label_angle=0,
    mean_line_visible=True,
    mean_offset=0.1,
    split=False,
    figsize=(8, 6)
)""", f"""
    plot_violin_chart(
    data_list=[np.random.beta(a=[3, 1, 4], b=[4, 3, 2], size=(1000, 3)), np.random.beta(a=[2, 3, 5], b=[1, 2, 4], size=(1000, 3))],
    xticklabels=["Productivity", "Satisfaction", "Quality"],
    xlabel="Metrics",
    ylabel="Performance Score",
    ylim=[0, 1],
    xticks=[1, 2, 3],
    title='Performance Comparison',
    colors=["#ff9999", "#66b3ff"],
    edgecolor="#4d4d4d",  # Darker edge color
    linewidth=2,  # Increased linewidth
    xlabel_rotation=30,
    show_grid=True,
    show_means=False,
    mean_label_angle=0,
    mean_line_visible=False,
    mean_offset=0.1,
    split=True,  
    custom_legends=["Low Memory", "High Memory"],  # Custom legends
    figsize=(10, 6)
)"""
    ],
    "plot_contour_chart": [
        f"""plot_contour_chart(
        X=None,  # Not used when contour_data is provided
        Y=None,  # Not used when contour_data is provided
        Z=None,  # Not used when contour_data is provided
        contour_type='lines',
        cmap=None,
        title='Multiple Contours Example',
        xlabel='SBP (mmHg)',
        ylabel='DBP (mmHg)',
        contour_data=[
        (np.meshgrid(np.linspace(90, 160, 100), np.linspace(50, 110, 100))[0], np.meshgrid(np.linspace(90, 160, 100), np.linspace(50, 110, 100))[1], np.exp(-((np.meshgrid(np.linspace(90, 160, 100), np.linspace(50, 110, 100))[0] - 125) ** 2 + (np.meshgrid(np.linspace(90, 160, 100), np.linspace(50, 110, 100))[1] - 80) ** 2) / 100), 'blue', 'Female'),
        (np.meshgrid(np.linspace(90, 160, 100), np.linspace(50, 110, 100))[0], np.meshgrid(np.linspace(90, 160, 100), np.linspace(50, 110, 100))[1], np.exp(-((np.meshgrid(np.linspace(90, 160, 100), np.linspace(50, 110, 100))[0] - 135) ** 2 + (np.meshgrid(np.linspace(90, 160, 100), np.linspace(50, 110, 100))[1] - 70) ** 2) / 100), 'red', 'Male')
        ],
        legend_labels=['City Temp', 'Country Temp'],
        alpha=0.7,
        levels=10,
        show_colorbar=None,
        colorbar_label=None,
        line_colors=['blue', 'red'],
        annotate_lines=True,
        figsize=(8, 8)
    )""", f"""
    plot_contour_chart(
            X=np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))[0],
            Y=np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))[1],
            Z=np.sqrt(np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))[0]**2 + np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))[1]**2),
            contour_type='filled',
            cmap='coolwarm',
            title='Filled Contour Plot',
            xlabel='X-axis',
            ylabel='Y-axis',
            contour_data=None,
            legend_labels=None,
            alpha=0.7,
            levels=10,
            show_colorbar=True,
            colorbar_label='Distance',
            line_colors=['blue', 'red'],
            annotate_lines=True,
            figsize=(10, 6)
        )"""
    ]
}

agent_single = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI,
    model_type=ModelType.GPT_4O,
    model_config_dict=ChatGPTConfig(temperature=1.0).as_dict(),
)

agent_combine = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI,
    model_type=ModelType.GPT_4O,
    model_config_dict=ChatGPTConfig(temperature=1.0).as_dict(),
)

agent_debug = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI,
    model_type=ModelType.GPT_4O,
    model_config_dict=ChatGPTConfig(temperature=0.0).as_dict(),
)

# Assistant 1: use for single plot data generation
assistant_sys_msg_for_single = BaseMessage.make_assistant_message(
    role_name="Assistant",
    content=f"""
### Intelligent Assistant Prompt: Input Synthesis Helper

You are a "Function Input Synthesis Helper," and your task is to generate an appropriate function call based on a provided function, filling the call with reasonable parameters.

**User-Provided Function:**
```python
{{user_function}}
```

**Synthesized Function Call Example:**
```python
{{function_name}}(
    {{parameters_and_values}}
)
```
    
**Important Considerations:**
--Ensure that data_list contains realistic data with variability and noise.
--The pattern for parameters should reflect potential trends, cycles, randomness, and correlations.
--Data should exhibit complexity and noise, avoiding simple linear or polynomial.
--Make sure the elements in data_list match labels.
--Parameters such as colors_list, linestyles_list, and markers_list should match the structure and amount of data in data_list.
--Each parameter should follow a unique potential pattern.
--Users may provide data themes and constraints on certain parameters, and data should be generated accordingly.
--Data values should make sense and fall within appropriate ranges for the given context.
--Labels should be specific and meaningful, avoiding generic terms like "Product One" or "Product Two." The names and chart labels should be specific to the data and context.
--Chart should be clear and visually appealing, with appropriate colors and styles for each line chart. linewidths should larger than 1 for better visibility.
--Adjust the width of any elements (e.g., bars, lines) to ensure they are visible and distinguishable.
--Generate **random and diverse** colors/styles/markers, and don't start with **a fixed color/style/marker**. Explore elegant and unique color tones suitable for research papers, while also considering commonality.
--You MUST add noise to the data to make it more realistic and challenging.
--DO NOT generate linear data.
--DO NOT return function code, just the function call with parameters.
""",
)

# Assistant 2: use for combined code generation
assistant_for_combined_code = BaseMessage.make_assistant_message(
    role_name="Assistant",
    content=f"""
### Intelligent Assistant Prompt: Combined Subplot Chart Code Synthesis Helper

You are a 'Combined Subplot Chart Code Synthesis Assistant.' Your task is to synthesize an appropriate combined subplot chart code based on the individual chart functions and calling examples provided by the user, ensuring that all functions and examples integrate seamlessly.

**User-Provided Functions:**
```python
{{user_functions}}
```

**User-Provided Calling Examples:**
```python
{{user_calling_examples}}
```

**Synthesized Code Example:**
```python
{{code_body}}
```
""",
)

# Assistant 3: use for combined code debug
assistant_for_debug = BaseMessage.make_assistant_message(
    role_name="Assistant",
    content=f"""
### Intelligent Assistant Prompt: Code Debug Helper

You are a 'Code Debug Assistant.' Your task is to identify and fix issues in the user's code based on any provided errors, ensuring it works correctly.

**User-Provided Code:**
```python
{{user_code}}
```

**Error Message:**
{{error_message}}

**Returned Code:**
```python
{{whole_code_after_fix}}
```
""",
)

theme_selection = [
    "Economics", "Psychology", "Sociology", "Biology", "Education",
    "Engineering", "Law", "Astronomy", "Computer_Science", "Geography",
    "Physics", "Chemistry", "History", "Environmental_Science", "Anthropology",
    "Media_and_Journalism", "Mathematics", "Statistics", "Finance", "Medicine",
    "Art_and_Design", "Agriculture", "Linguistics", "Architecture", "Sports"
]


def verify_code(code):
    exec(code, globals())


def process_plot(select_chart_types, iter, theme_selection, max_tries=2):

    synthesis_assistant = ChatAgent(assistant_sys_msg_for_single,
                                    model=agent_single,
                                    token_limit=16384)

    synthesis_combined_code_assistant = ChatAgent(assistant_for_combined_code,
                                                  model=agent_combine,
                                                  token_limit=32768)

    synthesis_combined_debug_assistant = ChatAgent(assistant_for_debug,
                                                   model=agent_debug,
                                                   token_limit=32768)

    plot_type = 'plot_combination_' + '_and_'.join(
        select_chart_types) + '_subplots'

    single_types = [
        'plot_' + chart_name + '_chart' for chart_name in select_chart_types
    ]
    layout_selection = [(x, y) for x in range(1, 5) for y in range(1, 5)]
    layout_selection.remove((1, 1))
    layout_selection.remove((3, 4))
    layout_selection.remove((4, 3))
    layout_selection.remove(
        (4, 4)
    )  # layout: (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (4, 1), (4, 2) -> 12 layouts, remove (1, 1), (3, 4), (4, 3) and (4, 4), 4 types of layouts
    layout_selection_filter = [
        layout for layout in layout_selection
        if layout[0] * layout[1] >= len(select_chart_types)
    ]

    if len(set(select_chart_types)) == 1:
        code_list = [inspect.getsource(globals()[single_types[0]])]
    else:
        code_list = [
            inspect.getsource(globals()[ptype]) for ptype in single_types
        ]

    single_example_list = []
    if len(set(select_chart_types)) == 1:
        single_type = single_types[0]
        example_usage = random.choice(example_usages[single_type])
        single_example_list.append(example_usage)
    else:
        for single_type in single_types:
            example_usage = random.choice(example_usages[single_type])
            single_example_list.append(example_usage)

    #################################################################################
    random_theme = random.choice(theme_selection)
    random_layout = random.choice(layout_selection_filter)
    png_dir = f"./ecd_combined_subplot_charts/{plot_type}/single_png/"
    txt_dir = f"./ecd_combined_subplot_charts/{plot_type}/single_txt/"
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)

    combined_png_dir = f"./ecd_combined_subplot_charts/{plot_type}/png/"
    combined_code_dir = f"./ecd_combined_subplot_charts/{plot_type}/code/"
    os.makedirs(combined_png_dir, exist_ok=True)
    os.makedirs(combined_code_dir, exist_ok=True)

    nums_of_generation = int(random_layout[0] * random_layout[1])

    condition = False
    for combined_png_path in glob.glob(combined_png_dir + "*.png"):
        print('combined_png_path:', combined_png_path)
        combined_png_path_modified = str(combined_png_path).replace("\\", "/")
        if f"{iter:06}" in combined_png_path_modified:
            condition = True
    # print('iter:', f"{iter:06}")
    # print('condition:', condition)

    if not condition:
        print('==============================')

        # List all subdirectories in png_dir
        single_png_folder_names = [
            f for f in os.listdir(png_dir)
            if os.path.isdir(os.path.join(png_dir, f))
        ]
        print('single_png_folder_names:', single_png_folder_names)

        # Check if the folder with the specific iter number exists
        if f"{iter:06}" in single_png_folder_names:
            # Delete non-empty directories and all their contents
            shutil.rmtree(os.path.join(png_dir, f"{iter:06}"))
            shutil.rmtree(os.path.join(txt_dir, f"{iter:06}"))

        # Remove the Python files related to the current iter number
        for code_iter in glob.glob(combined_code_dir + "*.py"):
            if f"{iter:06}" in code_iter:
                os.remove(code_iter)

        previous_sample = []
        print(
            '====First Stage====: Start Synthesizing the data of Single Charts'
        )
        for num in range(nums_of_generation):
            trend_list = ["upward", "downward", "fluctuating"]
            random_trend = random.sample(trend_list,
                                         k=random.randint(1, len(trend_list)))
            trend_string = ', '.join([f"{item} data" for item in random_trend])

            png_save_path = f"{png_dir}{iter:06}/{random_theme}_{random_layout}_index{num+1}.png"
            txt_save_path = f"{txt_dir}{iter:06}/{random_theme}_{random_layout}_index{num+1}.txt"
            os.makedirs(f"{png_dir}{iter:06}", exist_ok=True)
            os.makedirs(f"{txt_dir}{iter:06}", exist_ok=True)

            if len(set(select_chart_types)) == 1:
                current_code = code_list[0]
                current_example = single_example_list[0]
            else:
                if num + 1 <= len(set(select_chart_types)):
                    current_code = code_list[num]
                    current_example = single_example_list[num]
                else:
                    current_code = random.choice(code_list)
                    index_of_element = code_list.index(current_code)
                    current_example = single_example_list[index_of_element]

            if num >= 1:
                before_sample = "\n\n".join(previous_sample)

                user_prompt = f"""
                    {current_code}
                    **Example Usage of single chart**: {current_example}.
                    **Subplot Data generated previously**: {before_sample}.
                    **Color Scheme**: Ensure that the color scheme across subplots is consistent, balancing both aesthetic appeal and clarity in data presentation.
                    **Chart Theme**: Imagine a real application scenario under the **{random_theme}** theme and generate the data accordingly. Be specific. For example, if the theme is Finance, you could imagine a company struggling financially but expanding into new business areas. If the theme is Astronomy, you could picture a star nearing the end of its life.
                    **The number of elements**:
                    First level: {random.randint(3, 6)};
                    If second level make sense in the context, Second level: {random.randint(3, 6)}.
                    **Trend of data**: create {trend_string}.
                    **grid**: {random.choice([False, True])}.
                    **save_path**: {png_save_path}.
                    **Important Considerations**: You must generate parallel sub-graphs that maintain strong internal cohesion while introducing meaningful variations, ensuring both similarity and distinctiveness within the data structure.
                    """

                print('user_prompt > 1:', user_prompt)

                for _ in range(max_tries):
                    user_msg = BaseMessage.make_user_message(
                        role_name="User",
                        content=user_prompt,
                    )

                    print('Synthesizing the chart plot function:', plot_type)
                    # Get the response containing the generated docstring
                    response = synthesis_assistant.step(user_msg)

                    # print('===Response===:', response)
                    # Extract the generated python code from the response
                    generated_example = response.msg.content.split(
                        "```python\n")[1].split("\n```")[0]
                    print('===Function Code:===', generated_example)

                    try:
                        verify_code(generated_example)
                        with open(txt_save_path, 'w') as ex_txt_file:
                            ex_txt_file.write(generated_example)
                        plt.close()
                        previous_sample.append(generated_example)
                        break
                    except Exception as e:
                        user_prompt += f"""Previous non-working example: {generated_example}, Corresponding error: {e}"""
                        print('Error:', e)
                        continue
            else:
                user_prompt = f"""
                    {current_code}
                    **Example Usage of single chart**: {current_example}.
                    **Chart Theme**: Imagine a real application scenario under the **{random_theme}** theme and generate the data accordingly. Be specific. For example, if the theme is Finance, you could imagine a company struggling financially but expanding into new business areas. If the theme is Astronomy, you could picture a star nearing the end of its life.
                    **The number of elements**:
                    First level: {random.randint(3, 6)};
                    If second level make sense in the context, Second level: {random.randint(3, 6)}.
                    **Trend of data**: create {trend_string}.
                    **grid**: {random.choice([False, True])}.
                    **save_path**: {png_save_path}.
                    """
                print('user_prompt <= 1:', user_prompt)

                for _ in range(max_tries):
                    user_msg = BaseMessage.make_user_message(
                        role_name="User",
                        content=user_prompt,
                    )

                    print('Synthesizing the chart plot function:', plot_type)
                    # Get the response containing the generated docstring
                    response = synthesis_assistant.step(user_msg)

                    # print('===Response===:', response)
                    # Extract the generated python code from the response
                    generated_example = response.msg.content.split(
                        "```python\n")[1].split("\n```")[0]
                    print('===Function Code:===', generated_example)

                    try:
                        verify_code(generated_example)
                        with open(txt_save_path, 'w') as ex_txt_file:
                            ex_txt_file.write(generated_example)
                        plt.close()
                        previous_sample.append(generated_example)
                        break
                    except Exception as e:
                        print('Error:', e)
                        user_prompt += f"""Previous non-working example: {generated_example}, Corresponding error: {e}"""
                        continue

        print(
            '====First Stage End, Second Stage====: Start Synthesizing the Combined Charts Code'
        )

        ds_index, diversification_strategy = random.choice(
            list(
                enumerate([
                    "Add the title of the whole combined chart in a specific and meaningful way, avoiding general terms.",
                    "Add text labels, annotations, arrows, uncertainty bars, threshold lines (which should have specific names, not just 'threshold'; these could include curves like exponential/logarithmic curves, and not always be straight lines), or highlights (e.g., using a circle or highlighting a range in a specific color) to emphasize key data points, trends, or regions, ensuring that these annotations are contextually relevant and not generic. The number of items added can exceed one.",
                    "Modify the font styles, colors or sizes significantly for titles, labels or ticks.",
                    "Use gradient fills, area shading (typically along the line itself, within a defined range above and below it, must not between the line and the x-axis), or transparency effects to enhance depth. Additionally, fine-tune grid lines, background colors, and shadow effects to improve visual appeal.",
                    "(If Applicable) Remove axis borders for a cleaner, modern look.",
                    "(If Applicable) Incorporate a **zoomed-in inset** of a particular section, ensuring the area is appropriately sized (usually a very small size) and placed, making sure that the elements are visually separated without overlapping."
                ])))
        print('===========Diversification Strategy================:',
              diversification_strategy)

        # combined_code = '\n\n'.join(code_list)
        # combined_examples = '\n\n'.join(previous_sample)
        all_plot_functions = '\n\n'.join(
            [f"Code {i+1}: \n{code}" for i, code in enumerate(code_list)])

        combined_data = '\n\n'.join([
            f"Subplot {i+1}: \n{example}"
            for i, example in enumerate(previous_sample)
        ])
        combined_png_save_path = f"{combined_png_dir}{iter:06}_{random_theme}_{random_layout}_ds{ds_index}.png"
        combined_code_save_path = f"{combined_code_dir}{iter:06}_{random_theme}_{random_layout}_ds{ds_index}.py"

        # print('combined_example:', combined_examples)

        user_prompt_combined = f"""
            {all_plot_functions}
            **Provided chart data for each subplot**: {combined_data}. The data/labels/legends in the chart must be preserved at the combining stage.
            """

        subtitles = random.choice([
            "leave the subfigure titles as they are without any letters",
            "retain the original subfigure titles and add letters before the titles, such as (a), (b), (c), (d)",
            "choose to remove the original subfigure titles and use only the letters"
        ])

        user_prompt_combined += f"""
            **Layout**: A {random_layout[0]}x{random_layout[1]} grid of subplots, totaling {random_layout[0] * random_layout[1]} subplots. You must ensure the elements in the data_list match the number of subplots in the layout.
            **save_path**: {combined_png_save_path}.
            **Diversification Strategy for Combined Subplots Chart Code**: {diversification_strategy}.
            **Important Considerations:**
                --Code Diversity: Provide multiple styles and techniques for modifying the chart, using different libraries such as Seaborn to enhance the code and visual presentation. Must avoid returning results as **functions like 'def' or function calls**, and instead, provide complete executable code.
                --Layout and Organization: You must follow the **requested layout, save_path and change strategy** and ensure clear separation (without any overlap), alignment and labeling for readability. Additionally, feel free to adjust the figsize to ensure that all elements are fully visible and can be displayed.
                --Original Data Preservation: You are required to create the code, without modifying or ignoring the original chart data. It is worth noting that for the line_num and bar_num charts, the number annotations should not be modified or removed.
            **Note:** Be sure not to use any interactive elements that cannot be saved. Besides, {subtitles}.
            """

        for _ in range(max_tries):
            user_msg = BaseMessage.make_user_message(
                role_name="User",
                content=user_prompt_combined,
            )

            print('Synthesizing the chart plot code:', plot_type)
            # Get the response containing the generated docstring
            response = synthesis_combined_code_assistant.step(user_msg)

            print('===Response by the combined assistant===:', response)
            # Extract the generated python code from the response
            generated_combined_code = response.msg.content.split(
                "```python\n")[1].split("\n```")[0]
            print('===Combined Code:===', generated_combined_code)
            # with open(combined_code_save_path, 'w') as com_code_file:
            #     com_code_file.write(generated_combined_code)  # if failed, also save the code for check

            try:
                verify_code(generated_combined_code)
                with open(combined_code_save_path, 'w') as com_code_file:
                    com_code_file.write(generated_combined_code)
                plt.close()
                break
            except Exception as e:
                print('Error:', e)
                print("===Start Debugging===")

                user_prompt_debug = f"""
                    **Error Code:** {generated_combined_code},
                    **Error Message:** {e}.
                    You should return all the complete, working code, not just the modified part.
                """
                user_msg_debug = BaseMessage.make_user_message(
                    role_name="User",
                    content=user_prompt_debug,
                )
                response_refined = synthesis_combined_debug_assistant.step(
                    user_msg_debug)

                print('repsonse_refined:', response_refined)
                # Extract the generated python code from the response
                refined_combined_code = response_refined.msg.content.split(
                    "```python\n")[1].split("\n```")[0]
                print('code_refined:', refined_combined_code)

                try:
                    verify_code(refined_combined_code)
                    with open(combined_code_save_path,
                              'w') as com_code_file_ref:
                        com_code_file_ref.write(refined_combined_code)
                    plt.close()
                except Exception as e1:
                    print('Debug Error:', e1)
                    continue


def run_concurrent_tasks(plot_types,
                         iterations,
                         theme_selection,
                         max_workers=1):

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for plot_type in plot_types:
            for iter in range(0, iterations):  # 0 ~ 100
                iter = int(iter)
                # Submit the task to the executor
                futures.append(
                    executor.submit(process_plot, plot_type, iter,
                                    theme_selection))

        # Optional: process results as they complete
        for future in as_completed(futures):
            try:
                future.result(
                )  # Can raise exceptions if there was an error in the thread
            except Exception as e:
                print(f"Error in thread: {e}")


def filter_existing_combinations(plot_types, output_dir):
    """
    Filters out already existing combinations based on the folder names in the output directory.

    Args:
        plot_types (list): List of plot combinations as tuples (e.g., [('line', 'bar', 'pie'), ...]).
        output_dir (str): Path to the directory containing existing combination folders.

    Returns:
        list: Filtered plot combinations that do not already exist.
    """
    existing_combinations = set()

    # Check all folders in the output directory
    for folder_name in os.listdir(output_dir):
        if folder_name.startswith(
                "plot_combination_") and folder_name.endswith("_subplots"):
            # Extract the core combination (e.g., 'line_and_bar_and_pie')
            core_name = folder_name[len("plot_combination_"):-len("_subplots")]
            # Convert 'line_and_bar_and_pie' back to a tuple ('line', 'bar', 'pie')
            combination = tuple(core_name.split("_and_"))
            existing_combinations.add(combination)

    # Filter out already existing combinations
    filtered_plot_types = [
        ptype for ptype in plot_types if ptype not in existing_combinations
    ]
    return filtered_plot_types


if __name__ == "__main__":
    # Configure plot types and iterations
    plot_types_diff = list(
        itertools.combinations_with_replacement(chart_type_list,
                                                2))  # All 2-type combinations
    output_dir = "./ecd_combined_subplot_charts/"  # Path to the output directory

    # Filter out existing combinations
    plot_types = filter_existing_combinations(plot_types_diff, output_dir)
    print(
        f"Filtered combinations: {len(plot_types)} remaining out of {len(plot_types_diff)} total."
    )

    iterations = 30  # Total iterations for each plot type
    max_concurrent_workers = 1  # Configurable number of concurrent workers

    # Start concurrent tasks
    run_concurrent_tasks(plot_types,
                         iterations,
                         theme_selection,
                         max_workers=max_concurrent_workers)
