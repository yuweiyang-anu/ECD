import os
import re
import random
import inspect
import squarify
import matplotlib
import threading
import networkx as nx
import numpy as np
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde
from matplotlib.colors import LogNorm
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.configs import ChatGPTConfig
from camel.types import ModelPlatformType, ModelType
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.collections import PolyCollection
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

# 1. plot_line_chart
def plot_line_chart(data,
                    title=None,
                    xlabel='X-axis',
                    ylabel='Y-axis',
                    colors=None,
                    linestyles=None,
                    markers=None,
                    linewidths=None,
                    grid=True,
                    legend_labels=None,
                    rotation=0,
                    annotate_values=False,
                    figsize=(10, 5),
                    save_path=None):
    """
    Plots a customizable line chart.

    Parameters:
    - data: A list of tuples, where each tuple contains two lists/arrays: (x_data, y_data), not limited to 5 lines, can be more with intersects.
    - title: Title of the chart.
    - xlabel: Label for the x-axis. Please ensure the xlabel is not limited to 'Years', it can be diverse!
    - ylabel: Label for the y-axis.
    - colors: List of colors for each line.
    - linestyles: List of line styles for each line.
    - markers: List of markers for each line.
    - linewidths: List of line widths for each line.
    - grid: Boolean indicating whether to show grid lines.
    - legend_labels: List of legend labels for each line.
    - rotation: Angle for rotating x-axis tick labels, should be dynamic, varies in [0, 30, 45, 90], etc, sometimes it can be 0 or 90.
    - annotate_values: Boolean indicating whether to annotate data points with their values. It should be 'False' sometimes, but not always be 'False'.
    - figsize: Tuple indicating the figure size, should be dynamic, not always start from (12, 9), (10, 10), etc.
    - save_path: Path to save the figure.
    """
    plt.figure(figsize=figsize)

    for i, (x_data, y_data) in enumerate(data):
        if len(x_data) != len(y_data):
            raise ValueError(
                f"x_data and y_data must have the same length for line {i + 1}."
            )

        plt.plot(x_data,
                 y_data,
                 color=colors[i] if colors else None,
                 linestyle=linestyles[i] if linestyles else '-',
                 marker=markers[i] if markers else None,
                 linewidth=linewidths[i] if linewidths else 1,
                 label=legend_labels[i] if legend_labels else None)

        # Annotate values if required
        if annotate_values:
            for x, y in zip(x_data, y_data):
                plt.text(x,
                         y,
                         f'{y:.2f}',
                         fontsize=8,
                         verticalalignment='bottom',
                         horizontalalignment='right')

    plt.title(title if title else 'Line Chart')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if rotation is not None:
        plt.xticks(rotation=rotation)

    if grid:
        plt.grid(True)

    if legend_labels:
        plt.legend()

    plt.tight_layout()

    # Save the figure if a path is provided
    if save_path:
        fig = plt.gcf()  # Get the current figure
        fig.savefig(save_path)


#### 2. bar charts plot
def plot_bar_chart(data,
                   orientation='vertical',
                   stacked=False,
                   title=None,
                   xlabel='X-axis',
                   ylabel='Y-axis',
                   colors=None,
                   linewidths=None,
                   grid=True,
                   legend_labels=None,
                   label_rotation=0,
                   group_label_rotation=0,
                   group_labels=None,
                   annotate_values=True,
                   figsize=(10, 5),
                   save_path=None):
    """
    Plots a customizable bar chart with value labels on bars, which **does not reflect the time relationship**.

    Parameters:
    - data: A list of lists or 2D array, where each sub-list contains values for one category.
    - orientation: 'vertical' for vertical bars or 'horizontal' for horizontal bars. Please make sure sometimes it can be 'horizontal'.
    - stacked: Boolean indicating whether to create a stacked bar chart. Please make sure sometimes (**not always**) it can be 'True'.
    - title: Title of the chart.
    - xlabel: Label for the x-axis, should be dynamic, must not always be the value like 'Years'.
    - ylabel: Label for the y-axis.
    - colors: List of colors for each bar or series.
    - linewidths: List of line widths for the bar edges.
    - grid: Boolean indicating whether to show grid lines.
    - legend_labels: List of legend labels for each series. The legend_labels must be **specific terms** (MUST NOT be Common label + numbers / letters, legends should not be ['series A', 'series B'] or ['series 1', 'series 2'], should be like ['creativity', 'passion']).
    - label_rotation: Rotation angle for the value labels on bars, should be dynamic, varies in [0, 30, 45, 60, 90], etc, **most of the time it should be 0 or 90**.
    - group_label_rotation: Rotation angle for the group labels, should be dynamic, varies in [0, 30, 45, 60, 90], etc, sometimes it can be 0 or 90.
    - group_labels: Custom labels for the groups (x-axis or y-axis). The group_labels must be **specific terms** (**MUST NOT be Common label + numbers / letters**, like country in xlabel, group_labels should not be ['country A', 'country B'] or ['country 1', 'country 2'], must be like ['USA', 'China', 'Russia']) and consistent with xlabel or ylabel.  
    - annotate_values: Boolean indicating whether to annotate the values on bars. If set to `True`, the value of each bar will be annotated on the chart. It should be 'False' sometimes.
    - figsize: Tuple indicating the figure size, should be dynamic, not always start from (12, 9), (10, 10), etc.
    - save_path: Path to save the figure.
    """

    # Convert data to a numpy array for easier manipulation
    data = np.array(data)
    num_series = data.shape[0]  # Number of series
    num_bars = data.shape[1]  # Number of categories
    x_indices = np.arange(num_bars)  # The x locations for the groups

    # Calculate dynamic bar width based on the number of bars
    bar_width = max(0.1, 0.8 / num_series)  # Ensure a minimum width

    # Adjust group_labels to match the number of bars
    if group_labels is not None:
        if len(group_labels) != num_bars:
            if len(group_labels) < num_bars:
                group_labels.extend([
                    f'Group {i + 1}'
                    for i in range(len(group_labels), num_bars)
                ])
            else:
                group_labels = group_labels[:num_bars]

    # Create the figure with the specified size
    plt.figure(figsize=figsize)

    if orientation == 'vertical':
        if stacked:
            # Create stacked bars
            cumulative_values = np.zeros(num_bars)
            for i in range(num_series):
                bars = plt.bar(
                    x_indices,
                    data[i],
                    bottom=cumulative_values,
                    color=colors[i] if colors and i < len(colors) else None,
                    linewidth=linewidths[i]
                    if linewidths and i < len(linewidths) else None,
                    label=legend_labels[i]
                    if legend_labels and i < len(legend_labels) else None)
                cumulative_values += data[i]

                # Add value labels if required
                if annotate_values:
                    for j, bar in enumerate(bars):
                        yval = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width() / 2,
                                 cumulative_values[j] - yval / 2,
                                 f'{yval:.1f}',
                                 ha='center',
                                 va='center',
                                 rotation=label_rotation)

            # Center the group labels
            plt.xticks(x_indices,
                       group_labels if group_labels else
                       [f'Group {i+1}' for i in range(num_bars)],
                       rotation=group_label_rotation)

        else:
            # Create regular bars
            for i in range(num_series):
                bars = plt.bar(
                    x_indices + i * bar_width,
                    data[i],
                    width=bar_width,
                    color=colors[i] if colors and i < len(colors) else None,
                    linewidth=linewidths[i]
                    if linewidths and i < len(linewidths) else None,
                    label=legend_labels[i]
                    if legend_labels and i < len(legend_labels) else None)

                # Add value labels if required
                if annotate_values:
                    for bar in bars:
                        yval = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width() / 2,
                                 yval,
                                 f'{yval:.1f}',
                                 ha='center',
                                 va='bottom',
                                 rotation=label_rotation)

            # Center the group labels
            plt.xticks(x_indices + (num_series - 1) * bar_width / 2,
                       group_labels if group_labels else
                       [f'Group {i+1}' for i in range(num_bars)],
                       rotation=group_label_rotation)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    elif orientation == 'horizontal':
        if stacked:
            # Create stacked bars horizontally
            cumulative_values = np.zeros(num_series)
            for i in range(num_bars):
                bars = plt.barh(
                    np.arange(num_series),
                    data[:, i],
                    left=cumulative_values,
                    color=colors[i] if colors and i < len(colors) else None,
                    linewidth=linewidths[i]
                    if linewidths and i < len(linewidths) else None,
                    label=legend_labels[i]
                    if legend_labels and i < len(legend_labels) else None)
                cumulative_values += data[:, i]

                # Add value labels if required
                if annotate_values:
                    for j, bar in enumerate(bars):
                        plt.text(cumulative_values[j] - bar.get_width() / 2,
                                 j,
                                 f'{bar.get_width():.1f}',
                                 ha='center',
                                 va='center',
                                 rotation=label_rotation)

            # Center the group labels for horizontal bars
            plt.yticks(np.arange(num_bars),
                       group_labels if group_labels else
                       [f'Group {i+1}' for i in range(num_bars)],
                       rotation=group_label_rotation)

        else:
            # Create regular horizontal bars
            for i in range(num_series):
                bars = plt.barh(
                    np.arange(num_bars) + i * 0.2,
                    data[i],
                    height=0.2,
                    color=colors[i] if colors and i < len(colors) else None,
                    linewidth=linewidths[i]
                    if linewidths and i < len(linewidths) else None,
                    label=legend_labels[i]
                    if legend_labels and i < len(legend_labels) else None)

                # Add value labels if required
                if annotate_values:
                    for bar in bars:
                        plt.text(bar.get_width(),
                                 bar.get_y() + bar.get_height() / 2,
                                 f'{bar.get_width():.1f}',
                                 ha='left',
                                 va='center',
                                 rotation=label_rotation)

            # Center the group labels for horizontal bars
            plt.yticks(np.arange(num_bars) + (num_series - 1) * 0.1,
                       group_labels if group_labels else
                       [f'Group {i+1}' for i in range(num_bars)],
                       rotation=group_label_rotation)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    plt.title(title if title else 'Bar Chart')

    if grid:
        plt.grid(True)

    if legend_labels:
        plt.legend()

    plt.tight_layout()

    # Save the figure if a path is provided
    if save_path:
        fig = plt.gcf()  # Get the current figure
        fig.savefig(save_path)


#### 3. pie charts plot
def plot_pie_chart(data,
                   title=None,
                   labels=None,
                   colors=None,
                   explode=None,
                   startangle=90,
                   shadow=False,
                   autopct='%1.1f%%',
                   ring=False,
                   ring_width=0.3,
                   show_legend=True,
                   figsize=(8, 8),
                   save_path=None):
    """
    Plots a customizable pie chart or ring chart.

    Parameters:
    - data: A list or array of values for each slice.
    - title: Title of the chart.
    - labels: List of labels for each slice.
    - colors: List of colors for each slice.
    - explode: List indicating which slices to "explode."
    - startangle: The angle by which to start the pie chart.
    - shadow: Boolean indicating whether to add a shadow, at most time it should be 'False'.
    - autopct: String to format the percentage labels.
    - ring: Boolean indicating whether to create a ring chart, not always 'True', can be 'False' sometimes.
    - ring_width: Width of the ring if creating a donut chart.
    - show_legend: Boolean indicating whether to display the legend, it should be 'False' sometimes.
    - figsize: Tuple indicating the figure size, should be dynamic, not always start from (12, 9), (10, 10), etc.
    - save_path: Path to save the figure.
    """
    plt.figure(figsize=figsize)

    if ring:
        # Create a ring chart (donut chart)
        wedges, texts = plt.pie(data,
                                labels=labels,
                                colors=colors,
                                explode=explode,
                                startangle=startangle,
                                shadow=shadow,
                                radius=1)

        # Draw a circle at the center of the pie to make it a donut chart
        centre_circle = plt.Circle((0, 0), 1 - ring_width, color='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)

        # Add percentage labels to the ring chart
        for i, wedge in enumerate(wedges):
            angle = (wedge.theta2 + wedge.theta1) / 2
            x = (wedge.r * np.cos(np.deg2rad(angle))) * (1 - ring_width)
            y = (wedge.r * np.sin(np.deg2rad(angle))) * (1 - ring_width)
            percent = data[i] / sum(data) * 100
            plt.text(x, y, f'{percent:.1f}%', ha='center', va='center')

    else:
        # Create a regular pie chart
        wedges, texts, autotexts = plt.pie(data,
                                           labels=labels,
                                           colors=colors,
                                           explode=explode,
                                           startangle=startangle,
                                           shadow=shadow,
                                           autopct=autopct)

    plt.title(title if title else 'Pie Chart')
    plt.axis(
        'equal')  # Equal aspect ratio ensures the pie is drawn as a circle.

    if show_legend and labels:
        plt.legend(labels, loc='best')

    plt.tight_layout()

    # Save the figure if a path is provided
    if save_path:
        fig = plt.gcf()  # Get the current figure
        with save_lock:
            fig.savefig(save_path)


#### 4. area charts plot
def plot_area_chart(data,
                    title=None,
                    x_ticks=None,
                    colors=None,
                    alpha=0.5,
                    linestyle=None,
                    linewidth=None,
                    marker=None,
                    figsize=(10, 6),
                    rotation=0,
                    y_label=None,
                    x_label=None,
                    legend_labels=None,
                    legend_loc='upper left',
                    save_path=None):
    """
    Plots a customizable stacked area chart.

    Parameters: 
    - data: A list or 2D array of values to plot. Each row represents a series. There should be large fluctuations in data!
    - title: Title of the chart.
    - x_ticks: List of labels for the x-axis ticks.
    - colors: List of colors for each area.
    - alpha: Transparency level of the area.
    - linestyle: Style of the line ('solid', 'dashed', 'dotted', etc.) or None to disable. Please note that it should be 'None' at most cases but not always.
    - linewidth: Width of the lines or None to disable.
    - marker: Marker style (e.g., 'o', 's', None) or None to disable. Please note that sometimes it should be 'None'!
    - figsize: Tuple indicating the figure size, should be dynamic, not always start from (12, 9), (10, 10), etc.
    - rotation: Angle to rotate the x-axis labels, should be dynamic, varies in [0, 30, 45, 90], etc, sometimes it should be 0 or 90.
    - y_label: Label for the y-axis.
    - x_label: Label for the x-axis.
    - legend_labels: List of labels for the area legends.
    - legend_loc: Position of the legend.
    - save_path: Path to save the figure.
    """
    data = np.array(data)
    x = np.arange(len(data[0]))  # Assuming equal-length series

    plt.figure(figsize=figsize)

    # Initialize the bottom variable for stacking
    bottom = np.zeros(len(data[0]))

    for i, series in enumerate(data):
        plt.fill_between(
            x,
            bottom,
            bottom + series,
            color=colors[i] if colors else None,
            alpha=alpha,
            label=legend_labels[i] if legend_labels else f'Series {i + 1}')

        # Plot the line at the top of the current series
        if linestyle is not None and linewidth is not None:
            plt.plot(
                x,
                bottom + series,  # Line at the top of the current series
                color=colors[i] if colors else None,
                linestyle=linestyle,
                linewidth=linewidth,
                marker=marker if marker else '')

        # Update bottom for the next series
        bottom += series

    plt.title(title if title else 'Stacked Area Chart')
    plt.xticks(x,
               x_ticks if x_ticks else np.arange(len(data[0])),
               rotation=rotation)
    plt.xlabel(x_label if x_label else 'X-axis')  # Set x-axis label
    plt.ylabel(y_label if y_label else 'Y-axis')  # Set y-axis label

    # Add the legend for areas
    plt.legend(loc=legend_loc)

    # If markers are specified, add a separate legend for them with the same label
    if marker is not None and linewidth is not None and linestyle is not None:
        marker_handles = []
        for i in range(len(data)):
            marker_handle = plt.Line2D(
                [], [],
                marker=marker,
                color='w',
                label=legend_labels[i] if legend_labels else f'Series {i + 1}',
                markerfacecolor=colors[i],
                markersize=10)
            marker_handles.append(marker_handle)

        if rotation > 30:
            # Add the marker legend without a border
            plt.legend(handles=marker_handles,
                       loc='upper center',
                       bbox_to_anchor=(0.5, -0.2),
                       ncol=len(data),
                       frameon=False)
        elif rotation == 30:
            # Add the marker legend without a border
            plt.legend(handles=marker_handles,
                       loc='upper center',
                       bbox_to_anchor=(0.5, -0.15),
                       ncol=len(data),
                       frameon=False)
        else:
            # Add the marker legend without a border
            plt.legend(handles=marker_handles,
                       loc='upper center',
                       bbox_to_anchor=(0.5, -0.1),
                       ncol=len(data),
                       frameon=False)

    plt.tight_layout()

    # Save the figure if a path is provided
    if save_path:
        fig = plt.gcf()  # Get the current figure
        with save_lock:
            fig.savefig(save_path)


#### 5. error_point charts plot
def plot_error_point_chart(data,
                           pos_errors,
                           neg_errors,
                           error_direction='vertical',
                           color='blue',
                           marker='o',
                           title='Error Point Chart',
                           xlabel='X-axis',
                           ylabel='Y-axis',
                           annotate=True,
                           label_rotation=0,
                           grid=True,
                           legend_labels=None,
                           ticks=None,
                           figsize=(10, 6),
                           save_path=None):
    """
    Plots a customizable error point chart with support for multiple datasets with asymmetric errors.

    Parameters:
    - data: list of array-like, values for each dataset. Please ensure that different errorpoints are distinguishable from each other (after the center point + error, different errorpoints have **obvious gaps**). The number of errorpoint must < 4 when annotate is 'True'.
    - pos_errors: list of array-like, positive error values for each dataset
    - neg_errors: list of array-like, negative error values for each dataset
    - error_direction: 'vertical' or 'horizontal'. Please make sure sometimes it can be 'horizontal'., not always 'vertical'.
    - color: list of colors for each dataset. 
    - marker: style of the markers
    - title: title of the chart
    - xlabel: label for the x-axis. **NOTE: When it is 'horizontal', please note that the xlabel will be ylabel compared to the 'vertical' case.**, should not always be the value like 'Years'.
    - ylabel: label for the y-axis. **NOTE: When it is 'horizontal', please note that the ylabel will be xlabel compared to the 'vertical' case.**
    - annotate: boolean, whether to annotate error values. Please make sure sometimes (**just sometimes, not always**) it can be 'True', at the most time it should be 'False'.
    - label_rotation: rotation angle for axis labels, should be dynamic, varies in [0, 30, 45, 90], etc.
    - grid: boolean, whether to show grid lines
    - legend_labels: list of labels for the legend. The legend_labels must be **specific terms** (MUST NOT be Common label + numbers / letters, legends must not be ['series A', 'series B'] or ['series 1', 'series 2'], should be like ['creativity', 'passion']).
    - ticks: list of ticks labels (numeric or text). The ticks must be **specific terms** (**MUST NOT be Common label + numbers / letters**, like country in xlabel, ticks must not be ['country A', 'country B'] or ['country 1', 'country 2'], must be like ['USA', 'China', 'Russia']) and consistent with xlabel or ylabel. 
    - figsize: tuple indicating the figure size, should be dynamic, not always start from (12, 9), (10, 10), etc.
    - save_path: path to save the figure
    """

    plt.figure(figsize=figsize)

    for i in range(len(data)):
        if error_direction == 'vertical':
            plt.errorbar(ticks,
                         data[i],
                         yerr=[neg_errors[i], pos_errors[i]],
                         fmt=marker,
                         color=color[i],
                         label=legend_labels[i],
                         capsize=5)
        elif error_direction == 'horizontal':
            plt.errorbar(data[i],
                         ticks,
                         xerr=[neg_errors[i], pos_errors[i]],
                         fmt=marker,
                         color=color[i],
                         label=legend_labels[i],
                         capsize=5)
        else:
            raise ValueError(
                "error_direction must be either 'vertical' or 'horizontal'")

        # Annotate each point for the current dataset
        if annotate:
            for j in range(len(data[i])):
                if error_direction == 'vertical':
                    plt.annotate(f'+{pos_errors[i][j]:.2f}',
                                 (ticks[j], data[i][j] + pos_errors[i][j]),
                                 textcoords="offset points",
                                 xytext=(0, 10),
                                 ha='center',
                                 fontsize=8)
                    plt.annotate(f'-{neg_errors[i][j]:.2f}',
                                 (ticks[j], data[i][j] - neg_errors[i][j]),
                                 textcoords="offset points",
                                 xytext=(0, -15),
                                 ha='center',
                                 fontsize=8)
                    plt.xticks(rotation=label_rotation)
                    plt.xticks(ticks=range(len(ticks)), labels=ticks)
                elif error_direction == 'horizontal':
                    plt.annotate(f'+{pos_errors[i][j]:.2f}',
                                 (data[i][j] + pos_errors[i][j], ticks[j]),
                                 textcoords="offset points",
                                 xytext=(10, 0),
                                 ha='center',
                                 fontsize=8)
                    plt.annotate(f'-{neg_errors[i][j]:.2f}',
                                 (data[i][j] - neg_errors[i][j], ticks[j]),
                                 textcoords="offset points",
                                 xytext=(-15, 0),
                                 ha='center',
                                 fontsize=8)
                    plt.yticks(rotation=label_rotation)
                    plt.yticks(ticks=range(len(ticks)), labels=ticks)

    plt.title(title)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if legend_labels:
        plt.legend()

    if grid:
        plt.grid(True)

    plt.tight_layout()

    # Save the figure if a path is provided
    if save_path:
        fig = plt.gcf()  # Get the current figure
        with save_lock:
            fig.savefig(save_path)


#### 6. treemap charts plot
def plot_treemap_chart(
        data,
        title=None,
        labels=None,
        colors=None,
        show_values=False,
        show_axes=False,
        border=True,
        pad=False,  # True for padding, False for no padding
        font_size=10,  # Font size for labels
        font_color='black',  # Font color for labels
        figsize=(12, 8),
        save_path=None):
    """
    Plots a customizable treemap chart with options for padding and font customization.

    Parameters:
    - data: A list of values to represent the sizes of the rectangles in the treemap, the length of data should between 3 ~ 10 (>=7 at most cases).
    - title: Title of the chart.
    - labels: List of labels for each rectangle.
    - colors: List of colors for each rectangle.
    - show_values: Boolean indicating whether to show values on labels. Make sure it can be set to 'False'.
    - show_axes: Boolean indicating whether to show axes. Make sure it can be set to 'True'.
    - border: Boolean indicating whether to draw borders around rectangles. Make sure it can be set to 'False'.
    - pad: Boolean indicating whether to add padding between rectangles. Make sure it can be set to 'False' sometimes.
    - font_size: Size of the font for the labels.
    - font_color: Color of the font for the labels.
    - figsize: Tuple indicating the figure size, should be dynamic, not always start from (12, 9), (10, 10), etc.
    - save_path: Path to save the figure.
    """
    plt.figure(figsize=figsize)

    # Normalize sizes for better visualization
    total = sum(data)
    sizes = [x / total for x in data]

    # Create the treemap
    ax = plt.gca()
    squarify.plot(sizes=sizes,
                  label=labels if not show_values else
                  [f"{lbl} ({size:.1%})" for lbl, size in zip(labels, sizes)],
                  color=colors,
                  alpha=0.7,
                  pad=5 if pad else 0)  # Use 5 if pad is True, otherwise 0

    # If borders are requested, add them manually
    if border:
        for rect in ax.patches:
            plt.gca().add_patch(
                plt.Rectangle((rect.get_x(), rect.get_y()),
                              rect.get_width(),
                              rect.get_height(),
                              linewidth=1,
                              edgecolor='black',
                              facecolor='none'))

    plt.title(title if title else 'Treemap Chart')

    # Customize label font size and color
    for label in ax.texts:
        label.set_fontsize(font_size)
        label.set_color(font_color)

    if not show_axes:
        plt.axis('off')
    else:
        plt.axis('on')

    plt.tight_layout()
    # Save the figure if a path is provided
    if save_path:
        fig = plt.gcf()  # Get the current figure
        with save_lock:
            fig.savefig(save_path)


#### 7. funnel charts plot
def plot_funnel_chart(y,
                      x,
                      title="Sales Funnel Analysis",
                      textinfo="value+percent previous",
                      textposition="inside",
                      textfont=dict(size=10, color='#FFFFFF'),
                      marker_color='#ff5733',
                      marker_line=dict(color='#c70039', width=1),
                      opacity=0.8,
                      font=dict(family="Arial, sans-serif",
                                size=12,
                                color="#333333"),
                      paper_bgcolor='rgba(255, 255, 255, 1)',
                      plot_bgcolor='rgba(255, 255, 255, 1)',
                      showgrid=True,
                      gridwidth=1,
                      gridcolor='LightGray',
                      figsize=(600, 400),
                      save_path=None):
    """
    Plots a customizable funnel chart.

    Parameters:
    - y: List or array of labels for each stage of the funnel.
    - x: List or array of values corresponding to each label, representing the size of each funnel stage.
    - title: Title of the funnel chart (default is "Sales Funnel Analysis").
    - textinfo: String to specify what information to display on the funnel sections (e.g., "value", "percent previous").
    - textposition: Position of the text on the funnel sections (e.g., "inside", "outside").
    - textfont: Dictionary to specify the font properties for the text displayed (e.g., size, color).
    - marker_color: Color of the funnel sections (default is a shade of orange).
    - marker_line: Dictionary to specify the border color and width for the funnel sections.
    - opacity: Opacity level of the funnel sections (default is 0.8).
    - font: Dictionary to specify the font properties for the overall chart (family, size, color).
    - paper_bgcolor: Background color of the paper (the area surrounding the plot).
    - plot_bgcolor: Background color of the plot area.
    - showgrid: Boolean to indicate whether to show grid lines on the axes.
    - gridwidth: Width of the grid lines if shown.
    - gridcolor: Color of the grid lines.
    - figsize: Tuple indicating the figure size in pixels (width, height), should be dynamic, not always start from (12, 9), (10, 10), etc.
    - save_path: Path to save the figure as an image file (if provided).
    """
    fig = go.Figure(
        go.Funnel(y=y,
                  x=x,
                  textinfo=textinfo,
                  textposition=textposition,
                  textfont=textfont,
                  marker=dict(color=marker_color, line=marker_line),
                  opacity=opacity))

    fig.update_layout(title=title,
                      font=font,
                      paper_bgcolor=paper_bgcolor,
                      plot_bgcolor=plot_bgcolor,
                      width=figsize[0],
                      height=figsize[1])

    if showgrid:
        fig.update_yaxes(showgrid=True,
                         gridwidth=gridwidth,
                         gridcolor=gridcolor)
        fig.update_xaxes(showgrid=True,
                         gridwidth=gridwidth,
                         gridcolor=gridcolor)

    # Saving the figure if save_path is provided
    if save_path:
        fig.write_image(save_path)


#### 8. node charts plot
def plot_node_chart(edges,
                     node_colors,
                     edge_colors=None,
                     node_size=300,
                     edge_width=1,
                     layout='spring',
                     title='Graph',
                     edge_labels=None,
                     node_labels=None,
                     with_arrows=False,
                     self_loops=False,
                     figsize=(10, 8),
                     save_path=None):
    """
    Plots a customizable directed graph using networkx.

    Parameters:
    - edges: List of tuples representing edges (e.g., [(1, 2), (2, 3), (1, 3)]).
    - node_colors: List of colors for each node.
    - edge_colors: List of colors for each edge. If None, uses default colors.
    - node_size: Size of the nodes.
    - edge_width: Width of the edges.
    - layout: Layout for the graph ('spring', 'circular', 'random', etc.).
    - title: Title of the graph.
    - edge_labels: List of labels for edges. Must match the edges.
    - node_labels: List of labels for nodes. Must match the nodes.
    - with_arrows: Boolean indicating whether to draw arrows on edges.
    - self_loops: Boolean indicating whether to allow self-loops.
    - figsize: Tuple indicating the figure size, should be dynamic, not always start from (12, 9), (10, 10), etc.
    - save_path: Path to save the figure.
    """

    # Create a directed graph from edges
    G = nx.DiGraph()

    # Check for self-loops and add edges accordingly
    for edge in edges:
        if self_loops or edge[0] != edge[1]:
            G.add_edge(*edge)

    # Choose layout
    if layout == 'spring':
        pos = nx.spring_layout(G)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'random':
        pos = nx.random_layout(G)
    else:
        pos = nx.spring_layout(G)  # default to spring

    # Draw the graph
    plt.figure(figsize=figsize)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_size)

    # Draw edges with black color for self-loops
    for edge in G.edges():
        if edge[0] == edge[1]:  # Check for self-loop
            nx.draw_networkx_edges(G,
                                   pos,
                                   edgelist=[edge],
                                   width=edge_width,
                                   edge_color='black',
                                   arrows=with_arrows)
        else:
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=[edge],
                width=edge_width,
                edge_color=edge_colors if edge_colors else 'grey',
                arrows=with_arrows)

    # Draw edge labels if provided, excluding self-loops
    if edge_labels:
        edge_label_dict = {
            edge: label
            for edge, label in zip(G.edges(), edge_labels)
            if edge[0] != edge[1]
        }
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_label_dict)

    # Draw node labels if provided
    if node_labels:
        label_dict = {i: label for i, label in enumerate(node_labels)}
        nx.draw_networkx_labels(G, pos, labels=label_dict)

    plt.title(title)
    plt.axis('off')  # Turn off the axis

    plt.tight_layout()

    # Save the figure if a path is provided
    if save_path:
        fig = plt.gcf()  # Get the current figure
        with save_lock:
            fig.savefig(save_path)


#### 9. density charts plot
def plot_density_chart(data,
                       labels=None,
                       colors=None,
                       title='Density Plot',
                       xlabel='X-axis',
                       ylabel='Density',
                       overlay=False,
                       bandwidth=None,
                       vertical=False,
                       is_bimodal=False,
                       grid=True,
                       figsize=(10, 6),
                       save_path=None):
    """
        Plots a customizable density plot using kernel density estimation (KDE).

        Parameters:
        - data: A single array-like object or a list of array-like objects representing the data to be plotted. 
                If a list is provided, it creates an overlay for multiple datasets.
        - labels: List of labels for the datasets. Must match the number of datasets if multiple are provided.
        - colors: List of colors to use for each dataset. Must match the number of datasets if multiple are provided.
        - title: Title of the plot.
        - xlabel: Label for the x-axis.
        - ylabel: Label for the y-axis.
        - overlay: Boolean indicating whether to overlay multiple density plots on the same axes.
        - bandwidth: Smoothing parameter for the KDE. If None, uses a default bandwidth.
        - vertical: Boolean indicating whether to plot the density on the vertical axis. **Make sure sometimes it should be 'True'!**
        - is_bimodal: Boolean indicating whether the data represents a bimodal distribution.
        - grid: Boolean indicating whether to display a grid on the plot.
        - figsize: Tuple indicating the figure size (width, height) in inches, should be dynamic, not always start from (12, 9), (10, 10), etc.
        - save_path: Path to save the figure as an image file. If None, the plot is displayed without saving.
    """

    plt.figure(figsize=figsize)

    if overlay:
        if is_bimodal:
            kde = gaussian_kde(data, bw_method=bandwidth)
            xs = np.linspace(min(data) - 1, max(data) + 1, 200)
            plt.fill_between(xs,
                             kde(xs),
                             color=colors[0],
                             alpha=0.4,
                             label='Bimodal Distribution')
            plt.plot(xs, kde(xs), color=colors[0], linestyle='-', linewidth=2)
        else:
            for i, dataset in enumerate(data):
                kde = gaussian_kde(dataset, bw_method=bandwidth)
                xs = np.linspace(min(dataset) - 1, max(dataset) + 1, 200)
                if vertical:
                    plt.fill_betweenx(xs,
                                      0,
                                      kde(xs),
                                      color=colors[i],
                                      alpha=0.2,
                                      label=labels[i])
                    plt.plot(kde(xs),
                             xs,
                             color=colors[i],
                             linestyle='-',
                             linewidth=2)
                else:
                    plt.fill_between(xs,
                                     kde(xs),
                                     color=colors[i],
                                     alpha=0.2,
                                     label=labels[i])
                    plt.plot(xs,
                             kde(xs),
                             color=colors[i],
                             linestyle='-',
                             linewidth=2)
    else:
        kde = gaussian_kde(data, bw_method=bandwidth)
        xs = np.linspace(min(data) - 1, max(data) + 1, 200)
        if vertical:
            plt.fill_betweenx(xs,
                              0,
                              kde(xs),
                              color=colors[0],
                              alpha=0.2,
                              label=labels[0])
            plt.plot(kde(xs), xs, color=colors[0], linestyle='-', linewidth=2)
        else:
            plt.fill_between(xs,
                             kde(xs),
                             color=colors[0],
                             alpha=0.2,
                             label=labels[0])
            plt.plot(xs, kde(xs), color=colors[0], linestyle='-', linewidth=2)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if vertical:
        plt.xlabel(ylabel)
        plt.ylabel(xlabel)

    if grid:
        plt.grid(True, linestyle='--')

    plt.legend()

    plt.tight_layout()

    # Save the figure if a path is provided
    if save_path:
        fig = plt.gcf()  # Get the current figure
        with save_lock:
            fig.savefig(save_path)


#### 10. histogram charts plot
def plot_histogram_chart(data_list,
                         bins=30,
                         labels=None,
                         title=None,
                         xlabel=None,
                         ylabel=None,
                         colors=None,
                         border_color=None,
                         alpha=0.75,
                         grid=True,
                         direction='vertical',
                         rotation=0,
                         figsize=(10, 5),
                         save_path=None):
    """
    Plots multiple histograms on the same figure.

    Parameters:
    - data_list: List of lists or arrays containing data for each histogram. **The number of the hists can be just 1 or 2 sometimes**, not always >= 3. If multiple hists are included, ensure that the centers of the distributions **do not differ too much**.
    - bins: Number of bins or sequence of bin edges for the histograms.
    - labels: List of labels for each histogram.
    - title: Title for the plot.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    - colors: List of colors for each histogram.
    - border_color: Color for the histogram borders.
    - alpha: Transparency level for the histograms.
    - grid: Boolean indicating whether to show grid lines.
    - direction: 'vertical' or 'horizontal' for histogram orientation. It should be 'horizontal' sometimes, please not always start from 'vertical'.
    - rotation: Angle for tick label rotation, should be dynamic, varies in [0, 30, 45, 90], etc, sometimes it can be 0 or 90.
    - figsize: Tuple indicating the figure size, should be dynamic, not always start from (12, 9), (10, 10), etc.
    - save_path: Path to save the figure.
    """

    # Create the figure
    plt.figure(figsize=figsize)

    # Plot each histogram from the data list
    for i, data in enumerate(data_list):
        color = colors[i] if colors and i < len(colors) else None
        label = labels[i] if labels and i < len(labels) else None

        if direction == 'vertical':
            plt.hist(data,
                     bins=bins,
                     color=color,
                     edgecolor=border_color,
                     alpha=alpha,
                     label=label)
        elif direction == 'horizontal':
            plt.hist(data,
                     bins=bins,
                     color=color,
                     edgecolor=border_color,
                     alpha=alpha,
                     label=label,
                     orientation='horizontal')

    # Customize labels and title
    plt.title(title if title else 'Histogram Comparison')

    # Show legend if labels are provided
    if labels:
        plt.legend()

    # Set rotation for tick labels
    if direction == 'vertical':
        plt.xticks(rotation=rotation)
        plt.xlabel(xlabel if xlabel else 'X-axis')
        plt.ylabel(ylabel if ylabel else 'Frequency')
    elif direction == 'horizontal':
        plt.yticks(rotation=rotation)
        plt.xlabel(ylabel if ylabel else 'X-axis')
        plt.ylabel(xlabel if xlabel else 'Frequency')

    # Show grid if specified
    if grid:
        plt.grid(True)

    plt.tight_layout()

    # Save the figure if a path is provided
    if save_path:
        fig = plt.gcf()  # Get the current figure
        with save_lock:
            fig.savefig(save_path)


#### 11. box charts plot
def plot_box_chart(
        data_list,
        labels=None,
        title=None,
        xlabel=None,
        ylabel=None,
        colors=None,
        median_color='black',
        hatch_styles=None,
        widths=0.5,
        grid=True,
        orientation='horizontal',
        rotation=0,
        annotate_medians=False,
        outlier_settings=None,  # Consolidated parameter for outliers
        figsize=(10, 5),
        save_path=None):
    """
    Plots multiple box plots on the same figure, with optional outliers.

    Parameters:
    - data_list: List of lists or arrays containing data for each box plot. Ensure that the differences between the data_list of boxes are not too large to maintain their visibility and ease of distinction.
    - labels: List of labels for each box plot.
    - title: Title for the plot.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    - colors: List of colors for the boxes.
    - median_color: Color for the median line.
    - hatch_styles: List of hatch styles for the boxes. Please make sure it can be 'None' sometimes, not always have values.
    - widths: Width of the boxes.
    - grid: Boolean indicating whether to show grid lines.
    - orientation: Orientation of the box plots ('horizontal' or 'vertical'). Please make sure sometimes it can be 'vertical'.
    - rotation: Rotation angle for axis labels, should be dynamic, varies in [0, 30, 45, 60, 90], etc, most of the time it can be 0 or 90.
    - annotate_medians: Boolean indicating whether to annotate the median values. Please make sure it can be 'False' sometimes.
    - outlier_settings: Dictionary for outlier settings, or None to disable. Please make sure it can have values sometimes, but not always.
    - figsize: Tuple indicating the figure size, should be dynamic, not always start from (12, 8), (10, 10), etc.
    - save_path: Path to save the figure.
    """
    plt.figure(figsize=figsize)
    box = plt.boxplot(data_list,
                      vert=(orientation == 'vertical'),
                      patch_artist=True,
                      widths=widths)

    if colors:
        for j, patch in enumerate(box['boxes']):
            patch.set_facecolor(colors[j])
            if hatch_styles:
                patch.set_hatch(hatch_styles[j % len(hatch_styles)])

    if orientation == 'horizontal':
        plt.yticks(range(1, len(labels) + 1), labels, rotation=rotation)
        plt.ylabel(xlabel if xlabel else 'Y-axis')
        plt.xlabel(ylabel if ylabel else 'X-axis')
    else:
        plt.xticks(range(1, len(labels) + 1), labels, rotation=rotation)
        plt.xlabel(xlabel if xlabel else 'X-axis')
        plt.ylabel(ylabel if ylabel else 'Y-axis')

    for median in box['medians']:
        median_line = median
        median_line.set_color(median_color)

        if annotate_medians:
            if orientation == 'vertical':
                x, y = median_line.get_xydata()[1]  # Get the median position
                plt.annotate(f"{y:.1f}", (x, y),
                             textcoords="offset points",
                             xytext=(0, 5),
                             ha="center",
                             color=median_color)
            else:
                x, y = median_line.get_xydata()[1]  # Get the median position
                plt.annotate(f"{x:.1f}", (x, y),
                             textcoords="offset points",
                             xytext=(0, 5),
                             ha="center",
                             color=median_color)

    # Plot outliers if provided
    if outlier_settings:
        outlier_points = outlier_settings.get('points')
        outlier_color = outlier_settings.get('color', 'red')
        outlier_marker = outlier_settings.get('marker', 'o')
        outlier_fill = outlier_settings.get('fill', True)

        if outlier_points:
            for outlier in outlier_points:
                x, y = outlier
                if orientation == 'horizontal':
                    plt.plot(y,
                             x,
                             marker=outlier_marker,
                             color=outlier_color,
                             markerfacecolor='none'
                             if not outlier_fill else outlier_color,
                             linestyle='None'
                             )  # Hollow or filled markers for horizontal
                else:
                    plt.plot(x,
                             y,
                             marker=outlier_marker,
                             color=outlier_color,
                             markerfacecolor='none'
                             if not outlier_fill else outlier_color,
                             linestyle='None'
                             )  # Hollow or filled markers for vertical

    plt.title(title if title else 'Box Plot')
    if grid:
        plt.grid(True)

    plt.tight_layout()

    # Save the figure if a path is provided
    if save_path:
        fig = plt.gcf()  # Get the current figure
        with save_lock:
            fig.savefig(save_path)


#### 12. bubble charts plot
def plot_bubble_chart(
        data,
        sizes=None,
        colors=None,
        labels=None,
        title=None,
        xlabel=None,
        ylabel=None,
        x_tick_angle=0,
        y_tick_angle=0,
        alpha=0.8,
        edge_color='black',
        color_map='viridis',
        show_legend=True,
        grid=False,  # New parameter for grid
        legend_title=None,  # New parameter for legend title
        colorbar_label=None,  # New parameter for colorbar label
        annotate=False,
        figsize=(12, 8),
        save_path=None):
    """
    Plots a bubble chart with customizable parameters.

    Parameters:
    - data: 2D array of shape (n, 2) representing the (x, y) coordinates of the bubbles. The scale of x and y should vary, like 0~100, 0~1000, 500~1000, etc.
    - sizes: 1D array of bubble sizes, should match the length of data (default: None). **Make sure that the size difference between bubbles is large, e.g. 50 vs 3000, 120 vs 4500, etc. The larger size ranges from 2000 to 5000, while the smaller size ranges from 0 to 500.**
    - colors: List of colors corresponding to each bubble (default: None). It should be 'None' when color_map is specified. It can have values sometimes (not always).
    - labels: List of labels for each bubble (default: None). Instances in the labels list should be completely different.
    - title: Title of the chart (default: None).
    - xlabel: Label for the x-axis (default: None), do not always start from 'Years'.
    - ylabel: Label for the y-axis (default: None).
    - x_tick_angle: Rotation angle for x-axis tick labels (default: 0), random select from [0, 30, 45, 90].
    - y_tick_angle: Rotation angle for y-axis tick labels (default: 0), random select from [0, 30, 45, 90].
    - alpha: Transparency level of the bubbles (0 to 1, default: 0.8). It should be random generated.
    - edge_color: Color of the bubble edges (default: 'black').
    - color_map: Colormap used if colors are not specified (default: 'viridis'). It should be 'None' sometimes (**must have value most of the time**) when colors is specified, do not always start from 'viridis'.
    - show_legend: Boolean indicating whether to show the legend (default: True). **It can be 'True' most of the time.***
    - grid: Boolean indicating whether to show a grid (default: False), it should be 'True' sometimes.
    - legend_title: Title for the legend (default: None).
    - colorbar_label: Label for the colorbar (default: None).
    - annotate: whether to annotate the label in the middle of the bubbles. **It can be 'True' sometimes when show_legend is 'False'**.
    - figsize: Tuple indicating the size of the figure (default: (12, 8)), should be dynamic, not always start from (12, 9), (10, 10), etc.
    - save_path: Path to save the figure (default: None).
    """

    plt.figure(figsize=figsize)

    # Create a scatter plot with specified colors or a color map
    if colors is None and color_map is not None:
        norm = mcolors.Normalize(vmin=min(sizes), vmax=max(sizes))
        scatter = plt.scatter(data[:, 0],
                              data[:, 1],
                              s=sizes,
                              c=sizes,
                              cmap=color_map,
                              alpha=alpha,
                              edgecolors=edge_color,
                              linewidth=1)
        if colorbar_label:
            plt.colorbar(scatter, label=colorbar_label)
        else:
            plt.colorbar(scatter)
    else:
        scatter = plt.scatter(data[:, 0],
                              data[:, 1],
                              s=sizes,
                              c=colors,
                              alpha=alpha,
                              edgecolors=edge_color,
                              linewidth=1)

    # Set labels for x and y axes
    plt.title(title if title else 'Bubble Chart')
    plt.xlabel(xlabel if xlabel else 'X-axis')
    plt.ylabel(ylabel if ylabel else 'Y-axis')

    # Set x and y ticks
    plt.xticks(rotation=x_tick_angle)
    plt.yticks(rotation=y_tick_angle)

    # Create a legend if specified
    if show_legend and labels is not None and annotate is False:
        unique_labels = list(set(labels))
        unique_colors = list(
            set(colors)) if colors is not None else plt.get_cmap(color_map)(
                np.linspace(0, 1, len(unique_labels)))

        # Create a legend handle for each unique label
        legend_handles = [
            plt.Line2D([0], [0],
                       marker='o',
                       color='w',
                       label=label,
                       markerfacecolor=unique_colors[i]
                       if colors is not None else unique_colors[i],
                       markersize=10) for i, label in enumerate(unique_labels)
        ]
        plt.legend(handles=legend_handles, title=legend_title)

    if annotate:
        for i, label in enumerate(labels):
            plt.annotate(label, (data[i, 0], data[i, 1]),
                         fontsize=9,
                         ha='center',
                         va='center')

    # Show grid if specified
    if grid:
        plt.grid()

    plt.tight_layout()

    # Save the figure if a path is provided
    if save_path:
        fig = plt.gcf()  # Get the current figure
        with save_lock:
            fig.savefig(save_path)


#### 13. candlestick charts plot
def plot_candlestick_chart(dates,
                           opens,
                           highs,
                           lows,
                           closes,
                           volumes,
                           title=None,
                           ylabel='Price',
                           xlabel='Date',
                           style='classic',
                           enable_volume=True,
                           x_ticks=None,
                           x_tick_rotation=0,
                           colors=None,
                           yaxis_range=None,
                           margin=None,
                           grid=True,
                           figsize=(12, 8),
                           save_path=None):
    """
    Plots a candlestick chart with customizable parameters.

    Parameters: (**The opens, highs and lows must not be the random, give the specific and concrete values.**)
    - dates: List of dates corresponding to the candlestick data.
    - opens: List of opening prices for the corresponding dates.
    - highs: List of highest prices for the corresponding dates.
    - lows: List of lowest prices for the corresponding dates.
    - closes: List of closing prices for the corresponding dates.
    - volumes: List of trading volume for the corresponding dates. **Please make sure it must have values when enable_volume is 'True'.**
    - title: Title of the chart (default: None).
    - ylabel: Label for the y-axis (default: 'Price').
    - xlabel: Label for the x-axis (default: 'Date').
    - style: Style of the chart, can be 'classic', 'yahoo', etc. (default: 'classic').
    - enable_volume: Whether to include a volume subplot (default: True). **Please make sure sometimes it can be 'True'**, but most of the time it should be 'False'.
    - x_ticks: Custom x-tick labels (default: None).
    - x_tick_rotation: Rotation angle for x-tick labels (default: 0), should be dynamic, varies in [0, 30, 45, 60, 90], etc, sometimes it can be 0 or 90.
    - colors: Dictionary specifying colors for up and down candles (default: None).
    - yaxis_range: Range for the y-axis as a list [min, max] (default: None).
    - margin: Dictionary specifying margins for the plot (default: None).
    - grid: Boolean indicating whether to show grid lines.
    - figsize: Size of the figure as a tuple (width, height) (default: (12, 8)), should be dynamic, not always start from (12, 8), (10, 10), etc.
    - save_path: Path to save the figure (default: None).
    """
    # Create DataFrame from input lists
    data = pd.DataFrame({
        'Date': pd.to_datetime(dates),
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'Volume': volumes
    })

    # Set the index to the Date column for mplfinance
    data.set_index('Date', inplace=True)

    # Define custom colors for up and down candles
    if colors is None:
        colors = {
            'up': '#1f77b4',
            'down': '#ff7f0e'
        }  # Blue for up, orange for down

    # Prepare the style for the candlestick chart
    mc = mpf.make_marketcolors(up=colors['up'],
                               down=colors['down'],
                               inherit=True)
    s = mpf.make_mpf_style(marketcolors=mc, base_mpf_style=style)

    # Create a new figure
    if enable_volume:
        fig, (ax,
              ax_volume) = plt.subplots(2,
                                        1,
                                        figsize=figsize,
                                        sharex=True,
                                        gridspec_kw={'height_ratios': [5, 1]})
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax_volume = None

    # Plotting the candlestick chart
    mpf.plot(
        data,
        type='candle',
        style=s,
        volume=enable_volume
        and ax_volume,  # Pass the volume axis only if enabled
        ax=ax,
        show_nontrading=False,
        xrotation=x_tick_rotation
    )  # Apply x-tick rotation directly in the plot function

    # Set the title if provided
    if title:
        ax.set_title(title)

    # Set x-axis and y-axis labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Customize x-ticks if provided
    if x_ticks is not None:
        ax.set_xticks(range(len(x_ticks)))
        ax.set_xticklabels(x_ticks, rotation=x_tick_rotation)

    # Set y-axis label position to the left
    ax.yaxis.tick_left()
    ax.yaxis.set_label_position("left")

    # Set y-axis range if provided
    if yaxis_range is not None:
        ax.set_ylim(yaxis_range)

    # Set margins if provided
    if margin is not None:
        plt.subplots_adjust(left=margin['l'],
                            right=margin['r'],
                            top=margin['t'],
                            bottom=margin['b'])

    # Show grid if specified
    if grid:
        plt.grid(True)

    plt.tight_layout()

    # Save the figure if a path is provided
    if save_path:
        fig = plt.gcf()  # Get the current figure
        with save_lock:
            fig.savefig(save_path)


#### 14. heatmap charts plot
def plot_heatmap_chart(data,
                       title=None,
                       xlabel=None,
                       ylabel=None,
                       xticks=None,
                       yticks=None,
                       xtickangle=0,
                       ytickangle=0,
                       cmap='viridis',
                       annot=False,
                       fmt=None,
                       linewidths=0.5,
                       linecolor='black',
                       colorbar=True,
                       use_circles=False,
                       figsize=(10, 8),
                       save_path=None):
    """
    Plots a heatmap, optionally using circular markers.

    Parameters:
    - data: 2D array representing the data for the heatmap.
    - title: Title for the heatmap.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    - xticks: List of x-axis tick labels. The ticks must be **specific terms** (MUST NOT be Common label + numbers / letters, like country in xlabel, ticks should not be ['country A', 'country B'] or ['country 1', 'country 2'], should be like ['USA', 'China', 'Russia']) and consistent with xlabel.
    - yticks: List of y-axis tick labels. The ticks must be **specific terms** (MUST NOT be Common label + numbers / letters, like country in xlabel, ticks should not be ['country A', 'country B'] or ['country 1', 'country 2'], should be like ['USA', 'China', 'Russia']) and consistent with ylabel.
    - xtickangle: Angle for x-axis tick labels, should be dynamic, varies in [0, 30, 45, 90], etc, sometimes it can be 0 or 90.
    - ytickangle: Angle for y-axis tick labels, should be dynamic, varies in [0, 30, 45, 90], etc, sometimes it can be 0 or 90.
    - cmap: Color map for the heatmap.
    - annot: Boolean indicating whether to annotate cells with values. Please make sure **it can be 'False'**, not always be 'True'.
    - fmt: Format for annotations.
    - linewidths: Line widths for cell borders.
    - linecolor: Color of the cell borders.
    - colorbar: Boolean indicating whether to show the colorbar. Please make sure just sometimes (not always) it can be 'False' only when annot is 'True'.
    - use_circles: Boolean indicating whether to use circular markers instead of rectangles. Please make sure just sometimes (not always) it can be 'True'.
    - figsize: Tuple indicating the figure size. It should be dynamic, not always start from a fixed size.
    - save_path: Path to save the figure.
    """

    plt.figure(figsize=figsize)
    ax = plt.gca()

    # Set up the colormap and norm (log scale if needed)
    norm = LogNorm(vmin=np.min(data[data > 0]), vmax=np.max(data))

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            color = plt.cm.get_cmap(cmap)(norm(data[i, j]))

            if use_circles:
                # Draw a circle
                circle = plt.Circle((j, i), 0.4, color=color)
                ax.add_artist(circle)

                # Calculate the midpoint for adaptive text color
                data_min = np.min(data)
                data_max = np.max(data)
                midpoint = (data_min + data_max) / 2

                # In the annotation loop
                text_color = "white" if data[i, j] > midpoint else "black"

                # Add the text inside the circle
                if annot:
                    ax.text(j,
                            i,
                            f"{data[i, j]:{fmt}}",
                            ha="center",
                            va="center",
                            color=text_color)
            else:
                # Create a rectangle (standard heatmap)
                rect = plt.Rectangle((j - 0.5, i - 0.5),
                                     1,
                                     1,
                                     color=color,
                                     linewidth=linewidths,
                                     edgecolor=linecolor)
                ax.add_artist(rect)

                # Calculate the midpoint for adaptive text color
                data_min = np.min(data)
                data_max = np.max(data)
                midpoint = (data_min + data_max) / 2

                # In the annotation loop
                text_color = "white" if data[i, j] > midpoint else "black"

                if annot:
                    ax.text(j,
                            i,
                            f"{data[i, j]:{fmt}}",
                            ha="center",
                            va="center",
                            color=text_color)

    # Set labels for x and y axes
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels(xticks, rotation=xtickangle)
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels(yticks, rotation=ytickangle)

    # Adding titles for the axes
    ax.set_title(title if title else 'Heatmap')
    ax.set_xlabel(xlabel if xlabel else 'X-axis')
    ax.set_ylabel(ylabel if ylabel else 'Y-axis')

    # Set the limits of the axes
    ax.set_xlim(-0.5, data.shape[1] - 0.5)
    ax.set_ylim(-0.5, data.shape[0] - 0.5)
    ax.set_aspect('equal')

    # Create a colorbar
    if colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax)

    plt.tight_layout()

    # Save the figure if a path is provided
    if save_path:
        fig = plt.gcf()
        fig.savefig(save_path)


#### 15. radar charts plot
def plot_radar_chart(data_list,
                     labels=None,
                     title=None,
                     colors=None,
                     alpha=0.75,
                     fill=True,
                     fill_colors=None,
                     border_styles=None,
                     border_shape='circle',
                     legend_labels=None,
                     show_data_points=True,
                     figsize=(8, 8),
                     show_grid=True,
                     save_path=None):
    """
    Plots multiple radar charts with customizable options.

    Parameters:
    - data_list: List of lists or arrays containing the data values for each area. The data scale **must vary**, **not always from 0~10 or 0~100**, can be other scales like 0~1000, 500~800. **Do not generate numbers too close to prevent overlap and ensure clarity.**
    - labels: List of labels for each axis (optional). **The length of labels must be consistent with the length of variable in the data_list**.
    - title: Title for the plot (optional).
    - colors: List of colors for the borders of the radar plots (optional).
    - alpha: Transparency level for the radar fill (optional, default is 0.75).
    - fill: Boolean indicating whether to fill the radar charts (default is True). Please make sure it can be 'True', **but not always**. 
    - fill_colors: List of colors to fill each radar chart if fill is True (optional).
    - border_styles: List of line styles for the borders ('solid', 'dashed', 'dotted') (default is 'solid'). The styles of different lines in the same chart can be the same, for example, they are all 'dashed'. Please ensure a certain proportion.
    - border_shape: Shape of the radar chart border ('circle', 'rectangle', 'pentagon', 'hexagon') (default is 'circle'). Please **random select** a shape, but make sure **it's more likely to select a 'circle' at the most of the time**. Please note the 'rectangle', 'pentagon', and 'hexagon' correspond to data_list lengths of 4, 5, and 6 respectively, and 'circle' is unrestricted.
    - legend_labels: List of labels for the legend. The legend_labels must be **specific terms** (MUST NOT be Common label + numbers / letters, legends must not be ['series A', 'series B', 'series X', 'series Y'] or ['series 1', 'series 2'], **must be concrete** like ['creativity', 'passion']).
    - show_data_points: Boolean to indicate if data points should be marked (default is 'True'). **Please make sure it can be 'False', not always 'True'**.
    - figsize: Tuple indicating the figure size (optional, default is (8, 8)), should be dynamic, not always start from (12, 9), (8, 8), etc.
    - show_grid: Boolean indicating whether to show radial grid lines (default is True). **Particularly, when border_shape is 'circle', the show_grid should be 'False'.**
    - save_path: Path to save the figure (optional).
    """

    # Determine number of variables based on the border shape
    shape_sides = {'rectangle': 4, 'pentagon': 5, 'hexagon': 6}

    if border_shape == 'circle':
        num_vars = len(data_list[0])  # Use the length of the first data entry
    else:
        num_vars = shape_sides.get(border_shape,
                                   6)  # Default to hexagon if not recognized

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    # Create the figure
    plt.figure(figsize=figsize)
    ax = plt.subplot(111, polar=True)

    # Adjust the length of the data for consistent plotting
    for i, data in enumerate(data_list):
        # Ensure data length matches num_vars
        if len(data) != num_vars:
            raise ValueError(
                f"Data length {len(data)} does not match the number of variables {num_vars}."
            )

        # Scale data for the specified shape
        data = np.concatenate((data, [data[0]]))  # Close the loop

        # Fill the area if specified
        if fill:
            fill_color = fill_colors[i] if fill_colors and i < len(
                fill_colors) else 'blue'
            ax.fill(angles, data, color=fill_color, alpha=alpha)

        # Plot the border
        border_color = colors[i] if colors and i < len(colors) else 'blue'
        border_style = border_styles[i] if border_styles and i < len(
            border_styles) else 'solid'
        ax.plot(angles,
                data,
                color=border_color,
                linestyle='-' if border_style == 'solid' else
                '--' if border_style == 'dashed' else ':',
                linewidth=2,
                label=legend_labels[i]
                if legend_labels and i < len(legend_labels) else None)

        # Draw data points if specified
        if show_data_points:
            for j in range(num_vars):
                ax.plot(angles[j],
                        data[j],
                        'o',
                        color=border_color,
                        markersize=6)

    # Add labels to the axes
    if labels:
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)

    # Hide the axes if the border shape is not 'circle'
    if border_shape != 'circle':
        ax.yaxis.grid(False)
        ax.spines['polar'].set_visible(False)

    # Customize title
    plt.title(title if title else 'Multiple Radar Charts',
              size=15,
              color='black')

    # Configure radial grid lines if required
    if show_grid and border_shape != 'circle':
        max_radius = np.max([np.max(data) for data in data_list])
        for i in range(1, len(data_list) + 1):
            radius = np.full_like(angles, (i * max_radius / len(data_list)))
            ax.plot(angles,
                    radius,
                    color='grey',
                    linewidth=1,
                    linestyle='dashed')

    # Add the legend if labels are provided
    if legend_labels:
        ax.legend(loc='best', bbox_to_anchor=(1.1, 1.1))

    plt.tight_layout()

    # Save the figure if a path is provided
    if save_path:
        fig = plt.gcf()  # Get the current figure
        with save_lock:
            fig.savefig(save_path)


#### 16. rose charts plot
def plot_rose_chart(data_labels,
                    data,
                    title='Rose Plot',
                    colors=None,
                    edge_color='black',
                    alpha=1.0,
                    figsize=(10, 10),
                    save_path=None):
    """
    Plots a rose plot (polar plot) that fills the entire 360°.

    Parameters:
    - data_labels: List of labels for each category.
    - data: List of values corresponding to each category (the length of data should >= 7 at most cases, not limited to 6. The values differences (e.g., > 50) and scales (e.g, 0 ~ 100, 0~200, 0 ~ 1000) between the data should be larger).
    - title: Title for the plot.
    - colors: List of colors for each category.
    - edge_color: Color of the edges of the bars.
    - alpha: Opacity of the bars (0 to 1).
    - figsize: Tuple indicating the figure size, should be dynamic, not always start from (12, 9), (14, 14), (10, 10), (12, 12), etc.
    - save_path: Path to save the figure.
    """
    num_categories = len(data_labels)
    sector_angle = (2 * np.pi) / num_categories

    plt.figure(figsize=figsize)
    ax = plt.subplot(111, projection='polar')

    # Draw the bars for each category
    for i in range(num_categories):
        ax.bar(x=i * sector_angle,
               width=sector_angle,
               height=data[i],
               color=colors[i] if colors else plt.cm.Set1(i / num_categories),
               edgecolor=edge_color,
               alpha=alpha,
               label=data_labels[i])

    # Set labels and title
    ax.set_xticks(np.linspace(0, 2 * np.pi, num_categories, endpoint=False))
    ax.set_xticklabels(data_labels, fontsize=12, rotation=90)
    ax.set_title(title, fontsize=14)

    # Add legend
    ax.legend(bbox_to_anchor=(1.2, 1.0))

    plt.tight_layout()

    # Save the figure if a path is provided
    if save_path:
        fig = plt.gcf()  # Get the current figure
        with save_lock:
            fig.savefig(save_path)


#### 17. quiver charts plot
def plot_quiver_chart(X,
                      Y,
                      U,
                      V,
                      title="Quiver Plot",
                      xlabel="X-axis",
                      ylabel="Y-axis",
                      color_map="viridis",
                      show_streamlines=False,
                      streamline_color='black',
                      legend_labels=None,
                      grid=False,
                      show_colorbar=True,
                      colorbar_label=None,
                      figsize=(8, 6),
                      save_path=None):
    """
    Create a quiver plot to visualize vector fields.

    Parameters: (**Sometimes X, Y, U, V can be concrete values**, but most of the time they should be generated by a function)
    - X: 2D array of x-coordinates for the arrows.
    - Y: 2D array of y-coordinates for the arrows.
    - U: 2D array of x-component of the vectors (arrows).
    - V: 2D array of y-component of the vectors (arrows).
    - title (str): Title of the plot (default is "Quiver Plot").
    - xlabel (str): Label for the x-axis (default is "X-axis").
    - ylabel (str): Label for the y-axis (default is "Y-axis").
    - color_map (str): Colormap for the arrows based on their magnitude (default is "viridis").
    - show_streamlines (bool): If True, display streamlines on the plot (default is False).
    - streamline_color (str): Color of the streamlines if shown (default is 'black').
    - legend_labels (list): Labels for the legend if arrow_labels are provided (default is None). Most of the time, it should be 'None'.
    - grid (bool): If True, show a grid on the plot (default is False).
    - show_colorbar (bool): If True, display a colorbar for the quiver plot (default is True). **Please make sure it can be 'False' sometimes.**
    - colorbar_label (str): Label for the colorbar (default is None).
    - figsize (tuple): Size of the figure in inches (default is (8, 6)), should be dynamic, not always start from (12, 9), (8, 6), etc.
    - save_path (str): File path to save the figure (default is None, meaning it won't be saved).    
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Set background color to white
    ax.set_facecolor('white')

    # Quiver plot
    colors = np.sqrt(U**2 + V**2)  # Magnitude for color mapping
    quiver = ax.quiver(X, Y, U, V, colors, cmap=color_map)

    # Add streamlines if specified
    if show_streamlines:
        ax.streamplot(X, Y, U, V, color=streamline_color, linewidth=0.5)

    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if show_colorbar:
        cbar = plt.colorbar(quiver)
        if colorbar_label:
            cbar.set_label(colorbar_label)  # Set the colorbar label

    # Add legend if labels are provided
    if legend_labels:
        handles = []
        for legend_label in legend_labels:
            arrow = Line2D([0], [0],
                           marker='>',
                           color='black',
                           markersize=10,
                           label=legend_label,
                           linestyle='None')
            handles.append(arrow)
        ax.legend(handles=handles, labels=legend_labels, loc='best')

    # Show grid if specified
    if grid:
        ax.grid(True, linestyle="--", alpha=0.5)

    # Adjust the aspect ratio
    ax.set_aspect("equal")

    plt.tight_layout()

    # Save the figure if a path is provided
    if save_path:
        fig = plt.gcf()  # Get the current figure
        with save_lock:
            fig.savefig(save_path)


#### 18. 3d charts plot
def plot_3d_chart(data,
                  plot_type,
                  labels,
                  title,
                  colors,
                  bar_width,
                  elev,
                  azim,
                  bar_alpha=1.0,
                  tick_angle_x=0,
                  tick_angle_y=0,
                  scatter_marker='o',
                  figsize=(9, 8),
                  save_path=None):
    """
    Plot a 3D chart of a specified type. Please note that the generated data should **strictly follow the example provided.**
    **ATTENTION: Random select a chart type, should not always be a fixed value like 'bar' or 'line'.**

    Parameters:
    - data: Data for the chart (can be different formats based on plot_type). 
      - For bar: (x_data, y_data, z_data)
      - For scatter: (x_data, y_data, z_data)
      - For line: (x_data, [y_data_series], [z_data_series])
      - For surface: (x_data, y_data, z_data) where x_data and y_data are 2D arrays
      - For skeleton: (x_data, y_data, z_data, segments) where segments is a list of tuples of indices to connect
      - For density: (x_data, y_data, z_data) where x_data and y_data are 1D lists, and z_data is a 2D list
    - plot_type: Type of chart (e.g., 'bar', 'scatter', 'line', 'surface', 'skeleton', 'density').
    - labels: Labels for each axis (list of length 3). 
    - title: Title for the chart.
    - colors: Colors for the chart (list of colors).
    - bar_width: Width of the bars for bar plots.
    - elev: Elevation angle for the 3D view.
    - azim: Azimuthal angle for the 3D view.
    - bar_alpha: Transparency level for bars (0 to 1).
    - tick_angle_x: Rotation angle for X-axis tick labels (default is 0), should be dynamic, varies in [0, 30, 45, 90], etc, it **should be 0 at the most of the time**.
    - tick_angle_y: Rotation angle for Y-axis tick labels (default is 0), should be dynamic, varies in [0, 30, 45, 90], etc, it **should be 0 at the most of the time**.
    - scatter_marker: Marker style for scatter plots (when plot_type='scatter'/ default is 'o').
    - figsize: Size of the figure, can be diverse.
    - save_path: Path to save the figure (optional).
    """

    def polygon_under_graph(x, y):
        """Construct the vertex list to fill the space under the (x, y) line graph."""
        return [(x[0], 0.0), *zip(x, y), (x[-1], 0.0)]

    # Create the figure with a white background
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor('white')

    # Create a 3D axis
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')

    # Handle different plot types
    if plot_type == 'bar':
        x_data, y_data, z_data = data
        x_data_indices = np.arange(len(x_data))
        y_data_indices = np.arange(len(y_data))

        colors = colors * (len(x_data) * len(y_data) // len(colors) + 1)

        for i, x in enumerate(x_data_indices):
            for j, y in enumerate(y_data_indices):
                ax.bar3d(x,
                         y,
                         0,
                         bar_width,
                         bar_width,
                         z_data[i][j],
                         color=colors[i * len(y_data) + j],
                         alpha=bar_alpha,
                         zsort='average')

        ax.set_xticks(x_data_indices)
        ax.set_xticklabels(x_data)
        ax.set_yticks(y_data_indices)
        ax.set_yticklabels(y_data)

    elif plot_type == 'scatter':
        x_data, y_data, z_data = data
        ax.scatter(x_data,
                   y_data,
                   z_data,
                   c=colors if colors else None,
                   marker=scatter_marker)

    elif plot_type == 'line':
        x_data, y_data_series, z_data_series = data
        for j in range(len(y_data_series)):
            ax.plot(x_data,
                    y_data_series[j],
                    z_data_series[j],
                    color=colors[j] if colors and j < len(colors) else None)

    elif plot_type == 'surface':
        x_data, y_data, z_data = data
        x_data, y_data = np.meshgrid(x_data, y_data)
        z_data = np.array(z_data)
        cmap = colors[0] if colors and len(colors) > 0 else 'viridis'
        surface = ax.plot_surface(x_data,
                                  y_data,
                                  z_data,
                                  cmap=cmap,
                                  edgecolor='none')
        fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10)

    elif plot_type == 'skeleton':
        x_data, y_data, z_data, segments = data
        ax.scatter(x_data, y_data, z_data, c=colors if colors else 'black')
        for seg in segments:
            ax.plot([x_data[seg[0]], x_data[seg[1]]],
                    [y_data[seg[0]], y_data[seg[1]]],
                    [z_data[seg[0]], z_data[seg[1]]],
                    color=colors[0] if colors else 'black')

    elif plot_type == 'density':
        x_data, y_data, z_data = data
        verts = [
            polygon_under_graph(x_data, z_data[i]) for i in range(len(z_data))
        ]
        facecolors = plt.cm.viridis(np.linspace(0, 1, len(verts)))
        poly = PolyCollection(verts, facecolors=facecolors, alpha=0.7)
        ax.add_collection3d(poly, zs=y_data, zdir="y")
        norm = plt.Normalize(vmin=0, vmax=len(verts) -
                             1)  # Normalize values for color mapping
        cmap = colors[0] if colors and len(colors) > 0 else 'viridis'
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Empty array for the colorbar
        fig.colorbar(sm, ax=ax, shrink=0.5, aspect=10)

    ax.set_xlabel(labels[0] if labels and len(labels) > 0 else 'X-axis')
    ax.set_ylabel(labels[1] if labels and len(labels) > 1 else 'Y-axis')
    ax.set_zlabel(labels[2] if labels and len(labels) > 2 else 'Z-axis')
    ax.set_title(title if title else '3D Chart')
    ax.view_init(elev=elev, azim=azim)

    for tick in ax.get_xticklabels():
        tick.set_rotation(tick_angle_x)
    for tick in ax.get_yticklabels():
        tick.set_rotation(tick_angle_y)

    plt.tight_layout()

    if save_path:
        fig = plt.gcf()
        with save_lock:
            fig.savefig(save_path)


#### 19. errorbar charts plot
def plot_error_bar_chart(data_x,
                         data_y,
                         errors,
                         bar_width,
                         colors,
                         title,
                         xlabel,
                         ylabel,
                         ylim,
                         ticks,
                         xtick_angle,
                         ytick_angle,
                         grid,
                         legend_labels,
                         orientation='vertical',
                         figsize=(10, 6),
                         save_path=None):
    """
    Plots a customizable error bar chart comparing multiple sets of data, which **not reflects the time relationship**.

    Parameters:
    - data_x: List of model names or categories for the x-axis. The data_x must be **specific terms** (MUST NOT be Common label + numbers / letters, like country in xlabel, ticks should not be ['country A', 'country B'] or ['country 1', 'country 2'], should be like ['USA', 'China', 'Russia']) and consistent with xlabel. 
    - data_y: List of lists, where each inner list contains accuracy values (one for each category).
    - errors: List of lists, where each inner list contains error values corresponding to the accuracies.
    - bar_width: Width of the bars in the chart.
    - colors: List of colors for each series of bars, should not always start from 'blue', 'green' and 'orange'.
    - title: Title of the chart.
    - xlabel: Label for the x-axis.  
    - ylabel: Label for the y-axis.
    - ylim: Limits for the y-axis, specified as a list [min, max].
    - ticks: Custom tick values for the axis. 
    - xtick_angle: Angle for the x-axis ticks, useful for better visibility, should be dynamic, varies in [0, 30, 45, 60, 90], etc, **sometimes it should be 0 or 90**.
    - ytick_angle: Angle for the y-axis ticks, should be dynamic, varies in [0, 30, 45, 60, 90], etc, **sometimes it should be 0 or 90**.
    - grid: Boolean indicating whether to show grid lines on the chart.
    - legend_labels: List of legend labels corresponding to each series (or a single label if there's only one series). The legend_labels must be **specific terms** (must not be Common label + numbers / letters, legends should not be ['Sample A', 'Sample B'] or ['Sample 1', 'Sample 2'], should be like ['creativity', 'passion']).
    - orientation: 'vertical' or 'horizontal' to specify the orientation of the bars. **Please make sure it can be 'horizontal', **not always be 'vertical'.**
    - figsize: Tuple indicating the size of the figure (width, height) in inches, should be dynamic, not always start from (12, 9), (10, 10), etc.
    - save_path: Optional; if provided, saves the figure to the specified path.
    """

    plt.figure(figsize=figsize)

    # Determine the number of series
    num_series = len(data_y)

    if orientation == 'horizontal':
        # Create horizontal bars
        num_models = len(data_x)
        bar_positions = np.arange(num_models)  # Base positions for each model

        for i in range(num_series):
            plt.barh(
                bar_positions +
                (i * (bar_width + 0.05)),  # Add a small gap to avoid overlap
                data_y[i],
                color=colors[i],
                xerr=errors[i],
                capsize=7,
                label=legend_labels[i],
                height=bar_width  # Adjust height for horizontal bars
            )

        plt.yticks(bar_positions + (num_series - 1) * (bar_width + 0.05) / 2,
                   data_x,
                   rotation=ytick_angle)  # Center y-ticks
        plt.xlabel(ylabel)
        plt.ylabel(xlabel)
        plt.title(title)
        plt.xlim(ylim)
        plt.xticks(rotation=xtick_angle)

    elif orientation == 'vertical':
        # Create vertical bars
        for i in range(num_series):
            bar_positions = np.arange(len(
                data_x)) + i * bar_width  # Adjust positions for each series
            plt.bar(bar_positions,
                    data_y[i],
                    color=colors[i],
                    width=bar_width,
                    label=legend_labels[i],
                    yerr=errors[i],
                    capsize=7)
        plt.xticks(np.arange(len(data_x)) + (num_series - 1) * bar_width / 2,
                   data_x,
                   rotation=xtick_angle)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.ylim(ylim)
        plt.yticks(ticks, rotation=ytick_angle)

    if grid:
        plt.grid(color="gray", linestyle="-", linewidth=0.5, axis="both")
        plt.gca().set_axisbelow(True)

    # Set legend based on the number of series
    if num_series == 1:
        plt.legend([legend_labels[0]], frameon=True,
                   loc="best")  # Updated for single series
    else:
        plt.legend(frameon=True, loc="best")

    plt.gca().set_facecolor("white")

    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.tight_layout()

    # Save the figure if a path is provided
    if save_path:
        fig = plt.gcf()  # Get the current figure
        with save_lock:
            fig.savefig(save_path)


#### 20. scatter charts plot
def plot_scatter_chart(data_list,
                       labels=None,
                       xlabel='X-axis',
                       ylabel='Y-axis',
                       title='Scatter Plot',
                       colors=None,
                       markers=None,
                       sizes=None,
                       alpha=0.7,
                       legend=True,
                       grid=True,
                       x_tick_rotation=0,
                       point_labels=None,
                       clustered=False,
                       figsize=(8, 6),
                       save_path=None):
    """
    Draws a scatter plot for the given data. (Different points can have the same marker but different colors or same color but different markers.)
    (If the points are clustered, please ensure to generate points that follow the **same pattern** in a chart or clustering together)

    Parameters:
    - data_list: List of datasets to be plotted. Each dataset should be a tuple containing two lists: x-coordinates and y-coordinates. **Sometimes should use the 'np.linspace' or 'np.random.normal'**, but **not always** (sometimes just give the specific values).
    - xlabel: String for the label of the x-axis, **should not always be the time-related concepts like 'Years'**.
    - ylabel: String for the label of the y-axis.
    - title: String for the title of the scatter plot.
    - colors: List of colors for each dataset. The number of colors should match the number of datasets in data_list.
    - markers: List of marker styles for each dataset. The number of markers should match the number of datasets in data_list. 
    - sizes: List of sizes for the markers for each dataset. The number of sizes should match the number of datasets in data_list.
    - alpha: Float between 0 and 1 indicating the transparency level of the points. 0 is fully transparent, and 1 is fully opaque.
    - legend: Boolean indicating whether to display the legend. Defaults to True.
    - grid: Boolean indicating whether to display a grid on the plot. Defaults to True.
    - x_tick_rotation: Integer specifying the angle (in degrees) to rotate the x-axis tick labels, should be dynamic, varies in [0, 30, 45, 60, 90], etc, most of the time it should be 0 or 90.
    - point_labels: List of labels for individual points. This can be a list of lists (for multiple points in a group) or a single list of labels. **Make sure it can be 'None' sometimes.**
    - clustered: Boolean indicating whether to use a clustered scatter style, where all points in a group are plotted together with a single style.
    - figsize: Tuple specifying the size of the figure as (width, height) in inches, should be dynamic, not always start from (12, 9), (10, 10), etc.
    - save_path: String specifying the path where the plot will be saved. If None, the plot will be displayed instead.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("white")

    if clustered:
        # Clustered scatter style
        for i, (x, y) in enumerate(data_list):
            color = colors[i] if colors is not None else None
            marker = markers[i] if markers is not None else 'o'
            size = sizes[i] if sizes is not None else 50

            ax.scatter(x,
                       y,
                       color=color,
                       alpha=alpha,
                       label=labels[i] if labels else None)

    else:
        # Standard scatter style
        for i, (x, y) in enumerate(data_list):
            color = colors[i] if colors is not None else None
            marker = markers[i] if markers is not None else 'o'
            size = sizes[i] if sizes is not None else 50

            scatter = ax.scatter(x,
                                 y,
                                 color=color,
                                 marker=marker,
                                 s=size,
                                 alpha=alpha,
                                 label=labels[i] if labels else None)

            # Annotate points if specified
            if point_labels is not None:
                if isinstance(point_labels[i], list):
                    for j, (xi, yi) in enumerate(zip(x, y)):
                        label = point_labels[i][j] if j < len(
                            point_labels[i]) else None
                        if label is not None:
                            ax.annotate(label, (xi, yi),
                                        textcoords="offset points",
                                        xytext=(0, 10),
                                        ha='center')
                else:
                    label = point_labels[i]
                    if label is not None:
                        ax.annotate(label, (x[0], y[0]),
                                    textcoords="offset points",
                                    xytext=(0, 10),
                                    ha='center')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if x_tick_rotation != 0:
        for tick in ax.get_xticklabels():
            tick.set_rotation(x_tick_rotation)

    if grid:
        ax.grid(True, linestyle='--', linewidth=0.5, color='gray')

    if legend and labels:
        ax.legend()

    plt.tight_layout()

    # Save the figure if a path is provided
    if save_path:
        fig = plt.gcf()  # Get the current figure
        with save_lock:
            fig.savefig(save_path)


#### 21. violin charts plot
def plot_violin_chart(
        data_list,
        xticklabels,
        xlabel,
        ylabel,
        ylim,
        xticks,
        title='Violin Plot',
        colors=None,
        edgecolor="#333333",
        linewidth=1.5,
        xlabel_rotation=0,
        show_grid=True,
        show_means=False,
        mean_label_angle=0,
        mean_line_visible=True,
        mean_offset=0.1,
        split=False,
        custom_legends=None,  # New parameter for custom legends
        figsize=(6, 4),
        save_path=None):
    """
    Draws a violin plot or a split violin plot for the given data.

    Parameters:
    - data_list: List of datasets to be plotted. **When Split='True', must be of length 2.**
    - xticklabels: List of labels for the x-axis, corresponding to each dataset in data_list.
    - xlabel: String for the label of the x-axis.
    - ylabel: String for the label of the y-axis.
    - ylim: List of two floats specifying the limits for the y-axis as [min, max]. 
    - xticks: List of positions on the x-axis where the violins will be plotted. The xticks must be **specific terms** (MUST NOT be Common label + numbers / letters, **must be concrete**).
    - title: String for the title of the plot.
    - colors: List of colors for the violins. Should match the number of datasets.
    - edgecolor: String for the color of the edges of the violins.
    - linewidth: Float indicating the width of the edges of the violins.
    - xlabel_rotation: Float for the rotation angle of the x-axis labels, should be dynamic, varies in [0, 30, 45, 60, 90], etc, most of the time it can be 0 or 90.
    - show_grid: Boolean indicating whether to display a grid on the y-axis.
    - show_means: Boolean indicating whether to show the mean of each dataset on the plot. Please make sure it can be 'False' sometimes, not always be 'True'. **When split='True', it must be 'False'!**
    - mean_label_angle: Float for the rotation angle of the mean labels, should be dynamic, varies in [0, 30, 45, 60, 90], etc, most of the time it should be 0.
    - mean_line_visible: Boolean indicating whether to show a line indicating the mean on the plot. Please make sure it can be 'False'.
    - mean_offset: Float specifying the offset for mean labels from the data points.
    - split: Boolean indicating whether to generate a split violin plot comparing two datasets. **Please make sure it can be 'True' sometimes, not always be 'False'.**
    - custom_legends: List of strings for custom legends in split plots. Must be of length 2. The custom_legends must be **specific terms** (MUST NOT be Common label + numbers / letters, legends must not be ['series A', 'series B'] or ['dataset A', 'dataset B'], **must be concrete**).
    - figsize: Tuple specifying the size of the figure as (width, height) in inches, should be dynamic, not always start from (12, 9), (8, 6), etc.
    - save_path: String specifying the path where the plot will be saved. If None, the plot will be displayed instead.
    """

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("white")

    if split:
        # Ensure data is structured for split violin plot
        for i in range(len(xticklabels)):
            data_left = data_list[0][:, i]
            data_right = data_list[1][:, i]

            # Calculate KDE for left dataset
            kde_left = gaussian_kde(data_left)
            kde_x = np.linspace(0, 1, 300)
            kde_left_y = kde_left(kde_x)

            # Scale KDE values for plotting
            kde_left_y_scaled = kde_left_y / max(kde_left_y) * 0.2

            # Plot left half
            ax.fill_betweenx(kde_x,
                             -kde_left_y_scaled + xticks[i],
                             xticks[i],
                             color=colors[0],
                             edgecolor=edgecolor)

            # Calculate KDE for right dataset
            kde_right = gaussian_kde(data_right)
            kde_right_y = kde_right(kde_x)
            kde_right_y_scaled = kde_right_y / max(kde_right_y) * 0.2

            # Plot right half
            ax.fill_betweenx(kde_x,
                             xticks[i],
                             kde_right_y_scaled + xticks[i],
                             color=colors[1],
                             edgecolor=edgecolor)

        # Add legends for split violin plot
        if custom_legends and len(custom_legends) == 2:
            ax.fill_between(
                [], [], [], color=colors[0],
                label=custom_legends[0])  # Placeholder for left legend
            ax.fill_between(
                [], [], [], color=colors[1],
                label=custom_legends[1])  # Placeholder for right legend
    else:
        # Standard violin plot
        violin_parts = ax.violinplot(data_list,
                                     positions=xticks,
                                     showmeans=mean_line_visible)

        # Customize violin plot colors
        for i, color in enumerate(colors):
            violin_parts['bodies'][i].set_facecolor(color)
            violin_parts['bodies'][i].set_alpha(0.7)
            for partname in ("cmaxes", "cmins", "cbars"):
                vp = violin_parts[partname]
                vp.set_edgecolor(edgecolor)
                vp.set_linewidth(linewidth)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=xlabel_rotation)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(ylim)

    if show_grid:
        ax.yaxis.grid(True)
        plt.grid(True,
                 which='both',
                 linestyle='-',
                 linewidth=0.5,
                 color='gray')

    if show_means:
        means = [np.mean(data) for data in data_list]
        for i, mean in enumerate(means):
            ax.text(xticks[i] + mean_offset,
                    mean,
                    f'{mean:.2f}',
                    ha='center',
                    va='bottom',
                    rotation=mean_label_angle,
                    color=edgecolor,
                    fontsize=10)

    # Adding legend for split violin
    if split and custom_legends:
        ax.legend(loc='best')

    plt.tight_layout()

    # Save the figure if a path is provided
    if save_path:
        fig = plt.gcf()  # Get the current figure
        with save_lock:
            fig.savefig(save_path)


#### 22. contour charts plot
def plot_contour_chart(X,
                       Y,
                       Z,
                       contour_type='filled',
                       cmap='viridis',
                       title='',
                       xlabel='',
                       ylabel='',
                       contour_data=None,
                       legend_labels=None,
                       alpha=1.0,
                       levels=None,
                       show_colorbar=True,
                       colorbar_label=None,
                       line_colors=None,
                       annotate_lines=False,
                       figsize=(10, 6),
                       save_path=None):
    """
    Plots contour graphs based on the provided data and parameters.

    Parameters:
    - X: 2D array (meshgrid for x-coordinates)
    - Y: 2D array (meshgrid for y-coordinates)
    - Z: 2D array (values at each (x, y) coordinate)
    - contour_type: 'filled' for filled contours, 'lines' for contour lines. Please randomly select a value, not always from a fixed start point.
    - cmap: Colormap for the contour
    - title: Title of the plot
    - xlabel: Label for x-axis
    - ylabel: Label for y-axis
    - contour_data: List of tuples containing (X, Y, Z, color, label) for multiple contours. When contour_type='lines', please provide the contour_data, otherwise, it should be 'None'. **Please note that the instances in this list must not exceed 2**. **MUST NOT ADD ANY RANDOM NOISE!!!** 
    - legend_labels: List of labels for the legend (for multiple datasets). **When contour_type='filled', it must be 'None'**. The legend_labels must be **specific terms** (MUST NOT be Common label + numbers / letters, legend_labels should not be ['series A', 'series B'] or ['series 1', 'series 2']).
    - alpha: Transparency for filled contours
    - levels: Number of levels for contour
    - show_colorbar: Boolean to show color bar. 
    - colorbar_label (str): Label for the colorbar (default is None).
    - line_colors: List of colors for contour lines (only used if contour_type is 'lines')
    - annotate_lines: Boolean to annotate contour lines with their values
    - figsize: Tuple for figure size (width, height), should be dynamic, not always start from (12, 9), (8, 6), etc.
    - save_path: Path to save the figure
    """

    plt.figure(figsize=figsize)

    if contour_data is not None:
        for (x, y, z, color, label) in contour_data:
            if contour_type == 'filled':
                contour = plt.contourf(x,
                                       y,
                                       z,
                                       levels=levels,
                                       cmap=cmap,
                                       alpha=alpha)
            else:
                contour = plt.contour(x, y, z, levels=levels, colors=color)

                # Annotate contour lines with their values
                if annotate_lines:
                    plt.clabel(contour, inline=True, fontsize=8, fmt='%.1f')
    else:
        if contour_type == 'filled':
            contour = plt.contourf(X,
                                   Y,
                                   Z,
                                   levels=levels,
                                   cmap=cmap,
                                   alpha=alpha)
        else:
            contour = plt.contour(X,
                                  Y,
                                  Z,
                                  levels=levels,
                                  colors=line_colors or cmap)

            # Annotate contour lines with their values
            if annotate_lines:
                plt.clabel(contour, inline=True, fontsize=8, fmt='%.1f')

    # Adding color bar only if contour type is 'filled'
    if show_colorbar and contour_type == 'filled':
        cbar = plt.colorbar(contour)
        if colorbar_label:
            cbar.set_label(colorbar_label)  # Set the colorbar label

    # Set labels and title
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Adding legend if labels are provided
    if legend_labels and contour_type == 'lines':
        legend_patches = [
            Patch(color=color, label=label)
            for color, label in zip(line_colors, legend_labels)
        ]
        plt.legend(handles=legend_patches)

    plt.tight_layout()

    # Save the figure if a path is provided
    if save_path:
        fig = plt.gcf()  # Get the current figure
        with save_lock:
            fig.savefig(save_path)


# endregion
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

agent = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI,
    model_type=ModelType.GPT_4O,
    model_config_dict=ChatGPTConfig(temperature=1.0).as_dict(),
)

assistant_sys_msg = BaseMessage.make_assistant_message(
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
--Do NOT generate linear data.
--DO NOT return function code, just the function call with parameters.
""",
)

# 25 Theme for selection
theme_selection = [
    "Economics", "Psychology", "Sociology", "Biology", "Education",
    "Engineering", "Law", "Astronomy", "Computer_Science", "Geography",
    "Physics", "Chemistry", "History", "Environmental_Science", "Anthropology",
    "Media_and_Journalism", "Mathematics", "Statistics", "Finance", "Medicine",
    "Art_and_Design", "Agriculture", "Linguistics", "Architecture", "Sports"
]


def execute_code_and_save(code):
    exec(code, globals())


def process_plot(plot_type, iter, theme_selection, max_tries=2):

    # synthesis_assistant = synthesis_assistant.new_chat()
    synthesis_assistant = ChatAgent(assistant_sys_msg,
                                    model=agent,
                                    token_limit=8192)

    layout_selection = [(x, y) for x in range(1, 4) for y in range(1, 4)]
    layout_selection.remove((1, 1))
    random_theme = random.choice(theme_selection)
    trend_list = ["upward", "downward", "fluctuating"]
    random_trend = random.sample(trend_list,
                                 k=random.randint(1, len(trend_list)))
    trend_string = ', '.join([f"{item} data" for item in random_trend])

    png_dir = f"./ecd_single_plot_charts/{plot_type}/png/"
    txt_dir = f"./ecd_single_plot_charts/{plot_type}/txt/"
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)

    png_save_path = f"{png_dir}{iter:06}_{random_theme}.png"
    txt_save_path = f"{txt_dir}{iter:06}_{random_theme}.txt"

    plot_function = inspect.getsource(globals()[plot_type])

    if plot_type == 'plot_density_chart' or plot_type == 'plot_quiver_chart' or plot_type == 'plot_scatter_chart' or plot_type == 'plot_contour_chart' or plot_type == 'plot_violin_chart' or plot_type == 'plot_3d_chart':
        example_usage = random.choice(example_usages[plot_type])
    else:
        example_usage = example_usages[plot_type][0]

    print('example_usage:', example_usage)

    user_prompt = f"""
        {plot_function}
        **Example Usage**:
        {example_usage}
        **Chart Theme**: Imagine a real application scenario under the **{random_theme}** theme and generate the data accordingly. Be specific. For example, if the theme is Finance, you could imagine a company struggling financially but expanding into new business areas. If the theme is Astronomy, you could picture a star nearing the end of its life.
        **The number of elements**:
        First level: {random.randint(3, 6)};
        If second level make sense in the context, Second level: {random.randint(3, 6)}.
        **Trend of data**: create {trend_string}.
        **grid**: {random.choice([False, True])}.
        **save_path**: {png_save_path}.
        """

    for tries_i in range(max_tries):
        user_msg = BaseMessage.make_user_message(
            role_name="User",
            content=user_prompt,
        )

        # Get the response containing the generated docstring
        response = synthesis_assistant.step(user_msg)
        # Extract the generated python code from the response
        generated_code = response.msg.content.split("```python\n")[1].split(
            "\n```")[0]
        print(f"Iteration: {iter}, Tries:{tries_i}, Path: {png_save_path}")

        try:
            execute_code_and_save(generated_code)
            with open(txt_save_path, 'w') as txt_file:
                txt_file.write(generated_code)
            plt.close()
            break
        except Exception as e:
            user_prompt += f"""Previous non-working example: {generated_code}, Corresponding error: {e}"""
            print('Error:', e)
            continue


def check_missing_files(iterations, plot_type):
    png_path = f"./ecd_single_plot_charts/{plot_type}/png/"
    os.makedirs(png_path, exist_ok=True)
    missing_files = []

    pattern = re.compile(
        r"^(\d{6})_.*\.png$"
    ) 

    existing_files = set()

    for filename in os.listdir(png_path):
        match = pattern.match(filename)
        if match:
            file_number = match.group(1) 
            existing_files.add(file_number)

    # Check for missing files
    for i in range(iterations):  # From 000000 to 000374
        file_number = str(i).zfill(
            6)  # Ensure the number has 6 digits (e.g., 000001)
        if file_number not in existing_files:
            missing_files.append(file_number)

    return missing_files


def run_concurrent_tasks(plot_types,
                         iterations,
                         theme_selection,
                         max_workers=5):

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for plot_type in plot_types:
            for iter in range(0, iterations):
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


if __name__ == "__main__":
    # Configure plot types and iterations
    plot_types = [
        'plot_line_chart', 'plot_violin_chart', 'plot_rose_chart',
        'plot_treemap_chart', 'plot_node_chart', 'plot_funnel_chart',
        'plot_area_chart', 'plot_pie_chart', 'plot_density_chart',
        'plot_bubble_chart', 'plot_histogram_chart', 'plot_bar_chart',
        'plot_heatmap_chart', 'plot_error_point_chart', 'plot_box_chart',
        'plot_radar_chart', 'plot_error_bar_chart', 'plot_scatter_chart',
        'plot_quiver_chart', 'plot_contour_chart', 'plot_candlestick_chart',
        'plot_3d_chart'
    ]

    iterations = 375  # Total iterations for each plot type
    max_concurrent_workers = 1  # Configurable number of concurrent workers

    # Start concurrent tasks
    run_concurrent_tasks(plot_types,
                         iterations,
                         theme_selection,
                         max_workers=max_concurrent_workers)


