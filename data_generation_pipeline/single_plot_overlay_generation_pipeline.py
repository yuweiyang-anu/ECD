import os
import re
import random
import inspect
import matplotlib
import threading
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.configs import ChatGPTConfig
from camel.types import ModelPlatformType, ModelType
import matplotlib.gridspec as gridspec
from matplotlib.patches import ConnectionPatch
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


## 1. plot scatter and hist combination
def plot_combination_scatter_and_histogram_plots_overlay(
        scatter_data_list,
        scatter_colors_list=None,
        scatter_sizes=None,
        scatter_labels=None,
        scatter_alpha=1.0,
        scatter_edge_color='k',
        scatter_edge_width=1.0,
        histogram_type=None,
        histogram_bins=10,
        title='Combined Scatter and Histogram Charts',
        x_label='X-axis',
        y_label='Y-axis',
        grid=False,
        figsize=(8, 8),
        save_path=None):
    """
    Plots a combination of scatter plots and marginal histograms in a single figure, supporting multiple scatter datasets. The scatters should follow various patterns, not always a fixed pattern.

    Parameters:
    - scatter_data_list: List of tuples/arrays, each containing (x, y) data for a scatter plot. **It can be ONLY 1 ELEMENT sometimes, not always > 1.** If number of elements > 1, it must not exceed 3 to ensure clear.
    - scatter_colors_list: List of colors for each scatter plot. If None, a default color will be used.
    - scatter_sizes: Size for the scatter plot markers, should be a constant value.
    - scatter_labels: List of labels for each scatter plot.
    - scatter_alpha: Alpha value for the scatter plots (transparency).
    - scatter_edge_color: Edge color for the scatter plot markers.
    - scatter_edge_width: Width of the edge lines for scatter plot markers.
    - histogram_type: Dictionary specifying color and fill/edge properties for histograms. It can be the same 'edge' or 'fill' type for both x_hist and y_hist.
    - histogram_bins: Number of bins for the histograms (optional, will auto-compute based on data).
    - title: Title of the plot.
    - x_label: Label for the X-axis of the scatter plot.
    - y_label: Label for the Y-axis of the scatter plot.
    - grid: Boolean to show/hide grid lines on the scatter plot.
    - figsize: Tuple indicating the figure size, should be various.
    - save_path: Path to save the figure.
    """
    # Create the figure and gridspec
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2,
                          2,
                          width_ratios=[4, 1],
                          height_ratios=[1, 4],
                          wspace=0.0,
                          hspace=0.0)

    ax = fig.add_subplot(gs[1, 0])  # Main scatter plot
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)  # X marginal histogram
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)  # Y marginal histogram

    # Set default values for scatter parameters
    if scatter_colors_list is None:
        scatter_colors_list = ['#3b76af'] * len(
            scatter_data_list)  # Default color if not provided
    if scatter_labels is None:
        scatter_labels = [None] * len(scatter_data_list)

    # Check if histogram_color is a dictionary
    if isinstance(histogram_type, dict):
        x_hist_mode = histogram_type.get('x_hist_mode',
                                         'fill')  # 'fill' or 'edge'
        y_hist_mode = histogram_type.get('y_hist_mode',
                                         'fill')  # 'fill' or 'edge'
    else:
        x_hist_mode = y_hist_mode = 'fill'

    # Loop over each scatter data and plot them
    for i, scatter_data in enumerate(scatter_data_list):
        x, y = scatter_data

        color = scatter_colors_list[i]  # Color for the i-th scatter plot
        label = scatter_labels[i]  # Label for the i-th scatter plot

        # Scatter plot
        scatter_plot = ax.scatter(x,
                                  y,
                                  color=color,
                                  s=scatter_sizes,
                                  alpha=scatter_alpha,
                                  edgecolor=scatter_edge_color,
                                  linewidth=scatter_edge_width,
                                  label=label)

        # Marginal histograms
        bins_x = histogram_bins if histogram_bins else 'auto'
        bins_y = histogram_bins if histogram_bins else 'auto'

        # X-axis histogram
        if x_hist_mode == 'fill':
            ax_histx.hist(x,
                          bins=bins_x,
                          color=color,
                          edgecolor=color,
                          alpha=0.6)
        elif x_hist_mode == 'edge':
            ax_histx.hist(x,
                          bins=bins_x,
                          color='white',
                          edgecolor=color,
                          linewidth=2,
                          alpha=0.6)

        # Y-axis histogram
        if y_hist_mode == 'fill':
            ax_histy.hist(y,
                          bins=bins_y,
                          orientation='horizontal',
                          color=color,
                          edgecolor=color,
                          alpha=0.6)
        elif y_hist_mode == 'edge':
            ax_histy.hist(y,
                          bins=bins_y,
                          orientation='horizontal',
                          color='white',
                          edgecolor=color,
                          linewidth=2,
                          alpha=0.6)

    # Finalize histograms and scatter plot
    ax_histx.tick_params(axis='x', labelbottom=False)
    ax_histx.set_yticks([])
    ax_histx.spines['top'].set_visible(False)
    ax_histx.spines['left'].set_visible(False)
    ax_histx.spines['right'].set_visible(False)

    ax_histy.tick_params(axis='y', labelleft=False)
    ax_histy.set_xticks([])
    ax_histy.spines['top'].set_visible(False)
    ax_histy.spines['bottom'].set_visible(False)
    ax_histy.spines['right'].set_visible(False)

    # Formatting main scatter plot
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if grid:
        ax.grid(True)

    # Show scatter plot legend
    if any(label is not None for label in scatter_labels):
        ax.legend()

    # Set title
    fig.suptitle(title)

    # Tight layout and save plot if path is provided
    plt.tight_layout()

    if save_path:
        fig = plt.gcf()
        with save_lock:
            fig.savefig(save_path)


## 2. scatter + density Single plot generation
def plot_combination_scatter_and_density_plots_overlay(
        scatter_data_list,
        scatter_colors_list=None,
        scatter_sizes=None,
        scatter_labels=None,
        scatter_alpha=1.0,
        scatter_edge_color='k',
        scatter_edge_width=1.0,
        density_bandwidth=0.2,
        title='Combined Scatter and Density Plots',
        x_label='X-axis',
        y_label='Y-axis',
        grid=False,
        figsize=(8, 8),
        save_path=None):
    """
    Plots a combination of scatter plots and marginal density plots in a single figure, supporting multiple scatter datasets. The scatters should follow various patterns, not always a fixed pattern.

    Parameters:
    - scatter_data_list: List of tuples/arrays, each containing (x, y) data for a scatter plot. **It should be ONLY 1 ELEMENT sometimes, not always > 1.** If number of elements > 1, it **must not exceed 3** to ensure clear.
    - scatter_colors_list: List of colors for each scatter plot. If None, a default color will be used.
    - scatter_sizes: Size for the scatter plot markers.
    - scatter_labels: List of labels for each scatter plot.
    - scatter_alpha: Alpha value for the scatter plots (transparency).
    - scatter_edge_color: Edge color for the scatter plot markers.
    - scatter_edge_width: Width of the edge lines for scatter plot markers.
    - density_bandwidth: Bandwidth parameter for KDE (controls smoothness).
    - title: Title of the plot.
    - x_label: Label for the X-axis of the scatter plot.
    - y_label: Label for the Y-axis of the scatter plot.
    - grid: Boolean to show/hide grid lines on the scatter plot.
    - figsize: Tuple indicating the figure size, should be various.
    - save_path: Path to save the figure.
    """
    # Create the figure and gridspec
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2,
                          2,
                          width_ratios=[4, 1],
                          height_ratios=[1, 4],
                          wspace=0.0,
                          hspace=0.0)

    ax = fig.add_subplot(gs[1, 0])  # Main scatter plot
    ax_densx = fig.add_subplot(gs[0, 0], sharex=ax)  # X marginal density plot
    ax_densy = fig.add_subplot(gs[1, 1], sharey=ax)  # Y marginal density plot

    # Set default values for scatter parameters
    if scatter_colors_list is None:
        scatter_colors_list = ['#3b76af'] * len(
            scatter_data_list)  # Default color if not provided
    if scatter_labels is None:
        scatter_labels = [None] * len(scatter_data_list)

    # Loop over each scatter data and plot them
    for i, scatter_data in enumerate(scatter_data_list):
        x, y = scatter_data

        color = scatter_colors_list[i]  # Color for the i-th scatter plot
        label = scatter_labels[i]  # Label for the i-th scatter plot

        # Scatter plot
        scatter_plot = ax.scatter(x,
                                  y,
                                  color=color,
                                  s=scatter_sizes,
                                  alpha=scatter_alpha,
                                  edgecolor=scatter_edge_color,
                                  linewidth=scatter_edge_width,
                                  label=label)

        # Marginal density plots (KDE) using gaussian_kde from scipy

        # X-axis density plot
        kde_x = gaussian_kde(x, bw_method=density_bandwidth)
        x_vals = np.linspace(
            np.min(np.concatenate([x for x, _ in scatter_data_list])),
            np.max(np.concatenate([x for x, _ in scatter_data_list])), 1000)
        ax_densx.fill_between(x_vals, 0, kde_x(x_vals), color=color, alpha=0.6)

        # Y-axis density plot (horizontal on the right side)
        kde_y = gaussian_kde(y, bw_method=density_bandwidth)
        y_vals = np.linspace(
            np.min(np.concatenate([y for _, y in scatter_data_list])),
            np.max(np.concatenate([y for _, y in scatter_data_list])), 1000)
        ax_densy.fill_betweenx(y_vals,
                               0,
                               kde_y(y_vals),
                               color=color,
                               alpha=0.6)  # Change here: fill_betweenx

    # Finalize density plots and scatter plot
    ax_densx.tick_params(axis='x', labelbottom=False)
    ax_densx.set_yticks([])
    ax_densx.spines['top'].set_visible(False)
    ax_densx.spines['left'].set_visible(False)
    ax_densx.spines['right'].set_visible(False)

    ax_densy.tick_params(axis='y', labelleft=False)
    ax_densy.set_xticks([])
    ax_densy.spines['top'].set_visible(False)
    ax_densy.spines['bottom'].set_visible(False)
    ax_densy.spines['right'].set_visible(False)

    # Formatting main scatter plot
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if grid:
        ax.grid(True)

    # Show scatter plot legend
    if any(label is not None for label in scatter_labels):
        ax.legend()

    # Set title
    fig.suptitle(title)

    # Tight layout and save plot if path is provided
    plt.tight_layout()

    if save_path:
        fig = plt.gcf()
        with save_lock:
            fig.savefig(save_path)


## 3. hex bin plot + histogram Single plot generation
def plot_combination_hex_and_histogram_plots_overlay(
        hex_data,
        hex_params=None,
        histogram_params=None,
        title='Hexbin Plot with Marginal Histograms',
        x_label='X-axis',
        y_label='Y-axis',
        grid=False,
        figsize=(10, 8),
        save_path=None):
    """
    Plots a combination of a hexbin plot and marginal histograms in a single figure. 

    Parameters:
    - hex_data: Tuple or arrays, where the first element is x and the second element is y for the hexbin plot. **The x and y must be the same size.** The hex_data should follow various patterns, not always a fixed pattern like the normal distribution.
    - hex_params: Dictionary containing parameters for the hexbin plot like 'gridsize', 'cmap', 'alpha'.
    - histogram_params: Dictionary containing parameters for the histograms like 'color', 'edgecolor', 'linewidth', 'alpha', etc. **Please note that the 'color' or 'edgecolor' must be similar to the hex cmap. In addition, the color can be 'white' sometimes.**
    - title: Title of the plot.
    - x_label: Label for the X-axis of the hexbin plot.
    - y_label: Label for the Y-axis of the hexbin plot.
    - grid: Boolean to show/hide grid lines on the plot.
    - figsize: Tuple controlling the size of the entire figure (width, height), should be various.
    - save_path: Path to save the figure, e.g., 'path/to/figure.png'.
    """

    # Set default hexbin parameters if none are provided
    if hex_params is None:
        hex_params = {
            'gridsize': 30,
            'cmap':
            'Blues',  # Smooth colormap transitioning from light blue to dark blue
            'alpha': 0.8
        }

    # Set default histogram parameters if none are provided
    if histogram_params is None:
        histogram_params = {
            'color': 'white',
            'edgecolor': 'black',
            'linewidth': 1.5,
            'alpha': 0.6,
            'bins': 30
        }

    # Create the figure and gridspec layout for subplots
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, width_ratios=[5, 1], height_ratios=[1, 5])

    # Main hexbin plot
    ax = plt.subplot(gs[1, 0])
    x, y = hex_data
    hb = ax.hexbin(x,
                   y,
                   gridsize=hex_params['gridsize'],
                   cmap=hex_params['cmap'],
                   alpha=hex_params['alpha'],
                   mincnt=1)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(x.min() - 0.05, x.max() + 0.05)  # Adjust the x-axis limit
    ax.set_ylim(y.min() - 0.05, y.max() + 0.05)  # Adjust the y-axis limit
    ax.axhline(0.5, color="gray", linestyle="--",
               linewidth=1)  # Optional horizontal line at y=0.5
    ax.axvline(0.5, color="gray", linestyle="--",
               linewidth=1)  # Optional vertical line at x=0.5
    if grid:
        ax.grid(True, linestyle='--', alpha=0.5)

    # Get the counts from the hexbin plot to generate histograms
    counts = hb.get_array()

    # Top histogram: X-axis distribution (e.g., Age)
    ax_histx = plt.subplot(gs[0, 0], sharex=ax)
    ax_histx.hist(
        x, **histogram_params,
        fill=True)  # Set fill to true for white fill and black border
    ax_histx.spines['top'].set_visible(False)
    ax_histx.spines['right'].set_visible(False)
    ax_histx.spines['left'].set_visible(False)
    ax_histx.spines['bottom'].set_visible(False)
    ax_histx.tick_params(labelleft=False,
                         labelbottom=False,
                         left=False,
                         bottom=False)  # Remove ticks and labels

    # Right histogram: Y-axis distribution (e.g., Income)
    ax_histy = plt.subplot(gs[1, 1], sharey=ax)
    ax_histy.hist(
        y, **histogram_params, orientation="horizontal",
        fill=True)  # Set fill to true for white fill and black border
    ax_histy.spines['top'].set_visible(False)
    ax_histy.spines['right'].set_visible(False)
    ax_histy.spines['left'].set_visible(False)
    ax_histy.spines['bottom'].set_visible(False)
    ax_histy.tick_params(labelleft=False,
                         labelbottom=False,
                         left=False,
                         bottom=False)  # Remove ticks and labels

    # Tighten the layout further to make the histograms fit tightly to the axes
    plt.subplots_adjust(hspace=0.1,
                        wspace=0.05,
                        top=0.92,
                        bottom=0.08,
                        left=0.08,
                        right=0.92)

    # Set the overall title for the plot
    fig.suptitle(title, fontsize=16)

    # Tight layout and save plot if path is provided
    plt.tight_layout()

    if save_path:
        fig = plt.gcf()
        with save_lock:
            fig.savefig(save_path)


## 4. histogram + density plot Single Plot generation
def plot_combination_histogram_and_density_plots_overlay(
        data_list,
        histogram_colors=None,
        density_colors=None,
        density_styles=None,
        histogram_bins=None,
        density_labels=None,
        histogram_labels=None,
        density_alpha=0.6,
        histogram_alpha=0.6,
        density_linewidth=2,
        histogram_edge_color='black',
        histogram_edge_width=1.0,
        title='Combined Histogram and Density Plot',
        x_label='X-axis',
        y_label='Density',
        histogram_x_label='Histogram X-axis',
        histogram_y_label='Histogram Y-axis',
        grid=False,
        figsize=(10, 6),
        save_path=None):
    """
    Plots a combination of multiple histograms and density plots in a single figure. **The hists should follow various patterns, not always a fixed pattern like the normal distribution**.
    
    Parameters:
    - data_list (list of ndarray): A list of datasets. Each dataset will be plotted as both a histogram and a density plot. **Please make sure the length of data_list should not exceed 4**. 
    - histogram_colors (list of str, optional): A list of colors for the histograms.
    - density_colors (list of str, optional): A list of colors for the density curves. It should be similar to the corresponding histogram color most of the time.
    - density_styles (list of str, optional): A list of line styles for the density curves (e.g., '-', '--', ':'). 
    - histogram_bins (int, optional): The number of bins to use for the histograms. 
    - density_labels (list of str, optional): A list of labels for the density curves. The density_labels must be **specific terms** (must not be Common label + numbers / letters, like ['Sample A', 'Sample B'] or ['Sample 1', 'Sample 2']).
    - histogram_labels (list of str, optional): A list of labels for the histograms.
    - density_alpha (float, optional): The transparency level of the density curves. 
    - histogram_alpha (float, optional): The transparency level of the histograms.
    - density_linewidth (float, optional): The width of the lines for the density curves. 
    - histogram_edge_color (str, optional): The color of the edges of the histogram bars.
    - histogram_edge_width (float, optional): The width of the edges of the histogram bars.
    - title (str, optional): The title of the plot.
    - x_label (str, optional): The label for the x-axis. 
    - y_label (str, optional): The label for the y-axis.
    - histogram_x_label (str, optional): The label for the x-axis in the histogram.
    - histogram_y_label (str, optional): The label for the y-axis in the histogram.
    - grid (bool, optional): Whether to display a grid.
    - figsize (tuple of int, optional): The size of the figure (width, height), should be various.
    - save_path (str, optional): If provided, the plot will be saved to this path as an image. 
    """

    fig, ax = plt.subplots(figsize=figsize)

    # Plot histograms
    bins = histogram_bins or 30
    for i, data in enumerate(data_list):
        color = histogram_colors[i] if histogram_colors and i < len(
            histogram_colors) else 'lightblue'
        label = histogram_labels[i] if histogram_labels and i < len(
            histogram_labels) else None
        _, _, patches = ax.hist(data,
                                bins=bins,
                                color=color,
                                edgecolor=histogram_edge_color,
                                alpha=histogram_alpha,
                                density=True,
                                label=label)
        for patch in patches:
            patch.set_edgecolor(histogram_edge_color)
            patch.set_linewidth(histogram_edge_width)

    # Plot density plots using matplotlib (Gaussian KDE)
    for i, data in enumerate(data_list):
        color = density_colors[i] if density_colors and i < len(
            density_colors) else 'blue'
        linestyle = density_styles[i] if density_styles and i < len(
            density_styles) else '-'
        label = density_labels[i] if density_labels and i < len(
            density_labels) else None

        # Create KDE using scipy's gaussian_kde
        kde = gaussian_kde(
            data, bw_method=0.5)  # Adjust bandwidth to match histogram density
        x_vals = np.linspace(min(data), max(data), 1000)
        y_vals = kde(x_vals)

        # Plot the density curve
        ax.plot(x_vals,
                y_vals,
                color=color,
                linestyle=linestyle,
                linewidth=density_linewidth,
                alpha=density_alpha,
                label=label)

    # Formatting
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    if grid:
        ax.grid(True)

    # Custom x and y labels for histogram
    ax.set_xlabel(histogram_x_label)
    ax.set_ylabel(histogram_y_label)

    if histogram_labels or density_labels:
        ax.legend()

    # Tight layout and save plot if path is provided
    plt.tight_layout()

    if save_path:
        fig = plt.gcf()
        with save_lock:
            fig.savefig(save_path)


## 5. bar + line plot Single Plot generation
def plot_combination_bar_and_line_plots_overlay(bar_data,
                                                line_data,
                                                bar_colors=None,
                                                line_colors=None,
                                                bar_linewidths=None,
                                                line_linestyles=None,
                                                line_widths=None,
                                                bar_width=0.3,
                                                line_marker='o',
                                                annotate_line_values=False,
                                                bar_legend_labels=None,
                                                line_legend_labels=None,
                                                bar_xlabel='X-axis',
                                                line_xlabel=None,
                                                bar_ylabel='Y-axis',
                                                line_ylabel='Y-axis',
                                                grid=True,
                                                title=None,
                                                x_tick_labels=None,
                                                x_tick_rotation=0,
                                                bar_legend_position=None,
                                                line_legend_position=None,
                                                legend_title=None,
                                                figsize=(10, 5),
                                                save_path=None):
    """
    Plots a combination of bar and line charts on the same figure. The data should include both integers and decimals.

    Parameters:
    - bar_data: A list of lists or 2D array for the bar chart, **must not be random values**. The number of bar in a group should varies, but not exceed 4.
    - line_data: A list of tuples, where each tuple contains two lists (x_data, y_data) for the line chart, **must not be random values**. The number of lines may not have to be the same as the number of bars, but the length of bar_data and line_data must be the same.
    - bar_colors: List of colors for each bar series.
    - line_colors: List of colors for each line series.
    - bar_linewidths: List of line widths for the bar edges.
    - line_linestyles: List of line styles for each line.
    - line_widths: List of line widths for each line.
    - bar_width: Width of the bars, must not exceed 0.2.
    - line_marker: Marker style for the lines.
    - annotate_line_values: Boolean flag to annotate line chart values at each point. Sometimes it should be 'True'.
    - bar_legend_labels: Legend labels for the bar chart.
    - line_legend_labels: Legend labels for the line chart.
    - bar_xlabel: Label for the x-axis for the bar chart.
    - line_xlabel: Label for the x-axis for the line chart.
    - bar_ylabel: Label for the y-axis for the bar chart.
    - line_ylabel: Label for the y-axis for the line chart.
    - grid: Boolean to enable grid on the plot.
    - title: Title of the chart.
    - x_tick_labels: Custom labels for x-ticks (e.g., months).
    - x_tick_rotation: Rotation angle for the chart x-tick labels, should be dynamic, varies in [0, 30, 45, 60, 90], etc, sometimes it can be 0 or 90.
    - bar_legend_position: The position of the bar chart legend (e.g., "lower center"). Sometimes it should be 'lower center'.
    - line_legend_position: The position of the line chart legend (e.g., "upper right").
    - legend_title: Title for the bar chart legend. Sometimes it can be 'None'.
    - figsize: Figure size, should be various.
    - save_path: Path to save the plot.
    """

    fig, ax1 = plt.subplots(figsize=figsize)

    # Plot bar chart (left y-axis)
    bar_data = np.array(bar_data)
    num_bars = bar_data.shape[1]

    bar_x_indices = np.arange(num_bars)

    # Plot bars with dynamic width
    for i in range(bar_data.shape[0]):
        ax1.bar(bar_x_indices + i * bar_width,
                bar_data[i],
                width=bar_width,
                color=bar_colors[i] if bar_colors else None,
                linewidth=bar_linewidths[i] if bar_linewidths else None,
                label=bar_legend_labels[i] if bar_legend_labels else None)

    ax1.set_xlabel(bar_xlabel)
    ax1.set_ylabel(bar_ylabel)

    # Use provided custom x-tick labels
    if x_tick_labels:
        ax1.set_xticks(bar_x_indices + bar_width * (bar_data.shape[0] - 1) / 2)
        ax1.set_xticklabels(x_tick_labels, rotation=x_tick_rotation)
    else:
        ax1.set_xticks(bar_x_indices)
        ax1.set_xticklabels([f'Group {i+1}' for i in range(num_bars)],
                            rotation=x_tick_rotation)

    # Create the second y-axis for the line chart
    ax2 = ax1.twinx()

    # Adjust line data's x-position to align with the center of the bar groups
    line_x_offsets = bar_x_indices + bar_width * (bar_data.shape[0] - 1) / 2

    # Plot the line chart and adjust x_data to be centered with respect to the bars
    for i, (x_data, y_data) in enumerate(line_data):
        # Adjust the x_data to match the center of the bar groups
        adjusted_x_data = [line_x_offsets[int(x)] for x in x_data]

        ax2.plot(adjusted_x_data,
                 y_data,
                 color=line_colors[i] if line_colors else None,
                 linestyle=line_linestyles[i] if line_linestyles else '-',
                 linewidth=line_widths[i] if line_widths else 2,
                 marker=line_marker,
                 label=line_legend_labels[i] if line_legend_labels else None)

        # Annotate line values if the flag is set to True
        if annotate_line_values:
            for x, y in zip(adjusted_x_data, y_data):
                ax2.text(x,
                         y,
                         f'{y}',
                         color=line_colors[i] if line_colors else 'black',
                         ha='center',
                         va='bottom')

    ax2.set_xlabel(line_xlabel if line_xlabel else bar_xlabel)
    ax2.set_ylabel(line_ylabel)
    ax2.tick_params(axis='x', rotation=x_tick_rotation)

    # Set title, grid, and legend
    if title:
        plt.title(title)

    if grid:
        ax1.grid(True)

    # Set the bar legend position if specified
    if bar_legend_labels:
        if bar_legend_position == 'lower center':
            ax1.legend(loc=bar_legend_position,
                       ncol=4,
                       bbox_to_anchor=(0.5, -0.3),
                       title=legend_title)
        else:
            ax1.legend(loc=bar_legend_position, title=legend_title)

    if line_legend_labels:
        if line_legend_position == 'lower center':
            ax2.legend(loc=line_legend_position,
                       ncol=4,
                       bbox_to_anchor=(0.5, -0.3))
        else:
            ax2.legend(loc=line_legend_position)

    # Tight layout and save plot if path is provided
    plt.tight_layout()

    if save_path:
        fig = plt.gcf()
        with save_lock:
            fig.savefig(save_path)


## 6. violin + box plot Single Plot generation
def plot_combination_box_and_violin_plots_overlay(data,
                                                  title=None,
                                                  grid=True,
                                                  xticks=None,
                                                  xtick_rotation=0,
                                                  xlabel="Algorithms",
                                                  ylabel="Error Rate",
                                                  colors=None,
                                                  kde_scale_factor=0.1,
                                                  box_width=0.15,
                                                  scatter_alpha=0.3,
                                                  box_color="black",
                                                  median_color="black",
                                                  show_median=True,
                                                  figsize=(7, 5),
                                                  save_path=None):
    """
    Generalized function to plot combination of KDE, boxplot, and scatter plots.

    Parameters:
    - data: List of datasets to plot (each dataset should be a 1D numpy array or list). The distribution of the data should follow various patterns, not always a fixed pattern. 
    - title: Title for the plot (default: None).
    - grid: Boolean to control whether grid lines are displayed (default: True).
    - xticks: List of xticks labels (default: ['Dataset 1', 'Dataset 2']).
    - xtick_rotation: Rotation angle for xtick labels (default: 0), should be dynamic, varies in [0, 30, 45, 60, 90], etc. **Please make sure sometimes it can be 0 or 90, not alaways be 30, 45 or 60.**
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    - colors: List of colors for each dataset.
    - kde_scale_factor: Scaling factor for the KDE plots.
    - box_width: Width of the boxplot.
    - scatter_alpha: Alpha transparency for scatter plot points.
    - box_color: Color of the box boundary.
    - median_color: Color of the median line inside the box.
    - show_median: Boolean to control whether the median line is shown in the boxplot. **Please make sure it can be 'False', not always 'True'.**
    - figsize: Tuple for the size of the plot, should be various.
    - save_path: Path to save the figure (default: None, does not save).
    """

    # Default values
    if xticks is None:
        xticks = [f"Dataset {i+1}" for i in range(len(data))]
    if colors is None:
        colors = plt.cm.tab10.colors[:len(data)]  # Using default tab10 colors

    # Set the figure size
    plt.figure(figsize=figsize)

    # Calculate the kernel density estimate (KDE) and plot the density plot for each dataset
    for i, d in enumerate(data):
        # Calculate KDE
        kde = gaussian_kde(d)
        kde_x = np.linspace(min(d), max(d), 300)
        kde_y = kde(kde_x)

        # Scale KDE values to match the position of the boxplot
        kde_y_scaled = kde_y / kde_y.max() * kde_scale_factor
        offset = 0.2

        # Plot filled density plot to the left of the boxplot
        plt.fill_betweenx(kde_x,
                          i - kde_y_scaled - offset,
                          i - offset,
                          alpha=0.5,
                          color=colors[i],
                          edgecolor="black")

    # Create boxplots inside the violin plots
    for i, d in enumerate(data):
        boxprops = dict(facecolor="none", color=box_color)
        medianprops = dict(color=median_color) if show_median else dict(
            color="none")

        plt.boxplot(d,
                    positions=[i],
                    widths=box_width,
                    patch_artist=True,
                    medianprops=medianprops,
                    boxprops=boxprops)

    # Add scatter plot for individual data points with transparency
    for i, d in enumerate(data):
        x = np.random.normal(i, 0.04, size=len(d))
        plt.scatter(x, d, alpha=scatter_alpha, color=colors[i], s=10)

    # Set the x-axis labels and add title
    plt.xticks(range(len(data)), xticks,
               rotation=xtick_rotation)  # Apply xtick rotation
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Set title if provided
    if title:
        plt.title(title)

    # Add grid if 'grid' is True
    if grid:
        plt.grid(True)

    # Tight layout and save plot if path is provided
    plt.tight_layout()

    if save_path:
        fig = plt.gcf()
        with save_lock:
            fig.savefig(save_path)


## 7. pie + bar plot Single Plot generation
def plot_combination_pie_and_bar_plots_overlay(pie_params,
                                               bar_params,
                                               connection_params=None,
                                               highlight_wedge=None,
                                               figsize=(9, 5),
                                               save_path=None):
    """
    Plot a combination of a pie chart and stacked bar chart in a single figure.

    Parameters:
    - pie_params (dict): Dictionary with keys for pie chart customization:
        - 'data': Data for the pie chart (e.g. [0.2, 0.3, 0.5]). Must add up to 1.0.
        - 'labels': Labels for the pie chart sections (e.g. ['Approve', 'Disapprove', 'Undecided']).
        - 'explode': How much each slice should "explode" (e.g. [0.1, 0, 0]).
        - 'colors': Colors for each pie slice (e.g. ['red', 'blue', 'green']).
        - 'startangle': Angle to start the pie chart (default 90).
        - 'offset': Offset for rotating the pie chart (default 0).
        - 'rings': Boolean flag to display as a ring chart (default False). Please make sure it can be 'True'.
        - 'title': Title for the pie chart (default None).
    - bar_params (dict): Dictionary with keys for bar chart customization:
        - 'data': Data for the stacked bar chart (e.g. [0.4, 0.3, 0.2, 0.1]). Must add up to 1.0.
        - 'labels': Labels for the bar segments (e.g. ['Under 35', '35-49', '50-65', 'Over 65']). The labels must be **specific terms** (must not be Common label + numbers / letters).
        - 'colors': Colors for the bars (e.g. ['blue', 'orange', 'purple', 'gray']).
        - 'opacity': Alpha values for the bars (e.g. [0.3, 0.5, 0.7, 1.0]).
        - 'width': Width of the bars (default 0.2).
        - 'title': Title for the bar chart (default None).
    - connection_params (dict): Dictionary to customize the connection lines:
        - 'line_color': Color of the connection lines (default 'black').
        - 'line_width': Width of the connection lines (default 1).
        - 'line_style': Style of the connection lines (default 'solid'). 
    - highlight_wedge (int, optional): Index of the wedge in the pie chart to connect to the bar chart (default None).
    - figsize (tuple): Size of the figure (width, height), should be various.
    - save_path (str, optional): Path to save the plot (default None). If provided, the plot will be saved to this path.
    """
    # Default settings for connection lines
    if connection_params is None:
        connection_params = {
            'line_color': 'black',
            'line_width': 1,
            'line_style': 'solid'
        }

    # Ensure alpha values for the bars are between 0 and 1
    if 'opacity' in bar_params:
        bar_opacity = np.clip(bar_params['opacity'], 0, 1)
    else:
        # Default opacity values if not provided
        bar_opacity = [0.3 + 0.25 * i for i in range(len(bar_params['data']))]

    # Unpack pie parameters
    pie_data = pie_params['data']
    pie_labels = pie_params['labels']
    pie_explode = pie_params['explode']
    pie_colors = pie_params['colors']
    pie_startangle = pie_params.get('startangle', 90)
    pie_offset = pie_params.get('offset', 0)
    pie_rings = pie_params.get('rings', False)
    pie_title = pie_params.get('title',
                               "Pie Chart")  # Default title for pie chart

    # Unpack bar parameters
    bar_data = bar_params['data']
    bar_labels = bar_params['labels']
    bar_colors = bar_params.get('colors', ['C0'] * len(bar_data))
    bar_width = bar_params.get('width', 0.2)
    bar_title = bar_params.get('title',
                               "Bar Chart")  # Default title for bar chart

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.subplots_adjust(wspace=0)

    # Plot Pie chart
    if pie_rings:
        wedges, *_ = ax1.pie(pie_data,
                             autopct='%1.1f%%',
                             startangle=pie_startangle + pie_offset,
                             labels=pie_labels,
                             explode=pie_explode,
                             colors=pie_colors,
                             wedgeprops={'width': 0.3})
    else:
        wedges, *_ = ax1.pie(pie_data,
                             autopct='%1.1f%%',
                             startangle=pie_startangle + pie_offset,
                             labels=pie_labels,
                             explode=pie_explode,
                             colors=pie_colors)

    ax1.set_title(pie_title)

    # Plot Bar chart
    bottom = 1
    for j, (height, label, color, alpha) in enumerate(
            reversed(list(zip(bar_data, bar_labels, bar_colors,
                              bar_opacity)))):
        bottom -= height
        bc = ax2.bar(0,
                     height,
                     bar_width,
                     bottom=bottom,
                     color=color,
                     label=label,
                     alpha=alpha)
        ax2.bar_label(bc, labels=[f"{height:.0%}"], label_type='center')

    ax2.set_title(bar_title)
    ax2.legend(loc='best')
    ax2.axis('off')
    ax2.set_xlim(-2.5 * bar_width, 2.5 * bar_width)

    # If highlight_wedge is provided, we connect the corresponding wedge to the bar chart
    if highlight_wedge is not None and highlight_wedge < len(wedges):
        wedge = wedges[highlight_wedge]
        theta1, theta2 = wedge.theta1, wedge.theta2
        center, r = wedge.center, wedge.r
        bar_height = sum(bar_data)

        # Draw top connecting line
        x = r * np.cos(np.pi / 180 * theta2) + center[0]
        y = r * np.sin(np.pi / 180 * theta2) + center[1]
        con = ConnectionPatch(xyA=(-bar_width / 2, bar_height),
                              coordsA=ax2.transData,
                              xyB=(x, y),
                              coordsB=ax1.transData,
                              linestyle=connection_params['line_style'])
        con.set_color(connection_params['line_color'])
        con.set_linewidth(connection_params['line_width'])
        ax2.add_artist(con)

        # Draw bottom connecting line
        x = r * np.cos(np.pi / 180 * theta1) + center[0]
        y = r * np.sin(np.pi / 180 * theta1) + center[1]
        con = ConnectionPatch(xyA=(-bar_width / 2, 0),
                              coordsA=ax2.transData,
                              xyB=(x, y),
                              coordsB=ax1.transData,
                              linestyle=connection_params['line_style'])
        con.set_color(connection_params['line_color'])
        con.set_linewidth(connection_params['line_width'])
        ax2.add_artist(con)

    # Tight layout and save plot if path is provided
    plt.tight_layout()

    if save_path:
        fig = plt.gcf()
        with save_lock:
            fig.savefig(save_path)


example_usages = {
    "plot_combination_scatter_and_histogram_plots_overlay": [
        f"""
        plot_combination_scatter_and_histogram_plots_overlay(
            scatter_data_list=[(np.random.normal(loc=5, scale=2, size=300), 0.5 * np.random.normal(loc=5, scale=2, size=300) + np.random.normal(0, 1.5, size=300))],
            scatter_colors_list=['mediumvioletred'],
            scatter_sizes=40,  
            scatter_labels=['Social Media and Coffee'], 
            scatter_alpha=0.6,  
            scatter_edge_color='black', 
            scatter_edge_width=0.8,  
            histogram_type={{
                'x_hist_mode': 'edge', 
                'y_hist_mode': 'fill'
            }},
            histogram_bins=15,  
            title='Social Media Hours vs Coffee Cups per Day',  
            x_label='Social Media Hours', 
            y_label='Cups of Coffee', 
            figsize=(10, 10),
            grid=True
        )""", f"""plot_combination_scatter_and_histogram_charts_overlay(
            scatter_data_list=[(np.random.normal(loc=0, scale=1, size=100), np.random.normal(loc=0, scale=1, size=100)),
                            (np.random.uniform(-3, 3, size=100), np.random.uniform(-3, 3, size=100)),
                            (np.random.exponential(scale=1, size=100), np.random.exponential(scale=1, size=100))],
            scatter_colors_list=['#ff5733', '#33ff57', '#3357ff'],
            scatter_sizes=50, 
            scatter_labels=['Group 1: Normally Distributed', 'Group 2: Uniformly Distributed', 'Group 3: Exponentially Distributed'],
            scatter_alpha=0.8, 
            histogram_type=None,  
            histogram_bins=20,  
            title='Comparison of Three Distributions in Scatter Plots with Marginal Histograms',  
            x_label='Data Value',  # More appropriate for the variable being plotted
            y_label='Frequency',  # Represents the frequency of occurrences on the y-axis
            grid=True, 
            figsize=(10, 10), 
            save_path=None  
        )
    """
    ],
    "plot_combination_scatter_and_density_plots_overlay": [
        f"""
    plot_combination_scatter_and_density_plots_overlay(
    scatter_data_list= [
    (
        np.random.normal(loc=5, scale=2, size=300),  # X data (e.g., Hours of Exercise)
        0.5 * np.random.normal(loc=5, scale=2, size=300) + np.random.normal(0, 1.5, size=300)  
    ) ## Only need one element
],
    scatter_colors_list=['steelblue'],
    scatter_sizes=40,  # Size for scatter points
    scatter_labels=['Exercise vs Sleep'], 
    scatter_alpha=0.6,  # Transparency for scatter points
    scatter_edge_color='black', 
    scatter_edge_width=0.8,  
    density_bandwidth=0.2,  # Bandwidth for KDE
    title='Exercise Hours vs Sleep Hours',  # Custom title
    x_label='Hours of Exercise',  # Custom X-axis label
    y_label='Hours of Sleep',  # Custom Y-axis label
    figsize=(10, 10),
    grid=True
)
""", f"""
plot_combination_scatter_and_density_plots_overlay(
    scatter_data_list = [
    (
        np.random.normal(loc=0, scale=1, size=100),
        np.random.beta(a=2, b=5, size=100)
    ),
    (
        np.random.chisquare(df=3, size=100),
        np.random.weibull(a=1.5, size=100)
    ),
    (
        np.random.normal(loc=5, scale=2, size=100),
        np.random.gumbel(loc=0, scale=1, size=100)
    )
],
    scatter_colors_list=['#ff8c00', '#32cd32', '#1e90ff'],
    scatter_sizes=60,  # Size for scatter points
    scatter_labels=['Normal vs Normal', 'Normal vs Poisson', 'Normal vs Uniform'],
    scatter_alpha=0.7,  # Transparency for scatter points
    scatter_edge_color='black', 
    scatter_edge_width=1.0,  
    density_bandwidth=0.3, 
    title='Comparing Various Distributions with Scatter and Density Plots', 
    x_label='Sample Values',  
    y_label='Density Estimate', 
    figsize=(10, 10),
    grid=True
)
"""
    ],
    "plot_combination_hex_and_histogram_plots_overlay": [
        f"""
    plot_combination_hex_and_histogram_plots_overlay(
        hex_data=(np.random.normal(loc=40, scale=10, size=1000), np.random.normal(loc=50000, scale=15000, size=1000)),
        hex_params={{'gridsize': 20, 'cmap': 'Blues', 'alpha': 0.8}},  
        histogram_params={{'color': 'white', 'edgecolor': 'blue', 'linewidth': 1.5, 'alpha': 0.6, 'bins': 30}},
        title='Relationship Between Age and Annual Income',
        x_label='Age (years)',
        y_label='Annual Income ($)',
        grid=True,
        figsize=(12, 10)
    )
    """
    ],
    "plot_combination_histogram_and_density_plots_overlay": [
        f"""plot_combination_histogram_and_density_plots_overlay(
        data_list=[np.random.normal(loc=175, scale=7, size=1000), np.random.normal(loc=162, scale=6, size=1000)],
        histogram_colors=['skyblue', 'lightpink'],
        density_colors=['blue', 'red'],
        density_styles=['-', '--'],
        histogram_bins=30,
        density_labels=['Heights of Adult Men', 'Heights of Adult Women'],
        histogram_labels=['Distribution of Adult Men', 'Distribution of Adult Women'],
        density_alpha=0.7,
        histogram_alpha=0.5,
        density_linewidth=2,
        histogram_edge_color='black',
        histogram_edge_width=1.0,
        figsize=(10, 6),
        title='Height Distribution of Adult Men and Women',
        x_label='Height (cm)',
        y_label='Density',
        histogram_x_label='Height (cm)',
        histogram_y_label='Frequency (Normalized)',
        grid=True
    )"""
    ],
    "plot_combination_bar_and_line_plots_overlay": [
        f"""
    plot_combination_bar_and_line_plots_overlay(
        bar_data=[[500, 600, 700], [300, 400, 500]],
        line_data=[(np.array([0, 1, 2]), np.array([201.5, 75.4, 230.9])), (np.array([0, 1, 2]), np.array([150.3, 180.2, 170.0]))],
        bar_colors=['#3498db', '#e74c3c'], 
        line_colors=['#2ecc71', '#9b59b6'],  
        bar_linewidths=[1, 1], 
        line_linestyles=['-', '--'], 
        line_widths=[2, 3],
        bar_width=0.4, 
        line_marker='^',
        annotate_line_values=True,  
        bar_legend_labels=['Smartphone Sales', 'Laptop Sales'],
        line_legend_labels=['Social Media Ads', 'Email Marketing'],
        bar_xlabel='Month',
        line_xlabel='Month',
        bar_ylabel='Sales (Units)',
        line_ylabel='Website Traffic (Visitors)',
        title='Sales vs Website Traffic (Q1)',
        x_tick_labels=['January', 'February', 'March'], 
        x_tick_rotation=45, 
        grid=True,  
        bar_legend_position='lower center', 
        line_legend_position='upper right', 
        legend_title='Sales Legend',  
        figsize=(12, 6)
    )
"""
    ],
    "plot_combination_box_and_violin_plots_overlay": [
        f"""
        plot_combination_box_and_violin_plots_overlay(
            data=[np.random.normal(0.2, 0.05, 100), np.random.normal(0.25, 0.05, 100)], 
            title="Comparison of Algorithms Error Rates",
            grid=False,
            xticks=["Gradient Descent", "Simulated Annealing"], 
            xtick_rotation=0,
            xlabel="Optimization Methods",
            ylabel="Convergence Rate (%)",
            colors=["mediumseagreen", "darkorange"],
            kde_scale_factor=0.1,
            box_width=0.12,
            scatter_alpha=0.5,
            box_color="black",  
            median_color="red", 
            show_median=True,
            figsize=(8, 6)
        )
        """
    ],
    "plot_combination_pie_and_bar_plots_overlay": [
        f"""
        plot_combination_pie_and_bar_plots_overlay(pie_params={{
            'data': [0.45, 0.25, 0.15, 0.10, 0.05],
            'labels': ['Excellent', 'Good', 'Average', 'Poor', 'Very Poor'],
            'explode': [0.1, 0, 0, 0, 0],
            'colors': ['green', 'blue', 'orange', 'red', 'gray'],
            'startangle': 45,
            'offset': 10,
            'rings': False,
            'title': "Customer Satisfaction"
            }}, bar_params = {{
                'data': [0.40, 0.30, 0.15, 0.10, 0.05],
                'labels': ['18-24', '25-34', '35-44', '45-54', '55+'],
                'opacity': [0.6, 0.8, 0.7, 0.4, 0.5],
                'colors': ['purple', 'cyan', 'magenta', 'yellow', 'teal'],
                'width': 0.3,
                'title': "Age Group Distribution"
            }}, connection_params = {{'line_color': 'black', 'line_width': 1, 'line_style': 'dashed'}}, 
            highlight_wedge=3, 
            figsize=(10, 8)
        )     
        """
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

    if plot_type == 'plot_combination_scatter_and_histogram_plots_overlay' or plot_type == 'plot_combination_scatter_and_density_plots_overlay':
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

    # Iterate through files in the directory and match the pattern
    for filename in os.listdir(png_path):
        match = pattern.match(filename)
        if match:
            file_number = match.group(1) 
            existing_files.add(file_number)

    # Check for missing files
    for i in range(iterations):  
        file_number = str(i).zfill(
            6)
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
                futures.append(
                    executor.submit(process_plot, plot_type, iter,
                                    theme_selection))

            for future in as_completed(futures):
                try:
                    future.result(
                    )  # Can raise exceptions if there was an error in the thread
                except Exception as e:
                    print(f"Error in thread: {e}")


if __name__ == "__main__":
    # Configure plot types and iterations
    plot_types = [
        'plot_combination_scatter_and_histogram_plots_overlay',
        'plot_combination_scatter_and_density_plots_overlay',
        'plot_combination_hex_and_histogram_plots_overlay',
        'plot_combination_histogram_and_density_plots_overlay',
        'plot_combination_bar_and_line_plots_overlay',
        'plot_combination_box_and_violin_plots_overlay',
        'plot_combination_pie_and_bar_plots_overlay'
    ]

    iterations = 375  # Total iterations for each plot type
    max_concurrent_workers = 1  # Configurable number of concurrent workers

    # Start concurrent tasks
    run_concurrent_tasks(plot_types,
                         iterations,
                         theme_selection,
                         max_workers=max_concurrent_workers)
