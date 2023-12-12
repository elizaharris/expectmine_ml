import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.subplots as sp

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
from ml.src.pipeline.constants import FILE_FORMAT, METADATA_SUBSET_DIR_PATH

dest_folder_path = os.path.join(METADATA_SUBSET_DIR_PATH, 'conc')
os.makedirs(dest_folder_path, exist_ok=True)
dest_path = os.path.join(dest_folder_path, f"{0}{FILE_FORMAT}")
df = pd.read_parquet(dest_path)

metrics_to_plot = ['num_points', 'num_groups', 'num_replicates', 'range_min', 'range_max']
metric_rename_dict = {
    'num_points': '# Datapoints (avg)',
    'num_groups': '# Concentration Groups (avg)',
    'num_replicates': '# Replicates (avg)',
    'range_min': 'Lowest Concentration (avg)',
    'range_max': 'Highest Concentration (avg)',
}

def calculate_mean_metrics(df, group_by, metrics):
    aggregated_metrics = df.groupby(group_by)[metrics].mean().reset_index()
    aggregated_metrics = aggregated_metrics.rename(columns=metric_rename_dict)
    return aggregated_metrics

aeid_metrics = calculate_mean_metrics(df, group_by='aeid', metrics=metrics_to_plot)
substance_metrics = calculate_mean_metrics(df, group_by='dsstox_substance_id', metrics=metrics_to_plot)


bar_colors = ['royalblue', 'tomato', 'cornflowerblue', 'coral', 'black', 'black']
curve_colors = ['#1f77b4', '#ff7f0e', '#1f77b4', '#1f77b4', 'black', 'black']

# Create a single legend for all subplots
legend_labels = ['Assay Endpoint', 'Compound']

fig = make_subplots(rows=1, cols=len(metric_rename_dict))

ranges = [[0, 65], [5, 18], [1, 3.5], [0, 1.2], [0, 250]]
bin_sizes = [[5, 5], [1, 1], [0.25, 0.25], [0.1, 0.1], [25, 25]]


colors={'Assay Endpoint': 'rgba(31, 119, 180, 1.0)', 'Compound':'rgba(255, 127, 14, 1.0)', 'Assay Endpoint Background': 'rgba(31, 119, 180, 0.5)', 'Compound Background': 'rgba(255, 127, 14, 0.5)'}


def get_metrics(metric, row, col):
    for i, data in enumerate([aeid_metrics[metric].astype(float), substance_metrics[metric].astype(float)]):
        data = data[data.between(ranges[col-1][0], ranges[col-1][1])]
        distplot = ff.create_distplot(
            [data],
            group_labels=[legend_labels[i]],
            bin_size=bin_sizes[col-1],
            colors=[bar_colors[i]],
            # show_hist=True,
            show_curve=True,
        )

        for j, trace in enumerate(distplot['data']):
            name = trace['name']
            color1 = colors[name]
            color2 = colors[f"{name} Background"]
  
            if j % 3 == 1:
                fig.add_trace(go.Scatter(x=trace.x, y=trace.y, fill='tozeroy', fillcolor=color2, line=dict(color=color1), showlegend=False), row=row, col=col)

        fig.update_xaxes(title_text=f"{metric}", title_font=dict(size=22), tickfont=dict(size=20), showgrid=False, row=row, col=col) # range=ranges[col-1]
        fig.update_yaxes(title_text="Distribution", title_font=dict(size=22), showticklabels=False, showgrid=False, row=row, col=col)

for i, metric in enumerate(metric_rename_dict.values()):
    get_metrics(metric, 1, i + 1)


fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=16, color=colors['Assay Endpoint']), name='Assay Endpoint', legendgroup="Aggregated on"))
fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=16, color=colors['Compound']), name='Compound', legendgroup="Aggregated on"))


fig.update_layout(
    legend=dict(font=dict(size=20)),
    legend_title_font=dict(size=20), 
    legend_title_text="Aggregated By:"
)

fig.update_layout(title_text="Concentration metrics averaged across all concentration-response series aggregated by Assay Endpoint & Compound Index", title_font=dict(size=25), showlegend=True)

st.plotly_chart(fig, use_container_width=True)


