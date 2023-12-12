import os

import pandas as pd
import plotly.express as px
import streamlit as st
from plotly import graph_objs as go

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
from ml.src.pipeline.constants import MASSBANK_DIR_PATH, FILE_FORMAT

VALIDATION_COVERAGE_PLOTS_DIR_PATH = os.path.join(MASSBANK_DIR_PATH, 'validation_coverage_plots')


st.header("Massbank validation set coverage")

with st.spinner(f"Loading.."):
    subset_ids_list_names = ["validation_compounds_safe_and_unsafe", "validation_compounds_safe", "validation_compounds_unsafe"]

    dfs_wrapped = []
    for subset_ids_list_name in subset_ids_list_names:
        path = os.path.join(VALIDATION_COVERAGE_PLOTS_DIR_PATH, subset_ids_list_name, f"coverage_info{FILE_FORMAT}")
        df_coverage_info = pd.read_parquet(path)
        path = os.path.join(VALIDATION_COVERAGE_PLOTS_DIR_PATH, subset_ids_list_name,
                            f"presence_matrix{FILE_FORMAT}")
        df_presence_matrix = pd.read_parquet(path).T

        dfs_wrapped.append((subset_ids_list_name, df_coverage_info, df_presence_matrix))

    tabs = st.tabs([df_wrapped[0] for df_wrapped in dfs_wrapped])

    colorscale = [[0, 'white'], [0.5, 'lightgrey'], [1, 'purple']]
    value_labels = {0: "Absent", 1: "Present in assay endpoint", 2: ".. and present in validation set"}
    tickvals = list(value_labels.keys())
    ticktext = list(value_labels.values())

    for i, (df_name, df_coverage_info, df_presence_matrix) in enumerate(dfs_wrapped):
        print(i, df_name)
        with tabs[i]:
            fig = go.Figure(data=go.Heatmap(z=df_presence_matrix, colorscale=colorscale,
                                            colorbar=dict(tickvals=tickvals,
                                                            ticktext=ticktext),
                                            # text=df_presence_matrix,
                                            hoverinfo='text', showscale=True))

            fig.update_layout(
                title=f"Presence matrix",
                xaxis_title="Assay Endpoint Index",
                yaxis_title="Compound Index"
            )

            st.plotly_chart(fig)

            df_coverage_info = df_coverage_info.sort_values(by=['relative_coverage'], ascending=False)
            fig = px.scatter(df_coverage_info, x=df_coverage_info.index, y=df_coverage_info['relative_coverage'],
                                labels={'x': 'Assay endpoints', 'y': 'Relative coverage'})
            fig.update_layout(
                title_text=f'Relative coverage')
            fig.update_xaxes(type='category')
            fig.update_traces(marker=dict(size=3))
            fig.update_traces(marker_symbol='circle-open')
            fig.update_xaxes(title_text=f'Assay endpoints (truncated, total: {df_coverage_info.shape[0]})')
            fig.update_yaxes(title_text=f'Relative coverage')
            fig.update_yaxes(range=[0, 1])
            fig.update_layout(yaxis_tickformat='.0%')
            # fig.update_xaxes(showticklabels=False)
            st.plotly_chart(fig)
            st.caption('100% = all compounds from validation set are present in assay endpoint and correpsonding fingerprint from structure exists')
            # selected_points = plotly_events(fig)
            # st.write(selected_points)


            df_coverage_info = df_coverage_info.sort_values(by=['overlap'], ascending=False)
            fig = px.scatter(df_coverage_info, x=df_coverage_info.index, y=df_coverage_info['overlap'],
                                labels={'x': 'Assay endpoints', 'y': 'Overlap'})

            # fig.update_layout(title_text=f'overlap')
            fig.update_xaxes(type='category')
            fig.update_traces(marker=dict(size=4), marker_symbol='circle-open')
            fig.update_xaxes(showticklabels=False,
                             title_text=f'Assay endpoints',
                             title_font_size=20)
            fig.update_yaxes(title_text=f"# Compounds in Massbank validation set",
                            title_font_size=20)
            
            # set y range from 0 to max value

            fig.update_yaxes(range=[0, df_coverage_info['overlap'].max()+50])
            st.plotly_chart(fig)
