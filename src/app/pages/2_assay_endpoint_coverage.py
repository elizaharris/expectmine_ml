import pandas as pd
import streamlit as st
import os
import plotly.graph_objects as go


import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
from ml.src.pipeline.constants import FILE_FORMAT, INPUT_ML_DIR_PATH, METADATA_SUBSET_DIR_PATH, METADATA_ALL_DIR_PATH

EXPORT = 0

st.header('Assay endpoint coverage by compounds')
with st.spinner(f"Loading.."):
    aeid_chid_presence_matrix_path = os.path.join(METADATA_ALL_DIR_PATH, f"aeid_compound_presence_matrix_with_fingerprint{FILE_FORMAT}")
    df_all = pd.read_parquet(aeid_chid_presence_matrix_path)

    aeid_chid_presence_matrix_path = os.path.join(METADATA_SUBSET_DIR_PATH, f"aeid_compound_presence_matrix_with_fingerprint{FILE_FORMAT}")
    df_subset = pd.read_parquet(aeid_chid_presence_matrix_path)

    # df_all_merged = pd.DataFrame(0, index=df_all.index, columns=df_all.columns)
    # df_subset_merged = pd.DataFrame(0, index=df_subset.index, columns=df_subset.columns)
    # path = os.path.join(INPUT_ML_DIR_PATH, f"{0}{FILE_FORMAT}")
    # df_merged = pd.read_parquet(path)

    # # Fill the result_matrix with hitcall values from filtered_data
    # for index, row in df_all.iterrows():
    #     assay_endpoint = row['assay_endpoint']
    #     compound = row['compound']
    #     hitcall = row['hitcall']
    #     df_all_merged.at[assay_endpoint, compound] = hitcall
    
    # for index, row in df_subset.iterrows():
    #     assay_endpoint = row['assay_endpoint']
    #     compound = row['compound']
    #     hitcall = row['hitcall']
    #     df_subset_merged.at[assay_endpoint, compound] = hitcall
    

    dfs_wrapped = [("All assay endpoints", df_all), ("Subset of all assay endpoints", df_subset)]
    tabs = st.tabs([df_wrapped[0] for df_wrapped in dfs_wrapped])

    color_bottom = "#f7f7eb"
    color_1 = "#a50aff"
    color_top = "#6254f7"

    for i, (df_name, df) in enumerate(dfs_wrapped):
        print(i, df_name)

        with tabs[i]:
            shape = df.shape
            st.write(f"Number of assay endpoints: {shape[0]}")
            st.write(f"Number of compounds: {shape[1]}")
            total_ones = df.values.sum()
            st.write(f"Total number concentration-response series (samples): {total_ones}")
            total_values = df.size
            ratio_ones_to_total = 1 - total_ones / total_values

            column_ones_counts = df.sum()
            sorted_columns = column_ones_counts.sort_values(ascending=False).index
            df = df[sorted_columns]
            df['ones_count'] = df.sum(axis=1)
            df = df.sort_values(by='ones_count', ascending=False)
            df = df.drop(columns='ones_count')

            colorscale = [[0, color_bottom], [ratio_ones_to_total, color_bottom],
                          [ratio_ones_to_total, color_top], [1, color_top]]

            showscale = not EXPORT

            fig = go.Figure(go.Heatmap(z=df.values, 
                                    #    colorscale=colorscale,
                                        showscale=showscale,
                                       colorbar=dict(tickvals=[0, 1], ticktext=["Absent   ", "Present   "])))

            font_size = 30  # Good for 33% zoom in browser (laptop resolution)
            if not EXPORT:
                fig.update_layout(
                    title=f"Presence matrix",
                    font=dict(size=font_size),
                    title_font=dict(size=font_size)
                )

            fig.update_layout(
                yaxis=dict(autorange="reversed", tickfont=dict(size=(font_size-10))),  # Adjust label font size
                xaxis=dict(tickfont=dict(size=font_size-10)),  # Hide x-axis tick labels
            )

            fig.update_xaxes(title_text="Compound Index", title_font=dict(size=font_size))
            fig.update_yaxes(title_text="Assay Endpoint Index", title_font=dict(size=font_size))
            # set tick labels font size in x axis and y axis


            
            st.plotly_chart(fig, use_container_width=True)

            df = df.T
            num_compounds = df.shape[0]
            num_assay_endpoints = df.shape[1]
            df = df.sum(axis=1).sort_values(ascending=False) / num_assay_endpoints
            fig = go.Figure(go.Scatter(x=df.index, y=df.values))
            fig.update_layout(title_text=f'Assay endpoint coverage, 100% = {num_assay_endpoints} assay endpoints')
            fig.update_xaxes(type='category')
            fig.update_traces(marker=dict(size=3))
            fig.update_traces(marker_symbol='circle-open')
            fig.update_xaxes(title_text=f'Compounds (truncated, total: {num_compounds})')
            fig.update_yaxes(title_text=f'Assay endpoint coverage')
            fig.update_yaxes(range=[0, 1])
            fig.update_layout(yaxis_tickformat='.0%')
            fig.update_layout(
                xaxis_title="Compound Index",
            )
            # fig.update_xaxes(showticklabels=False)
            st.plotly_chart(fig)
