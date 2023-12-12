import streamlit as st
import pandas as pd
import numpy as np
import os
from joblib import Parallel, delayed
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from ml.src.pipeline.constants import METADATA_SUBSET_DIR_PATH, OUTPUT_DIR_PATH


st.set_page_config(
    layout="wide",
)


def blank_to_underscore(x):
    return x.replace(' ', '_')

rename = {
    'hitcall': 'hitcall',
    'hitcall_c': 'hitcall (c)',
    'True': 'positive',
    'False': 'negative',
    'macro avg': 'macro avg',
    'weighted avg': 'weighted avg',
    'val': 'Internal validation',
    'mb_val_structure': 'MB validation from structure',
    'mb_val_sirius': 'MB validation SIRIUS-predicted',
    'Internal validation': 'Internal',
    'MB validation from structure': 'MB structure',
    'MB validation SIRIUS-predicted': 'MB SIRIUS',
    'default': 'default=0.5',
    'tpr': 'TPR≈0.5',
    'tnr': 'TNR≈0.5',
    'optimal': 'cost(TPR,TNR)',
    'XGBClassifier': 'XGBoost',
    'XGBoost': 'XGB',
    'RF': 'RF',
    'RBF SVM': 'SVM',
    'MLP': 'MLP',
    'LR': 'LR',
    'RandomForestClassifier': 'RF',
    'LogisticRegression': 'LR',
    'SVC': 'RBF SVM',
    'MLPClassifier': 'MLP',
    'accuracy': 'accuracy',
}

subset_assay_info_columns = ["biological_process_target",
                            "MechanisticTarget",
                             "ToxicityEndpoint",
                             "intended_target_family",
                             "intended_target_family_sub",
                             "intended_target_type",
                             "intended_target_type_sub",
                             "burst_assay",
                             "assay_function_type",
                             "signal_direction",
                             "aeid",
                             "assay_component_endpoint_name",
                             ]


logs_folder = os.path.join(OUTPUT_DIR_PATH)
run_folder = st.sidebar.selectbox('Select Run', [run_folder for run_folder in os.listdir(logs_folder)][::-1])
folder = os.path.join(logs_folder, run_folder)

# Load the JSON data from the output files
with open(os.path.join(folder, 'aeid_paths.json'), 'r') as fp:
    aeid_paths = json.load(fp)

with open(os.path.join(folder, 'model_paths.json'), 'r') as fp:
    model_paths = json.load(fp)

with open(os.path.join(folder, 'feature_importances_paths.json'), 'r') as fp:
    feature_importances_paths = json.load(fp)

with open(os.path.join(folder, 'validation_results.json'), 'r') as fp:
    validation_results = json.load(fp)

with open(os.path.join(folder, 'validation_results_scores.json'), 'r') as fp:
    validation_results_scores = json.load(fp)



if "predictions_df" not in st.session_state:
    st.session_state.predictions_df = None

if "compounds" not in st.session_state:
    st.session_state.compounds = None

if "selected_compound" not in st.session_state:
    st.session_state.selected_compound = None

if "assay_info" not in st.session_state:
    st.session_state.assay_info = pd.read_parquet(os.path.join(METADATA_SUBSET_DIR_PATH, 'assay_info.parquet.gzip'))
    st.session_state.assay_info['aeid'] = st.session_state.assay_info['aeid'].astype(str)
    st.session_state.assay_info = st.session_state.assay_info[subset_assay_info_columns]


col1, col2 = st.columns(2)

with col1:
    col1.header("Predict on Environmental Sample Data")
    uploaded_file = st.file_uploader(
        "Input Environmental Sample Data: SIRIUS' Predicted Chemical Compound Fingerprints")
    test_data = None
    if uploaded_file is not None:
        test_data = pd.read_csv(uploaded_file)
        if test_data.shape[1] != 2363:
            st.error("Invalid file format. Please upload a CSV file with 2364 columns.")
            test_data = None
        elif test_data.columns[0] != "dsstox_substance_id":
            st.error("Invalid file format. Please upload a CSV file with the first column named 'dsstox_substance_id'.")
            test_data = None
        with st.expander("Preview"):
            st.dataframe(test_data, hide_index=True)
        st.success("File loaded successfully")
    else:
        st.info("Choose a CSV file.")

    model_paths = model_paths["hitcall"]["classification"]
    max_workers = -1
    predictions_df = None
    if test_data is not None:
        
        if st.button("Predict!"):
            with st.spinner('Wait for it...'):
                classifiers = {}
                for index, row in model_paths.items():
                    estimator = row['Feature_Selection_XGBClassifier']['XGBClassifier']
                    folder = os.path.dirname(estimator)
                    estimator_path = os.path.join(folder, 'mb_val_sirius', 'best_estimator_full_data.joblib')
                    feature_selection_model_path = folder[:-14]
                    feature_selection_model_path = os.path.join(feature_selection_model_path, 'preprocessing_model.joblib')
                    classifiers[index] = (joblib.load(feature_selection_model_path), joblib.load(estimator_path))


                def predict_for_endpoint(endpoint, clf_data, features):
                    feature_selection_model = clf_data[0]
                    features = feature_selection_model.transform(features)
                    clf = clf_data[1]
                    prediction = clf.predict(features)
                    return endpoint, prediction

                st.session_state.compounds = test_data['dsstox_substance_id'].values
                tasks = [(endpoint, clf_data, test_data.iloc[:, 1:]) for endpoint, clf_data in classifiers.items()]
                results = Parallel(n_jobs=max_workers)(delayed(predict_for_endpoint)(*task) for task in tasks)
                predictions = {endpoint: prediction for endpoint, prediction in results}
                predictions_df = pd.DataFrame(predictions)
                predictions_df.insert(0, 'dsstox_substance_id', test_data['dsstox_substance_id'])

                pivot_prediction = predictions_df.melt(id_vars=['dsstox_substance_id'], var_name='aeid',
                                                    value_name='prediction')
                predictions_df.set_index('dsstox_substance_id', inplace=True)
                predictions_df = predictions_df.transpose()
                st.session_state.predictions_df = predictions_df
                st.success("Prediction done!")

        # Insert a dropdown menu to choose the compound to display the prediction for
        if st.session_state.compounds is not None:
            st.session_state.selected_compound = st.selectbox("Select a compound to display the prediction for",
                                                              st.session_state.compounds)
            if st.session_state.selected_compound is not None:
                with st.expander("Prediction for selected compound"):
                    predictions_df = st.session_state.predictions_df[st.session_state.selected_compound]
                    horizontal_df = predictions_df.T
                    st.dataframe(horizontal_df)
                    # st.dataframe(st.session_state.predictions_df[st.session_state.selected_compound], hide_index=True)

with col2:
    if st.session_state.predictions_df is not None:
        col2.header("Interactive Assay Grouping")

        df = st.session_state.assay_info.merge(st.session_state.predictions_df, left_on='aeid', right_index=True)
        if 'Select' not in df.columns:
            df.insert(0, 'Select', True)
        group_column = st.sidebar.selectbox("Select Column to Group On", subset_assay_info_columns)

        grouped_df = df.groupby(group_column)

        selected_group_names = []
        grouped_new = {}
        length_meta = len(subset_assay_info_columns) + 1
        for group_name, group_data in grouped_df:
            # divide group_data into to parts of columns
            group_data_meta = group_data.iloc[:, :length_meta].reset_index(drop=True)
            group_data_preds = group_data.iloc[:, length_meta:-1].reset_index(drop=True)

            st.write(f"#### {group_name} Group:")

            checkbox_config = st.data_editor(
                group_data_meta,
                column_config={
                    "Select": st.column_config.CheckboxColumn(
                        "Select?",
                        help="Select assay endpoint for inclusion in predictive model",
                        default=True,
                    )
                },
                disabled=subset_assay_info_columns,
                hide_index=True,
                use_container_width=True,
            )
            df_group = pd.DataFrame(checkbox_config)
            grouped_new[group_name] = pd.concat([df_group, group_data_preds], axis=1)

        grouped_selected = {}
        for group_name, group_data in grouped_new.items():
            grouped_selected[group_name] = group_data[group_data['Select'] == True]

        counts = [(group_name, len(group_data)) for group_name, group_data in grouped_selected.items()]
        total_count = sum(count for _, count in counts)
        st.sidebar.write(f"Total assay endpoints selected: {total_count}")

        agg_hitcall = [(group_name, np.mean(group_data[st.session_state.selected_compound]), len(group_data)) for group_name, group_data in grouped_selected.items()]
        agg_hitcall_df = pd.DataFrame(agg_hitcall, columns=['Group', 'Aggregated Hitcall (avg.)', 'Count'])
        agg_hitcall_df_sorted = agg_hitcall_df.sort_values(by='Aggregated Hitcall (avg.)', ascending=False)

        # Add toxicity fingerprint column, round to 0 or 1
        agg_hitcall_df_sorted['Toxicitiy Fingerprint Bit'] = agg_hitcall_df_sorted['Aggregated Hitcall (avg.)'].apply(lambda x: 1 if x >= 0.5 else 0)

        fig = px.bar(agg_hitcall_df_sorted, x='Group', y='Aggregated Hitcall (avg.)', color='Group', text='Count', hover_data=['Count'],
                     title=f'{st.session_state.selected_compound}: Aggregated Hitcall per Group')
        # Add a line plot trace at y=0.5, make dashed
        fig.add_shape(type="line", x0=agg_hitcall_df_sorted['Group'].iloc[0], x1=agg_hitcall_df_sorted['Group'].iloc[-1], y0=0.5, y1=0.5, line=dict(color="red", dash="dash"))
        # Add an annotation text
        fig.add_annotation(
            text="Toxicity Fingerprint Bit Cutoff = 0.5",
            xref="paper",
            x=0.92,  # Adjust the x-coordinate as needed
            y=0.55,
            showarrow=False,
            font=dict(size=14, color="red")
        )


        #increse lable font size
        fig.update_layout(
            xaxis=dict(
                title=dict(
                    font=dict(
                        size=15,
                    )
                ),
                tickfont=dict(
                    size=15,
                )
            ),
            yaxis=dict(
                title=dict(
                    font=dict(
                        size=15,
                    )
                ),
                tickfont=dict(
                    size=15,
                )
            )
        )
        # increase title font size
        fig.update_layout(
            title=dict(
                font=dict(
                    size=16,
                )
            )
        )
        
        fig.update_traces(showlegend=False)
        # Set the x-axis as categorical
        fig.update_xaxes(type='category')

        # Create a radar chart using Plotly Express
        radar = px.line_polar(agg_hitcall_df_sorted, r='Aggregated Hitcall (avg.)', theta='Group', line_close=True)
        radar.update_traces(fill='toself')
        # Set the x-axis as categorical
        radar.update_xaxes(type='category')

        # Set the title
        radar.update_layout(
            title=f'{st.session_state.selected_compound}: Aggregated Hitcall per Group',
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(agg_hitcall_df_sorted['Aggregated Hitcall (avg.)']) + 0.2]  # Adjust the range as needed
                )
            )
        )


        with col1:
            with st.expander("Plot", expanded=True):
                st.plotly_chart(fig, use_container_width=True)
                st.plotly_chart(radar, use_container_width=True)
                st.dataframe(pd.DataFrame(agg_hitcall_df_sorted, columns=['Group', 'Toxicitiy Fingerprint Bit', 'Aggregated Hitcall (avg.)', 'Count']), hide_index=True)
