import pandas as pd
import os
import json

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
from ml.src.pipeline.constants import OUTPUT_DIR_PATH
from ml.src.utils.helper import render_svg

import streamlit as st

st.set_page_config(
    layout="wide",
)

MOST_RECENT = 0
TARGET_RUN = "2023-10-18_22-36-10"

# Add a checkbox to the sidebar: Enable Reports
report_is_enabled = st.sidebar.checkbox("Enable Reports", value=True)

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

# Add a dropdown to the sidebar: Select single target variable
selected_target_variable = st.sidebar.selectbox('Select Target Variable', list(validation_results.keys()))

# Add a dropdown to the sidebar: Select single ml algorithm
selected_ml_algorithm = st.sidebar.selectbox('Select ML Algorithm', list(validation_results[selected_target_variable].keys()))

# Add a dropdown to the sidebar: Select single aeid
selected_aeid = str(st.sidebar.selectbox('Select AEID', list(aeid_paths[selected_target_variable][selected_ml_algorithm].keys())))

# Add a dropdown to the sidebar: Select single preprocessing model
selected_preprocessing_model = st.sidebar.selectbox('Select Preprocessing Model', list(model_paths[selected_target_variable][selected_ml_algorithm][selected_aeid].keys()))

# Add a dropdown to the sidebar: Select single estimator model
selected_estimator_model = st.sidebar.selectbox('Select Estimator Model', list(model_paths[selected_target_variable][selected_ml_algorithm][selected_aeid][selected_preprocessing_model].keys()))

# Add a dropdown to the sidebar: Select single validation type
selected_validation_type = st.sidebar.selectbox('Select Validation Type', list(validation_results[selected_target_variable][selected_ml_algorithm][selected_aeid][selected_preprocessing_model][selected_estimator_model].keys())[::-1])

# # Add a dropdown to the sidebar: Select single classification threshold
# selected_classification_threshold = st.sidebar.selectbox('Select Classification Threshold', list(validation_results[selected_aeid][selected_preprocessing_model][selected_estimator_model][selected_validation_type].keys()))
# Load threshold_value from estimator model path    
# threshold_value_path = os.path.join(model_paths[selected_aeid][selected_preprocessing_model][selected_estimator_model], f'threshold_value.json')

st.title(f'Validation Results (using 4 different classification thresholds)')
info_data = {
    "Target Variable": [selected_target_variable],
    "ML Algorithm": [selected_ml_algorithm],
    "aeid": [selected_aeid],
    "Feature Selection": [selected_preprocessing_model.split('_')[2]],
    "Estimator": [selected_estimator_model],
    "Validation Set": [selected_validation_type],
    # "Classification Threshold": [selected_classification_threshold]
}
info_df = pd.DataFrame(info_data)
st.dataframe(info_df, hide_index=True, use_container_width=True)

# Plot the validation results
validation_result = validation_results[selected_target_variable][selected_ml_algorithm][selected_aeid][selected_preprocessing_model][selected_estimator_model][selected_validation_type]
validation_results_df = pd.DataFrame(validation_result)


# Show report as dataframe
# st.subheader('Classification Report')
# st.dataframe(report_df)

# Plot confusion matrices for all 4 classification thresholds


confusion_matrices_paths = {}
roc_curve_paths = {}
reports = {}
threshold_names = []
threshold_values = []
for classification_threshold in validation_results[selected_target_variable][selected_ml_algorithm][selected_aeid][selected_preprocessing_model][selected_estimator_model][selected_validation_type].keys():
    path_suffix = os.path.join(selected_preprocessing_model, selected_estimator_model, selected_validation_type)
    cm_path = os.path.join(aeid_paths[selected_target_variable][selected_ml_algorithm][selected_aeid], path_suffix, f'cm_{classification_threshold}.svg')
    confusion_matrices_paths[classification_threshold] = cm_path

    report_path = os.path.join(aeid_paths[selected_target_variable][selected_ml_algorithm][selected_aeid], path_suffix, f'report_{classification_threshold}.csv')   
    reports[classification_threshold] = pd.read_csv(report_path).reset_index(drop=True).rename(columns={'Unnamed: 0': 'class'})
    threshold_names.append(classification_threshold)

roc_path = os.path.join(aeid_paths[selected_target_variable][selected_ml_algorithm][selected_aeid], path_suffix, f'roc_curve.svg')

# Create a 2x2 grid using columns
i = 0
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        i = 0
        st.subheader(f'Classification Threshold: Default=0.5')
        threshold = threshold_names[i]
        cm_path = confusion_matrices_paths[threshold]
        render_svg(open(cm_path).read())
        if report_is_enabled:
            with st.expander("Show Report"):
                st.dataframe(reports[threshold])
    with col2:
        i = 1
        st.subheader(f'Classification Threshold: cost(TPR, TNR) = 2 * (1 - TPR) + (1- TNR)')
        threshold = threshold_names[i]
        cm_path = confusion_matrices_paths[threshold]
        render_svg(open(cm_path).read())
        if report_is_enabled:
            with st.expander("Show Report"):
                st.dataframe(reports[threshold])

st.divider()

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        i = 3
        st.subheader(f'Classification Threshold: TPR ≈ 0.5')
        threshold = threshold_names[i]
        cm_path = confusion_matrices_paths[threshold]
        render_svg(open(cm_path).read())
        if report_is_enabled:
            with st.expander("Show Report"):
                st.dataframe(reports[threshold])
    with col2:
        i = 2
        st.subheader(f'Classification Threshold: TNR ≈ 0.5')
        threshold = threshold_names[i]
        cm_path = confusion_matrices_paths[threshold]
        render_svg(open(cm_path).read())
        if report_is_enabled:
            with st.expander("Show Report"):
                st.dataframe(reports[threshold])

st.divider()

render_svg(open(roc_path).read())