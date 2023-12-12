import pandas as pd
import numpy as np
import seaborn as sns
import os
import json

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
from ml.src.pipeline.constants import OUTPUT_DIR_PATH
from sklearn.metrics import jaccard_score
import plotly.express as px

import matplotlib.pyplot as plt


TARGET_RUN = "2023-10-18_22-36-10"


logs_folder = os.path.join(OUTPUT_DIR_PATH)
folder = os.path.join(logs_folder, TARGET_RUN)

with open(os.path.join(folder, 'aeid_paths.json'), 'r') as fp:
    aeid_paths = json.load(fp)

top_n = 10

def get_top_n_features(aeid_paths, top_n):
    all_features = pd.DataFrame()

    for aeid in aeid_paths[target_variable][ml_algorithm].keys():
        aeid_path = aeid_paths[target_variable][ml_algorithm][aeid]
        sorted_feature_importances_path = os.path.join(aeid_path, preprocessing_model, estimator_model, 'mb_val_sirius', 'sorted_feature_importances.csv')
        sorted_feature_importances = pd.read_csv(sorted_feature_importances_path).reset_index(drop=True).head(top_n)
        sorted_feature_importances['aeid'] = aeid
        all_features = pd.concat([all_features, sorted_feature_importances], ignore_index=True)

    # log transorm importances
    all_features['importances'] = all_features['importances'].apply(lambda x: 1 if x != 0 else 0)

    return all_features


def get_figure(all_features, linearized_feature_index, unique_aeid, name, shortname):
    # Create a mapping of old feature indexes to new linear indexes
    # feature_index_mapping = {feature_index: linear_index for linear_index, feature_index in enumerate(unique_features)}

    # Replace the original feature indexes with the new linear indexes
    # all_features['linearized_feature_index'] = all_features['feature'].map(feature_index_mapping)
    all_features = pd.DataFrame(all_features, columns=linearized_feature_index)
    all_features['aeid'] = unique_aeid  # Add the aeid as a column

    all_features = pd.melt(all_features, id_vars=['aeid'], var_name='feature', value_name='importance')

    nbinsx = all_features['aeid'].unique().shape[0]
    nbinsy = len(linearized_feature_index)
    #     all_features,
    #     x='aeid',
    #     y='feature',
    #     z='importance',
    #     title='Density Heatmap',
    # )

    fig = px.density_heatmap(all_features,
                        x='aeid',
                        y="feature",
                        hover_data=["importance"],
                        nbinsx=nbinsx,
                        nbinsy=nbinsy,
                        # marginal_x="histogram",
                        # marginal_y="histogram",


                        z="importance",
                        color_continuous_scale=[[0, "white"], [1, "black"]],
                        )

    fig.update_layout(
        autosize=False,
        width=nbinsx*6,
        height=nbinsy*6,
        margin=dict(
            pad=3
        ),
        title=f"{shortname}'s Most Significant Features: A Presence Matrix Created from the Top 10 Features across 345 Target Assay Endpoints",
        xaxis_title="Assay Index",
        yaxis_title="Feature (reindexed)",
        font=dict(
            size=21,
        )
    )

    fig.update_xaxes(tickfont=dict(size=4))

# dont sho x lables
#     fig.update_xaxes(showticklabels=False)
#     fig.update_yaxes(showticklabels=False)

# hide color bar
    fig.update_layout(coloraxis_showscale=False)

    dest_path = os.path.join(name)
    fig.write_image(dest_path, format='png', engine='kaleido', scale=5)

    return fig, all_features


def hamming_distance(matrix1, matrix2):
    if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]):
        raise ValueError("Matrices must have the same dimensions.")

    # Initialize the Hamming distance to 0
    hamming_dist = 0

    for i in range(len(matrix1)):
        for j in range(len(matrix1[0])):
            if matrix1[i][j] != matrix2[i][j]:
                hamming_dist += 1

    return hamming_dist

def match_score(matrix1, matrix2):
    if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]):
        raise ValueError("Matrices must have the same dimensions.")

    # Initialize the Hamming distance to 0
    matches = 0

    for i in range(len(matrix1)):
        for j in range(len(matrix1[0])):
            if matrix1[i][j] == matrix2[i][j] and matrix2[i][j] == 1:
                matches += 1

    return matches




# set main def

if __name__ == '__main__':
    target_variable = 'hitcall'
    ml_algorithm = 'classification'
    preprocessing_model = 'Feature_Selection_XGBClassifier'
    estimator_model = 'XGBClassifier'
    short1 = "XGBoost"
    name1 = f"{preprocessing_model}__{preprocessing_model}_feature_importance.png"


    all_features1 = get_top_n_features(aeid_paths, top_n)

    target_variable = 'hitcall'
    ml_algorithm = 'classification'
    preprocessing_model = 'Feature_Selection_RandomForestClassifier'
    estimator_model = 'RandomForestClassifier'
    short2 = "Random Forest"
    name2 = f"{preprocessing_model}__{preprocessing_model}_feature_importance.png"

    all_features2 = get_top_n_features(aeid_paths, top_n)

    # Combine the feature dataframes and create a set of unique features
    merged_features = pd.concat([all_features1, all_features2])
    unique_features = merged_features['feature'].unique()
    unique_features = merged_features.groupby('feature').filter(lambda x: x['importances'].sum() > top_n)['feature'].unique()
    # suffle the features
    np.random.shuffle(unique_features, )

    all_features1 = all_features1[all_features1['feature'].isin(unique_features)]
    all_features2 = all_features2[all_features2['feature'].isin(unique_features)]

    feature_index_mapping = {feature_index: linear_index for linear_index, feature_index in enumerate(unique_features)}

    all_features1['linearized_feature_index'] = all_features1['feature'].map(feature_index_mapping)
    all_features2['linearized_feature_index'] = all_features2['feature'].map(feature_index_mapping)


    unique_aeid = merged_features['aeid'].unique()
    
    aeid_to_index = {aeid: i for i, aeid in enumerate(unique_aeid)}


    # Create separate matrices for all_features1 and all_features2
    def create_feature_matrix(features_df, unique_features, aeid_to_index):
        num_aeid = len(aeid_to_index)
        num_features = len(unique_features)
        feature_matrix = np.zeros((num_aeid, num_features))

        for idx, row in features_df.iterrows():
            aeid = row['aeid']
            feature = row['feature']
            importance = row['importances']

            aeid_index = aeid_to_index[aeid]
            feature_index = np.where(unique_features == feature)[0][0]

            feature_matrix[aeid_index][feature_index] = importance

        return feature_matrix

    matrix1 = create_feature_matrix(all_features1, unique_features, aeid_to_index).astype(np.uint8)
    matrix2 = create_feature_matrix(all_features2, unique_features, aeid_to_index).astype(np.uint8)


    def have_same_zero_sum_rows(matrix1, matrix2):
        if matrix1.shape != matrix2.shape:
            return False  # Matrices must have the same shape to compare row sums.

        num_rows, num_cols = matrix1.shape

        for row_index in range(num_rows):
            sum1 = np.sum(matrix1[row_index])
            sum2 = np.sum(matrix2[row_index])
            if sum1 == 0 and sum2 == 0:
                print(f"Row {row_index} has zero sum in both matrices.")
                return True

        return False


    # result = have_same_zero_sum_rows(matrix1, matrix2)

    # if result:
    #     print("Matrices have zero sum rows at the same index.")
    # else:
    #     print("Matrices do not have zero sum rows at the same index.")

    matrix1_feature_sum = matrix1.sum(axis=0)
    matrix2_feature_sum = matrix2.sum(axis=0)

    # # feature_index_mapping = {feature_index: linear_index for linear_index, feature_index in enumerate(union)}
    #
    # all_features1['linearized_feature_index'] = all_features1['feature'].map(feature_index_mapping)
    # all_features2['linearized_feature_index'] = all_features2['feature'].map(feature_index_mapping)


    fig1 = get_figure(matrix1, list(feature_index_mapping.values()), unique_aeid, name1, short1)

    fig2 = get_figure(matrix2, list(feature_index_mapping.values()), unique_aeid, name2, short2)


# Hamming distance

print("Total number of feature remaining:", matrix1.shape[1])
print("Total number of aeid:", matrix1.shape[0])
print("Total constraint feature space matrix1:", matrix1.shape[0] * matrix1.shape[1])
print("Total constraint feature space matrix2:", matrix2.shape[0] * matrix2.shape[1])
print(f"Sum top features matrix1: {matrix1.sum()}")
print(f"Sum top features matrix2: {matrix2.sum()}")
print(f"Sum top features together: {matrix1.sum() + matrix2.sum()}")
distance = hamming_distance(matrix1, matrix2)
matches = match_score(matrix1, matrix2)
print("Hamming distance:", distance)
print("Matching features:", matches)


def plot_heatmap(matrix1, matrix2):
    fig, ax = plt.subplots(figsize=(matrix1.shape[1] / 30, matrix1.shape[0] / 30))

    # Create a binary matrix indicating matches
    match_matrix = np.multiply(matrix1, matrix2)

    match_matrix = np.flip(match_matrix, axis=0)

    # Create a heatmap of the match_matrix
    ax.imshow(match_matrix, cmap="YlGnBu", aspect="auto")

    plt.title("Matching Features")
    plt.ylabel("Feature (reindexed)")
    plt.xlabel("Assay Index")

    # Remove the x-axis tick labels
    ax.set_xticks([])
    ax.set_yticks([])

    # Remove the color bar
    cbar = plt.colorbar(ax.imshow(match_matrix, cmap="gray_r"))
    cbar.remove()
    plt.savefig("matching_features.png", format='png', dpi=600)


# Use the function to plot the heatmap
plot_heatmap(matrix1.T, matrix2.T)







