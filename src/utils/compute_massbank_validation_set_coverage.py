import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


from ml.src.utils.helper import get_subset_aeids, get_validation_compounds
from ml.src.pipeline.constants import FILE_FORMAT, \
    METADATA_SUBSET_DIR_PATH, VALIDATION_COVERAGE_DIR_PATH, VALIDATION_COVERAGE_PLOTS_DIR_PATH, \
    INPUT_VALIDATION_DIR_PATH, MASSBANK_DIR_PATH, INPUT_ML_DIR_PATH, INPUT_FINGERPRINTS_DIR_PATH


def compute_massbank_validation_set_coverage(COLLECT_STATS=1, COMPUTE_PRESENCE_MATRIX=1, COMPUTE_HIT_CALL_MATRIX=1):
    print("Compute massbank validation set coverage")
    data_dict = {}
    aeids = get_subset_aeids()['aeid']
    for aeid in aeids:
        src_path = os.path.join(INPUT_ML_DIR_PATH, f"{aeid}{FILE_FORMAT}")
        data_dict[aeid] = pd.read_parquet(src_path)

    validation_compounds_safe_and_unsafe, validation_compounds_safe, validation_compounds_unsafe = get_validation_compounds()
    subset_ids_list = [validation_compounds_safe_and_unsafe, validation_compounds_safe, validation_compounds_unsafe]
    subset_ids_list_names = ["validation_compounds_safe_and_unsafe", "validation_compounds_safe", "validation_compounds_unsafe"]
    for subset_ids_list_name in subset_ids_list_names:
        os.makedirs(os.path.join(VALIDATION_COVERAGE_DIR_PATH, subset_ids_list_name), exist_ok=True)
        os.makedirs(os.path.join(VALIDATION_COVERAGE_PLOTS_DIR_PATH, subset_ids_list_name), exist_ok=True)

    # Get set of all compounds tested
    compounds_path = os.path.join(METADATA_SUBSET_DIR_PATH, f"compounds_tested_with_fingerprint{FILE_FORMAT}")
    all_compounds = pd.read_parquet(compounds_path)['dsstox_substance_id']

    if COLLECT_STATS:
        for j, subset_ids in enumerate(subset_ids_list):
            coverage_info = {}
            compound_list_name = subset_ids_list_names[j]

            for i, (aeid, df) in enumerate(data_dict.items()):
                intersection = df[df['dsstox_substance_id'].isin(subset_ids)]
                num_compounds = len(subset_ids)
                overlap = len(intersection)
                relative_coverage = overlap / num_compounds

                coverage_info[aeid] = {'compounds': num_compounds,
                                       'overlap': overlap,
                                       'relative_coverage': relative_coverage}

                filename = os.path.join(VALIDATION_COVERAGE_DIR_PATH, compound_list_name, f"{aeid}{FILE_FORMAT}")
                intersection.to_parquet(filename, compression='gzip')

                filename = os.path.join(VALIDATION_COVERAGE_DIR_PATH, compound_list_name, f"{aeid}.out")
                with open(filename, 'w') as file:
                    file.write(f"Compounds: {num_compounds}\n")
                    file.write(f"Overlap: {overlap}\n")
                    file.write(f"Relative_coverage: {relative_coverage}\n")

            coverage_info = pd.DataFrame(coverage_info).T
            coverage_info.to_parquet(
                os.path.join(VALIDATION_COVERAGE_PLOTS_DIR_PATH, compound_list_name, f"coverage_info{FILE_FORMAT}"),
                compression='gzip')

    if COMPUTE_PRESENCE_MATRIX:
        for j, subset_ids in enumerate(subset_ids_list):
            compound_list_name = subset_ids_list_names[j]

            presence_matrix = np.zeros((len(data_dict), len(all_compounds)), dtype=int)

            def process(aeid, df):
                mask_a = np.isin(all_compounds, df['dsstox_substance_id'])
                mask_b = np.isin(all_compounds, subset_ids)
                final_mask = np.logical_and(mask_a, mask_b)
                presence_row = np.where(final_mask, 2, mask_a).astype(int)
                return aeid, presence_row

            num_cores = 8
            with Parallel(n_jobs=num_cores) as parallel:
                results = parallel(
                    delayed(process)(aeid, df) for aeid, df in data_dict.items()
                )

            for i, (aeid, presence_row) in enumerate(results):
                presence_matrix[i, :] = presence_row

            save_path = os.path.join(VALIDATION_COVERAGE_PLOTS_DIR_PATH, compound_list_name,
                                     f"presence_matrix{FILE_FORMAT}")
            presence_matrix = pd.DataFrame(presence_matrix, index=[aeid for aeid, df in data_dict.items()],
                                           columns=all_compounds)
            presence_matrix.to_parquet(save_path, compression='gzip')

    if COMPUTE_HIT_CALL_MATRIX:
        for j, subset_ids in enumerate(subset_ids_list):
            compound_list_name = subset_ids_list_names[j]
            print(compound_list_name)
            heatmap_data = {}
            hitcall_infos = {}

            special_value = -1  # You can change this to any value you prefer
            for aeid, df in data_dict.items():
                hitcall_values = df[df['dsstox_substance_id'].isin(subset_ids)].set_index('dsstox_substance_id')[
                    'hitcall']
                hitcall_values = (hitcall_values >= 0.5).astype(int)

                total_size = len(hitcall_values)
                num_active = hitcall_values.sum()
                num_inactive = total_size - num_active
                hit_ratio = num_active / total_size
                hitcall_infos[aeid] = {"total_size": total_size, "num_active": num_active, "num_inactive": num_inactive,
                                       "hit_ratio": hit_ratio}

                missing_compounds = set(subset_ids) - set(hitcall_values.index)
                missing_data = pd.Series([special_value] * len(missing_compounds), index=missing_compounds)
                heatmap_data[aeid] = pd.concat([hitcall_values, missing_data])

            hitcall_infos_df = pd.DataFrame(hitcall_infos,
                                            index=['total_size', 'num_active', 'num_inactive', 'hit_ratio']).T
            hitcall_infos_df.reset_index(inplace=True)
            hitcall_infos_df.rename(columns={'index': 'aeid'}, inplace=True)
            dest_path = os.path.join(VALIDATION_COVERAGE_PLOTS_DIR_PATH, compound_list_name,
                                     f"hitcall_infos{FILE_FORMAT}")
            hitcall_infos_df.to_parquet(dest_path, compression='gzip')

            hitcall_infos_df_all_compounds = pd.read_parquet(
                os.path.join(METADATA_SUBSET_DIR_PATH, f"hitcall_infos{FILE_FORMAT}"))

            merged_df = hitcall_infos_df_all_compounds.merge(hitcall_infos_df, on='aeid', suffixes=('_df1', '_df2'))

            plt.figure(figsize=(8, 6))
            plt.scatter(merged_df['hit_ratio_df1'], merged_df['hit_ratio_df2'], marker='o', alpha=0.5)
            # Add the equation line y = x
            max_value = max(merged_df['hit_ratio_df1'].max(), merged_df['hit_ratio_df2'].max())
            plt.plot([0, max_value], [0, max_value], linestyle='solid', color='blue', label='Equal distribution of active compounds in both validation sets')
            plt.ylabel('Active compounds in %: Massbank Validation Set', fontsize=14)
            plt.xlabel('Active compounds in %: Assay Endpoint', fontsize=16)
            plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
            plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
            plt.tick_params(axis='both', labelsize=14) 

            plt.title('Active Compounds in % per Assay Endpoint vs. Massbank Validation Set')
            # plt.title('Representativeness of Validation Set')
            # plt.suptitle('Active Compounds in %')
            plt.grid(True)
            plt.legend()
            # plt.show()
            plt.savefig(os.path.join(VALIDATION_COVERAGE_PLOTS_DIR_PATH, compound_list_name, f"hit_ratio_scatter_plot.svg"))

            def custom_sum(row):
                return row.apply(lambda x: 0 if x < 0 else 1).sum()

            heatmap_df = pd.DataFrame(heatmap_data).T
            column_ones_counts = heatmap_df.apply(custom_sum, axis=0)
            sorted_columns = column_ones_counts.sort_values(ascending=False).index
            heatmap_df = heatmap_df[sorted_columns]

            heatmap_df['ones_count'] = heatmap_df.apply(custom_sum, axis=1)
            heatmap_df = heatmap_df.sort_values(by='ones_count', ascending=False)
            heatmap_df = heatmap_df.drop(columns='ones_count')

            plt.figure(figsize=(12, 8))
            plt.imshow(heatmap_df.values, cmap='viridis', interpolation='nearest', aspect='auto', vmin=-1, vmax=1)

            # Set x and y-axis labels
            plt.xticks(range(len(heatmap_df.columns)), heatmap_df.columns, rotation=90)
            plt.yticks(range(len(heatmap_df.index)), heatmap_df.index)
            plt.colorbar()
            # plt.show() # , dpi=300
            plt.savefig(os.path.join(VALIDATION_COVERAGE_PLOTS_DIR_PATH, compound_list_name, f"hitcall_matrix.svg"))

            # for i, (aeid, presence_row) in enumerate(results):
            #     presence_matrix[i, :] = presence_row

            # save_path = os.path.join(VALIDATION_COVERAGE_PLOTS_DIR_PATH, compound_list_name, f"presence_matrix{FILE_FORMAT}")
            # presence_matrix = pd.DataFrame(presence_matrix, index=[aeid for aeid, df in data_dict.items()], columns=all_compounds)
            # presence_matrix.to_parquet(save_path, compression='gzip')


def get_validation_set():
    print("Prepare validation set (massbank)")
    massbank_dtxsid_with_records_path = os.path.join(INPUT_FINGERPRINTS_DIR_PATH, f"sirius_massbank_fingerprints.csv")
    massbank_dtxsid_used_for_sirius_training_path = os.path.join(INPUT_VALIDATION_DIR_PATH, f"training_structures_for_positive_and_negative_ion_mode_dtxsids_unique.csv")

    massbank_dtxsid_with_records_df = pd.read_csv(massbank_dtxsid_with_records_path)
    massbank_dtxsid_used_for_sirius_training_df = pd.read_csv(massbank_dtxsid_used_for_sirius_training_path)

    # Filter compounds that have fingerprint from structure
    compounds_tested_with_fingerprint_path = os.path.join(METADATA_SUBSET_DIR_PATH, f"compounds_tested_with_fingerprint{FILE_FORMAT}")
    compounds_tested_with_fingerprint = pd.read_parquet(compounds_tested_with_fingerprint_path)['dsstox_substance_id']
    massbank_dtxsid_with_records_df = massbank_dtxsid_with_records_df[massbank_dtxsid_with_records_df['dsstox_substance_id'].isin(compounds_tested_with_fingerprint)]
    massbank_dtxsid_used_for_sirius_training_with_records_df = massbank_dtxsid_used_for_sirius_training_df[massbank_dtxsid_used_for_sirius_training_df['dsstox_substance_id'].isin(compounds_tested_with_fingerprint)]
    massbank_dtxsid_with_records = massbank_dtxsid_with_records_df['dsstox_substance_id']
    massbank_dtxsid_used_for_sirius_training_with_records = massbank_dtxsid_used_for_sirius_training_with_records_df['dsstox_substance_id']

    massbank_dtxsid_with_records = set(massbank_dtxsid_with_records)
    massbank_dtxsid_used_for_sirius_training_with_records = set(massbank_dtxsid_used_for_sirius_training_with_records)

    compounds_intersection = massbank_dtxsid_with_records.intersection(massbank_dtxsid_used_for_sirius_training_with_records)
    compounds_safe_for_validation = massbank_dtxsid_with_records.difference(massbank_dtxsid_used_for_sirius_training_with_records)
    compounds_unsafe_for_validation = massbank_dtxsid_used_for_sirius_training_with_records

    with open(os.path.join(MASSBANK_DIR_PATH, 'compounds_count.out'), 'w') as f:
        f.write(f"massbank_dtxsid_with_records: {len(massbank_dtxsid_with_records)} \n")
        f.write(f"massbank_dtxsid_with_records_sirius_training: {len(massbank_dtxsid_used_for_sirius_training_with_records)} \n")
        f.write("\n")
        f.write(f"intersection: {len(compounds_intersection)} \n")
        f.write("\n")
        f.write(f"validation_compounds_safe: {len(compounds_safe_for_validation)} \n")
        f.write(f"validation_compounds_unsafe: {len(compounds_unsafe_for_validation)} \n")

    path = os.path.join(MASSBANK_DIR_PATH, f"validation_compounds_safe{FILE_FORMAT}")
    df = pd.DataFrame({'dsstox_substance_id': list(compounds_safe_for_validation)})
    df.to_parquet(path, compression='gzip')
    with open(os.path.join(MASSBANK_DIR_PATH, 'validation_compounds_safe.out'), 'w') as f:
        for compound in compounds_safe_for_validation:
            f.write(compound + '\n')

    path = os.path.join(MASSBANK_DIR_PATH, f"validation_compounds_unsafe{FILE_FORMAT}")
    df = pd.DataFrame({'dsstox_substance_id': list(compounds_unsafe_for_validation)})
    df.to_parquet(path, compression='gzip')
    with open(os.path.join(MASSBANK_DIR_PATH, 'validation_compounds_unsafe.out'), 'w') as f:
        for compound in compounds_unsafe_for_validation:
            f.write(compound + '\n')

    path = os.path.join(MASSBANK_DIR_PATH, f"validation_compounds_safe_and_unsafe{FILE_FORMAT}")
    df = pd.DataFrame({'dsstox_substance_id': list(massbank_dtxsid_with_records)})
    df.to_parquet(path, compression='gzip')
    with open(os.path.join(MASSBANK_DIR_PATH, 'validation_compounds_safe_and_unsafe.out'), 'w') as f:
        for compound in massbank_dtxsid_with_records:
            f.write(compound + '\n')


if __name__ == '__main__':
    get_validation_set()
    compute_massbank_validation_set_coverage(COLLECT_STATS=1, COMPUTE_PRESENCE_MATRIX=1, COMPUTE_HIT_CALL_MATRIX=1)