import base64
import os
from datetime import datetime

import requests
import streamlit as st
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd


from ml.src.pipeline.constants import REMOTE_METADATA_DIR_PATH, MASSBANK_DIR_PATH, FILE_FORMAT, INPUT_DIR_PATH, \
    OUTPUT_DIR_PATH, INPUT_FINGERPRINTS_DIR_PATH, METADATA_DIR_PATH, METADATA_ALL_DIR_PATH, METADATA_SUBSET_DIR_PATH, \
    VALIDATION_COVERAGE_DIR_PATH, VALIDATION_COVERAGE_PLOTS_DIR_PATH, CONC_DIR_PATH, INPUT_VALIDATION_DIR_PATH, \
    INPUT_ML_DIR_PATH, FINGERPRINT_FILE


def get_subset_aeids():
    aeids_path = os.path.join(REMOTE_METADATA_DIR_PATH, 'subset', f"aeids_target_assays{FILE_FORMAT}")
    aeids = pd.read_parquet(aeids_path)
    return aeids


def get_validation_compounds():
    validation_compounds_safe_and_unsafe = pd.read_parquet(os.path.join(MASSBANK_DIR_PATH, f"validation_compounds_safe_and_unsafe{FILE_FORMAT}"))['dsstox_substance_id']
    validation_compounds_safe = pd.read_parquet(os.path.join(MASSBANK_DIR_PATH, f"validation_compounds_safe{FILE_FORMAT}"))['dsstox_substance_id']
    validation_compounds_unsafe = pd.read_parquet(os.path.join(MASSBANK_DIR_PATH, f"validation_compounds_unsafe{FILE_FORMAT}"))['dsstox_substance_id']
    return validation_compounds_safe_and_unsafe, validation_compounds_safe, validation_compounds_unsafe


def calculate_binarized_hitcall_statistics(hitcall_infos, aeid, df_aeid, hitcall):
    hitcall_values = df_aeid[f"{hitcall}"]
    hitcall_values = (hitcall_values >= 0.5).astype(int)
    total_size = len(hitcall_values)
    num_active = hitcall_values.sum()
    num_inactive = total_size - num_active
    hit_ratio = num_active / total_size
    hitcall_infos[aeid] = {"total_size": total_size, "num_active": num_active, "num_inactive": num_inactive,
                           "hit_ratio": hit_ratio}


def init_directories():
    print("init_directories")
    os.makedirs(INPUT_DIR_PATH, exist_ok=True)
    os.makedirs(OUTPUT_DIR_PATH, exist_ok=True)
    os.makedirs(INPUT_ML_DIR_PATH, exist_ok=True)
    os.makedirs(INPUT_FINGERPRINTS_DIR_PATH, exist_ok=True)
    os.makedirs(INPUT_VALIDATION_DIR_PATH, exist_ok=True)
    os.makedirs(METADATA_DIR_PATH, exist_ok=True)
    os.makedirs(METADATA_ALL_DIR_PATH, exist_ok=True)
    os.makedirs(METADATA_SUBSET_DIR_PATH, exist_ok=True)
    os.makedirs(MASSBANK_DIR_PATH, exist_ok=True)
    os.makedirs(VALIDATION_COVERAGE_DIR_PATH, exist_ok=True)
    os.makedirs(VALIDATION_COVERAGE_PLOTS_DIR_PATH, exist_ok=True)
    os.makedirs(CONC_DIR_PATH, exist_ok=True)


def compute_compounds_intersection(directory, compounds, compounds_with_zero_count, compounds_with_fingerprint):
    compounds = set(compounds)
    compounds_path = os.path.join(directory, f"compounds_tested{FILE_FORMAT}")
    pd.DataFrame(compounds, columns=['dsstox_substance_id']).to_parquet(compounds_path, compression='gzip')

    with open(os.path.join(directory, 'compounds_tested.out'), 'w') as f:
        for compound in compounds:
            f.write(str(compound) + '\n')

    with open(os.path.join(directory, 'compounds_absent.out'), 'w') as f:
        for compound in compounds_with_zero_count:
            f.write(str(compound) + '\n')

    intersection = compounds_with_fingerprint.intersection(compounds)
    compounds_not_tested = compounds_with_fingerprint.difference(compounds)
    compounds_tested_without_fingerprint = compounds.difference(compounds_with_fingerprint)

    with open(os.path.join(directory, 'compounds_count.out'), 'w') as f:
        f.write(f"Number of compounds tested: {len(compounds)} \n")
        f.write(f"Number of compounds with fingerprint available: {len(compounds_with_fingerprint)} \n")
        f.write(f"Number of compounds tested and fingerprint available: {len(intersection)} \n")
        f.write(f"Number of compounds tested and no fingerprint available: {len(compounds_tested_without_fingerprint)} \n")
        f.write(f"Number of compounds not tested but fingerprint available: {len(compounds_not_tested)} \n")

    dest_path = os.path.join(directory, f'compounds_tested_with_fingerprint{FILE_FORMAT}')
    pd.DataFrame({'dsstox_substance_id': list(intersection)}).to_parquet(dest_path, compression='gzip')
    with open(os.path.join(directory, f'compounds_tested_with_fingerprint.out'), 'w') as f:
        for compound in intersection:
            f.write(compound + '\n')

    dest_path = os.path.join(directory, f'compounds_not_tested{FILE_FORMAT}')
    pd.DataFrame({'dsstox_substance_id': list(compounds_not_tested)}).to_parquet(dest_path, compression='gzip')
    with open(os.path.join(directory, f'compounds_not_tested.out'), 'w') as f:
        for compound in compounds_not_tested:
            f.write(compound + '\n')

    dest_path = os.path.join(directory, f'compounds_tested_without_fingerprint{FILE_FORMAT}')
    pd.DataFrame({'dsstox_substance_id': list(compounds_tested_without_fingerprint)}).to_parquet(dest_path, compression='gzip')
    with open(os.path.join(directory, f'compounds_tested_without_fingerprint.out'), 'w') as f:
        for compound in compounds_tested_without_fingerprint:
            f.write(compound + '\n')

    return compounds_tested_without_fingerprint


def get_sirius_fingerprints():
    massbank_sirius_df = pd.read_csv(
        os.path.join(INPUT_VALIDATION_DIR_PATH, f"massbank_from-sirius_fps_pos_curated_20231009_withDTXSID.csv"))
    fps_toxcast_df = pd.read_csv(os.path.join(INPUT_FINGERPRINTS_DIR_PATH, f"ToxCast_20231006_fingerprints.csv"))
    df = massbank_sirius_df.merge(fps_toxcast_df[['index']], how='inner', left_on='DTXSID', right_on='index')
    cols_to_select = df.filter(regex='^\d+$').columns.to_list() + ['index']  # only fingerprint

    df = df[cols_to_select]
    # df = df.drop_duplicates()
    df = df.groupby("index").mean()
    df = df.round().astype(int)  # Convert the mean values to binary fingerprints again
    df = df.reset_index().rename(columns={'index': 'dsstox_substance_id'})
    return df


def get_guid_dtxsid_mapping():
    global massbank_guid_acc_df, massbank_metadata_df
    # Table with DTXSID and accession column
    massbank_dtxsid_acc_df = pd.read_csv(
        os.path.join(INPUT_VALIDATION_DIR_PATH, f"massbank_quality_filtered_full_table_20231005.csv"))
    massbank_dtxsid_acc_df = massbank_dtxsid_acc_df.rename(columns={'CH$LINK': 'DTXSID'})
    massbank_dtxsid_acc_df['DTXSID'] = massbank_dtxsid_acc_df['DTXSID'].str.replace('COMPTOX ', '')
    # Table with GUID and accession column
    massbank_guid_acc_df = pd.read_csv(
        os.path.join(INPUT_VALIDATION_DIR_PATH, f"massbank_smiles_guid_acc_20231005.csv"))
    # Merge both tables on accession column
    massbank_metadata_df = massbank_dtxsid_acc_df.merge(massbank_guid_acc_df, on="accession")
    # Get GUID/DTXSID pairs
    pairs = massbank_metadata_df[['GUID', 'DTXSID']].drop_duplicates()
    # Check if chemical with DTXSID have a unique relationship to GUID
    unique_guid = pairs['GUID'].nunique()  # = 1783
    unique_dtxsid = pairs['DTXSID'].nunique()  # = 1466
    is_one_to_one_relationship = (unique_guid == unique_dtxsid) and (unique_guid == len(pairs))  # False
    print("'GUID'/'DTXSID' is_one_to_one_relationship:", is_one_to_one_relationship)
    return pairs


def collect_sirius_training_set_compounds():
    # Retrieve the inchi_keys of the training structures for positive and negative ion mode
    df1 = None
    df2 = None

    # Positive ion mode: predictor=1 (https://www.csi-fingerid.uni-jena.de/v2.6/api/fingerid/trainingstructures?predictor=1)
    training_structures_for_positive_ion_mode_url = 'https://www.csi-fingerid.uni-jena.de/v2.6/api/fingerid/trainingstructures?predictor=1'
    response = requests.get(training_structures_for_positive_ion_mode_url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        training_structures_for_positive_ion_mode_inchi_keys = [line.split('\t')[0] for line in soup.get_text().splitlines() if 'InChI=' in line]
        df1 = pd.DataFrame({'inchi_key': training_structures_for_positive_ion_mode_inchi_keys})
        path = os.path.join(INPUT_VALIDATION_DIR_PATH, f'training_structures_for_positive_ion_mode_inchi_keys{FILE_FORMAT}')
        df1.to_parquet(path)
    else:
        print(f"Failed to retrieve content. Status code: {response.status_code}")

    # Negative ion mode: predictor=2 (https://www.csi-fingerid.uni-jena.de/v2.6/api/fingerid/trainingstructures?predictor=2)
    training_structures_for_negative_ion_mode_url = 'https://www.csi-fingerid.uni-jena.de/v2.6/api/fingerid/trainingstructures?predictor=2'
    response = requests.get(training_structures_for_negative_ion_mode_url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        training_structures_for_negative_ion_mode_inchi_keys = [line.split('\t')[0] for line in soup.get_text().splitlines() if 'InChI=' in line]
        df2 = pd.DataFrame({'inchi_key': training_structures_for_negative_ion_mode_inchi_keys})
        path = os.path.join(INPUT_VALIDATION_DIR_PATH, f'training_structures_for_negative_ion_mode_inchi_keys{FILE_FORMAT}')
        df2.to_parquet(path)
    else:
        print(f"Failed to retrieve content. Status code: {response.status_code}")

    df = df1.merge(df2, on='inchi_key').drop_duplicates()
    # Save inchi_keys to .out file by splitting into batch size of 10000
    inchi_keys = df['inchi_key']
    inchi_keys.to_csv(os.path.join(INPUT_VALIDATION_DIR_PATH, f'training_structures_for_positive_and_negative_ion_mode_inchi_keys.csv'), index=False, header=False)
    
    # Load dtxsids from resulting inchi keys using batch search (https://comptox.epa.gov/dashboard/batch-search), do this manually!
    path = os.path.join(INPUT_VALIDATION_DIR_PATH, f'training_structures_for_positive_and_negative_ion_mode_inchi_keys_dtxsids.csv')
    dtxsids = pd.read_csv(path, sep=',')
    dtxsids = dtxsids['DTXSID']
    dtxsids = dtxsids.rename('dsstox_substance_id')
    dtxsids = dtxsids.dropna()
    dtxsids = dtxsids.drop_duplicates()
    dtxsids.to_csv(os.path.join(INPUT_VALIDATION_DIR_PATH, f'training_structures_for_positive_and_negative_ion_mode_dtxsids_unique.csv'), index=False, columns=['dsstox_substance_id'])

    return dtxsids


def csv_to_parquet_converter():
    print("Preprocess fingerprint from structure input file")
    src_path = os.path.join(INPUT_FINGERPRINTS_DIR_PATH, f"{FINGERPRINT_FILE}.csv")
    dest_path = os.path.join(INPUT_FINGERPRINTS_DIR_PATH, f"{FINGERPRINT_FILE}{FILE_FORMAT}")

    df = pd.read_csv(src_path)
    # Old: Skip the first 3 columns (Unnamed: 0, relativeIndex, absoluteIndex) and transpose the dataframe
    # df = df.iloc[:, 3:].T
    # data = df.iloc[1:].values.astype(int)
    # index = df.index[1:]
    # columns = df.iloc[0]

    # New: Skip  first column (Unnamed: 0), drop duplicated index.1
    if 'index.1' in df.columns:
        df = df.drop('index.1', axis=1)
    df = df.iloc[:, 1:]
    index = df['index']
    data = df.iloc[:, 1:].values.astype(np.uint8)
    columns = df.columns[1:]

    df = pd.DataFrame(data=data, index=index, columns=columns).reset_index()
    df = df.rename(columns={"index": "dsstox_substance_id"})
    df.to_parquet(dest_path, compression='gzip')
    df.sample(n=100, replace=False).to_csv(os.path.join(INPUT_FINGERPRINTS_DIR_PATH, f"test_sample.csv"), index=False)

    unique_chemicals = df['dsstox_substance_id'].unique()
    with open(os.path.join(INPUT_FINGERPRINTS_DIR_PATH, f"{FINGERPRINT_FILE}_compounds.out"), 'w') as f:
        f.write('\n'.join(list(filter(lambda x: x is not None, unique_chemicals))))

    # guid_dtxsid_mapping = get_guid_dtxsid_mapping()
    collect_sirius_training_set_compounds()
    sirius_fingerprints = get_sirius_fingerprints()
    sirius_fingerprints.to_csv(os.path.join(INPUT_FINGERPRINTS_DIR_PATH, f"sirius_massbank_fingerprints.csv"), index=False)
    sirius_fingerprints.to_parquet(os.path.join(INPUT_FINGERPRINTS_DIR_PATH, f"sirius_massbank_fingerprints{FILE_FORMAT}"), compression='gzip')

    unique_chemicals = sirius_fingerprints['dsstox_substance_id'].unique()
    with open(os.path.join(INPUT_FINGERPRINTS_DIR_PATH, f"sirius_fingerprints_compounds.out"), 'w') as f:
        f.write('\n'.join(list(filter(lambda x: x is not None, unique_chemicals))))


def folder_name_to_datetime(folder_name):
    return datetime.strptime(folder_name, '%Y-%m-%d_%H-%M-%S')


def get_verbose_name(t, default_threshold):
    threshold_mapping = {
                    'default': 'default_threshold',
                    'optimal': 'optimal_threshold',
                    'tpr': 'fixed_threshold_tpr',
                    'tnr': 'fixed_threshold_tnr'
    }
    return threshold_mapping.get(t)


def render_svg(svg):
    """
    Renders the given svg string.
    https://gist.github.com/treuille/8b9cbfec270f7cda44c5fc398361b3b1
    """
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    html = r'<img src="data:image/svg+xml;base64,%s"/>' % b64
    st.write(html, unsafe_allow_html=True)
