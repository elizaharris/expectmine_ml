import os
import time

import pandas as pd

import sys

sys.path.append(r"C:\Users\bossh\Documents\GitHub")
from pytcpl.src.pipeline.pipeline_helper import query_db
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(parent_dir)

from ml.src.utils.helper import get_subset_aeids, compute_compounds_intersection
from ml.src.pipeline.constants import INPUT_ML_DIR_PATH, METADATA_SUBSET_DIR_PATH, FILE_FORMAT, METADATA_ALL_DIR_PATH,\
    FINGERPRINT_FILE, INPUT_FINGERPRINTS_DIR_PATH


def compute_assay_endpoint_compound_presence_matrix(ALL=1, SUBSET=1):
    print("Get intersection: compounds with fingerprint from structure & compounds tested in assay endpoints")
    with open(os.path.join(INPUT_FINGERPRINTS_DIR_PATH, f"{FINGERPRINT_FILE}_compounds.out"), 'r') as f:
        compounds_with_fingerprint = set(line.strip() for line in f)

    print("Compute assay endpoint-compound presence matrix (might take a while)")
    if ALL:
        aeid_compound_mapping_path = os.path.join(METADATA_ALL_DIR_PATH, f"aeid_compound_mapping{FILE_FORMAT}")

        if os.path.exists(aeid_compound_mapping_path):
            df = pd.read_parquet(aeid_compound_mapping_path)
        else:
            query_all = f"SELECT * FROM " \
                        f"(SELECT DISTINCT mc4.aeid, dsstox_substance_id " \
                        f"FROM (SELECT aeid, spid " \
                        f"FROM invitrodb_v3o5.mc4) as mc4 " \
                        f"JOIN sample ON mc4.spid = sample.spid " \
                        f"JOIN chemical ON sample.chid = chemical.chid) AS aeid_compound_map"
            df = query_db(query_all)
            df.to_parquet(aeid_compound_mapping_path, compression='gzip')


        presence_matrix = pd.crosstab(df['aeid'], df['dsstox_substance_id'])  # get presence

        # aeid_compound_presence_matrix_path = os.path.join(METADATA_ALL_DIR_PATH, f"aeid_compound_presence_matrix{FILE_FORMAT}")
        # presence_matrix.to_parquet(aeid_compound_presence_matrix_path, compression='gzip')

        # Get compounds
        compounds = presence_matrix.columns.tolist()
        compounds_tested_without_fingerprint = compute_compounds_intersection(METADATA_ALL_DIR_PATH, compounds, [], compounds_with_fingerprint)

        # Filter df by compounds that have a fingerprint from structure
        presence_matrix = presence_matrix.drop(columns=compounds_tested_without_fingerprint)
        aeid_compound_presence_matrix_path = os.path.join(METADATA_ALL_DIR_PATH,
                                                          f"aeid_compound_presence_matrix_with_fingerprint{FILE_FORMAT}")
        presence_matrix.to_parquet(aeid_compound_presence_matrix_path, compression='gzip')

    if SUBSET:
        aeids = get_subset_aeids()['aeid'].tolist()
        aeid_compound_presence_matrix_path = os.path.join(METADATA_ALL_DIR_PATH, f"aeid_compound_presence_matrix{FILE_FORMAT}")
        presence_matrix = pd.read_parquet(aeid_compound_presence_matrix_path)
        presence_matrix = presence_matrix[presence_matrix.index.isin(aeids)].reset_index(drop=True)

        path = os.path.join(INPUT_ML_DIR_PATH, f"{0}{FILE_FORMAT}")
        df = pd.read_parquet(path)
        pivot_table = df.pivot(index='aeid', columns='dsstox_substance_id', values='hitcall')
        pivot_table = pivot_table.fillna(-1)
        pivot_table_path = os.path.join(METADATA_SUBSET_DIR_PATH, f"pivot_table{FILE_FORMAT}")
        pivot_table.to_parquet(pivot_table_path, compression='gzip')

        cols_with_zero_count = presence_matrix.columns[presence_matrix.sum(axis=0) == 0]
        presence_matrix = presence_matrix.drop(columns=cols_with_zero_count)
        compounds_with_zero_count = cols_with_zero_count.tolist()
        aeid_compound_presence_matrix_path = os.path.join(METADATA_SUBSET_DIR_PATH,
                                                    f"aeid_compound_presence_matrix_with_fingerprint{FILE_FORMAT}")

        compounds = presence_matrix.columns.tolist()
        compounds_tested_without_fingerprint = compute_compounds_intersection(METADATA_SUBSET_DIR_PATH, compounds, compounds_with_zero_count, compounds_with_fingerprint)

        # Filter df by compounds that have a fingerprint from structure
        presence_matrix = presence_matrix.drop(columns=compounds_tested_without_fingerprint)
        presence_matrix.to_parquet(aeid_compound_presence_matrix_path, compression='gzip')
