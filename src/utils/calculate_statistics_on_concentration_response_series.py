import os

import numpy as np
import pandas as pd
import json

from ml.src.pipeline.constants import REMOTE_DATA_DIR_PATH, FILE_FORMAT, METADATA_SUBSET_DIR_PATH, CONC_DIR_PATH
from ml.src.utils.helper import get_subset_aeids


def calculate_statistics_on_tested_concentrations():
    print("Calculate statistics on tested concentrations of all concentration-response series across compounds and assay endpoints")
    dest_path = os.path.join(CONC_DIR_PATH, f"{0}{FILE_FORMAT}")
    metrics_to_plot = ['num_points', 'num_groups', 'num_replicates', 'range_min', 'range_max']

    if not os.path.exists(dest_path):
        src_path = os.path.join(REMOTE_DATA_DIR_PATH, 'merged', 'output', f"0{FILE_FORMAT}")
        df = pd.read_parquet(src_path)
        df = df[['aeid', 'dsstox_substance_id', 'conc']]
        df['conc'] = df['conc'].apply(json.loads)

        def calculate_metrics(conc):
            num_groups = len(set(conc))
            num_replicates = len(conc) // num_groups
            num_points = len(conc)
            min_val = np.min(conc)
            max_val = np.max(conc)
            return pd.Series([num_points, num_groups, num_replicates, min_val, max_val], index=metrics_to_plot)

        df[metrics_to_plot] = df['conc'].apply(calculate_metrics)
        df.to_parquet(dest_path, compression='gzip')
    else:
        df = pd.read_parquet(dest_path)

    total_datapoints = df['conc'].apply(len).sum()
    with open(os.path.join(METADATA_SUBSET_DIR_PATH, 'concentrations.out'), 'w') as f:
        f.write(str(total_datapoints) + '\n')

    df.to_parquet(os.path.join(CONC_DIR_PATH, f"{0}{FILE_FORMAT}"), compression='gzip')

    aeids = get_subset_aeids()['aeid']
    for aeid in aeids:
        df_aeid = df[df['aeid'] == aeid]
        df_aeid = df_aeid[['dsstox_substance_id', 'conc']]
        df_aeid.to_parquet(os.path.join(CONC_DIR_PATH, f"{aeid}{FILE_FORMAT}"), compression='gzip')
