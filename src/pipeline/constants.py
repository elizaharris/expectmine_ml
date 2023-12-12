import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR_PATH = os.path.join(ROOT_DIR, '../../config')
CONFIG_PATH = os.path.join(CONFIG_DIR_PATH, 'config.yaml')
CONFIG_CLASSIFIERS_PATH = os.path.join(CONFIG_DIR_PATH, 'config_classification.yaml')
CONFIG_REGRESSORS_PATH = os.path.join(CONFIG_DIR_PATH, 'config_regression.yaml')


LOG_DIR_PATH = os.path.join(ROOT_DIR, '../../logs')
DATA_DIR_PATH = os.path.join(ROOT_DIR, '../../data')
EXPORT_DIR_PATH = os.path.join(DATA_DIR_PATH, 'export')
CSV_DIR_PATH = os.path.join(EXPORT_DIR_PATH, 'csv')
INPUT_DIR_PATH = os.path.join(DATA_DIR_PATH, 'input')
OUTPUT_DIR_PATH = os.path.join(DATA_DIR_PATH, 'output')
METADATA_DIR_PATH = os.path.join(DATA_DIR_PATH, 'metadata')

INPUT_ML_DIR_PATH = os.path.join(INPUT_DIR_PATH, 'ml')
INPUT_VALIDATION_DIR_PATH = os.path.join(INPUT_DIR_PATH, 'validation')
INPUT_FINGERPRINTS_DIR_PATH = os.path.join(INPUT_DIR_PATH, 'fingerprints')

COMPOUNDS_DIR_PATH = os.path.join(METADATA_DIR_PATH, 'compounds')
METADATA_SUBSET_DIR_PATH = os.path.join(METADATA_DIR_PATH, 'subset')
METADATA_ALL_DIR_PATH = os.path.join(METADATA_DIR_PATH, 'all')
CONC_DIR_PATH = os.path.join(METADATA_SUBSET_DIR_PATH, 'conc')
MASSBANK_DIR_PATH = os.path.join(METADATA_SUBSET_DIR_PATH, 'validation', 'massbank')
VALIDATION_COVERAGE_DIR_PATH = os.path.join(MASSBANK_DIR_PATH, 'validation_coverage')
VALIDATION_COVERAGE_PLOTS_DIR_PATH = os.path.join(MASSBANK_DIR_PATH, 'validation_coverage_plots')

REMOTE_DATA_DIR_PATH = os.path.join(ROOT_DIR, '../../../../pytcpl/data/')
REMOTE_METADATA_DIR_PATH = os.path.join(REMOTE_DATA_DIR_PATH, 'metadata/')
REMOTE_METADATA_SUBSET_DIR_PATH = os.path.join(REMOTE_METADATA_DIR_PATH, 'subset/')

FILE_FORMAT = '.parquet.gzip'
FINGERPRINT_FILE = "ToxCast_20231006_fingerprints"
