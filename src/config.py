import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Dataset URL (TUDataset standard repo)
DATASET_URL = "https://www.chrsmrrs.com/graphkerneldatasets/PROTEINS_full.zip"
DATASET_NAME = "PROTEINS_full"

# Topological Parameters
MAX_FILTRATION_SCALE = 15
HOMOLOGY_DIMENSIONS = (0, 1, 2)

# Random / CV
RANDOM_STATE = 42
CV_FOLDS = 10