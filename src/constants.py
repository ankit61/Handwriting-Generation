import os
from pathlib import Path

CHARACTER_SET_SIZE  = 52
STYLE_VECTOR_SIZE   = 128
PRINT_FREQ          = 20

TENSORBOARDX_BASE_DIR = os.path.join(
    Path(os.path.dirname(os.path.abspath(__file__))).parent, 'runs')
MODELS_BASE_DIR = os.path.join(
    Path(os.path.dirname(os.path.abspath(__file__))).parent, 'models')