import os
from pathlib import Path

CHARACTER_SET_SIZE     = 52
STYLE_VECTOR_SIZE      = 128
PRINT_FREQ             = 20

MAX_LINE_POINTS        = 800
MINIMUM_CHAR_FREQUENCY = 50
MAX_LINE_TEXT_LENGTH   = 66
PAD_CHARACTER          = '@'

TENSORBOARDX_BASE_DIR = os.path.join(
    Path(os.path.dirname(os.path.abspath(__file__))).parent, 'runs')
MODELS_BASE_DIR = os.path.join(
    Path(os.path.dirname(os.path.abspath(__file__))).parent, 'models')