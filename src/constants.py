import os
from pathlib import Path
import getpass
from datetime import datetime

CHARACTER_SET_SIZE          = 34  # 68
STYLE_VECTOR_SIZE           = 256 # 128
PRINT_FREQ                  = 20

NUM_WRITERS                 = 195
CHARACTER_EMBEDDING_SIZE    = 128 # 64
LSTM_HIDDEN_SIZE            = 128 # 32
GEN_BATCH_SIZE              = 128 # 64

MAX_LINE_POINTS             = 100 # 800
MINIMUM_CHAR_FREQUENCY      = 0   # 50
MAX_LINE_TEXT_LENGTH        = 66
PAD_CHARACTER               = '@'

TENSORBOARDX_BASE_DIR = os.path.join(
    Path(os.path.dirname(os.path.abspath(__file__))).parent, os.path.join('runs', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + getpass.getuser()))
MODELS_BASE_DIR = os.path.join(
    Path(os.path.dirname(os.path.abspath(__file__))).parent, 'models')
DATA_BASE_DIR = os.path.join(
    Path(os.path.dirname(os.path.abspath(__file__))).parent, 'data')
