import os
from pathlib import Path

CHARACTER_SET_SIZE          = 68
STYLE_VECTOR_SIZE           = 128
PRINT_FREQ                  = 20

NUM_WRITERS                 = 195
CHARACTER_EMBEDDING_SIZE    = 64
LSTM_HIDDEN_SIZE            = 32
GEN_BATCH_SIZE              = 64

MAX_LINE_POINTS             = 800
MINIMUM_CHAR_FREQUENCY      = 50
MAX_LINE_TEXT_LENGTH        = 66
PAD_CHARACTER               = '@'

STYLE_GENERATOR_ARCH        = 'vgg11_bn'

TENSORBOARDX_BASE_DIR = os.path.join(
    Path(os.path.dirname(os.path.abspath(__file__))).parent, 'runs')
MODELS_BASE_DIR = os.path.join(
    Path(os.path.dirname(os.path.abspath(__file__))).parent, 'models')
DATA_BASE_DIR = os.path.join(
    Path(os.path.dirname(os.path.abspath(__file__))).parent, 'data')