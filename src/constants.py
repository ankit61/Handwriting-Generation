import os
from pathlib import Path
import getpass
from datetime import datetime
import multiprocessing

EPOCHS                      = 300

CHARACTER_SET_SIZE          = 32
STYLE_VECTOR_SIZE           = 256
PRINT_FREQ                  = 20
INTERMITTENT_OUTPUT_FREQ    = 5 # Num batches between outputs

ATTENTION_NUM_GAUSSIAN_FUNC = 5

NUM_WRITERS                 = 195
CHARACTER_EMBEDDING_SIZE    = 32
RNN_HIDDEN_SIZE             = 64
RNN_DEPTH                   = 4
RNN_TYPE                    = 'LSTM'
RNN_OUT_SIZE                = 3

GEN_BATCH_SIZE              = 64

MAX_LINE_POINTS             = 100
MINIMUM_CHAR_FREQUENCY      = 0
MAX_LINE_TEXT_LENGTH        = 66
PAD_CHARACTER               = '@'
STANDARDIZE_POINTS          = True
MAX_DATA_LOAD_THREADS       = multiprocessing.cpu_count()

XY_PRED_TOLERANCE           = 0.05
SIGMOID_THRESH_P            = 0.5

MIN_LEARNING_RATE           = 0.000001

TENSORBOARDX_BASE_DIR = os.path.join(
    Path(os.path.dirname(os.path.abspath(__file__))).parent, os.path.join('runs', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + getpass.getuser()))
INTERMITTENT_OUTPUTS_BASE_DIR = os.path.join(
    Path(os.path.dirname(os.path.abspath(__file__))).parent, os.path.join('intermittent_outputs', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + getpass.getuser()))
MODELS_BASE_DIR = os.path.join(
    Path(os.path.dirname(os.path.abspath(__file__))).parent, 'models')
DATA_BASE_DIR = os.path.join(
    Path(os.path.dirname(os.path.abspath(__file__))).parent, 'data')

DATA_FILE_BLACKLIST = set([
    'd09-651z-05_10083',
    'b05-465z-08_10206',
    'a07-421z-05_10174'
])
