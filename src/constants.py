import os
from pathlib import Path
import getpass
from datetime import datetime
import multiprocessing

CHARACTER_SET_SIZE          = 69
STYLE_VECTOR_SIZE           = 256
PRINT_FREQ                  = 20
INTERMITTENT_OUTPUT_FREQ    = 10 # Num batches between outputs

NUM_WRITERS                 = 195
CHARACTER_EMBEDDING_SIZE    = 64 # 64
RNN_HIDDEN_SIZE             = 3 # 32
RNN_DEPTH                   = 1
RNN_TYPE                    = 'LSTM'

GEN_BATCH_SIZE              = 128

MAX_LINE_POINTS             = 800
MINIMUM_CHAR_FREQUENCY      = 50
MAX_LINE_TEXT_LENGTH        = 66
PAD_CHARACTER               = '@'

MAX_DATA_LOAD_THREADS       = multiprocessing.cpu_count()

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
