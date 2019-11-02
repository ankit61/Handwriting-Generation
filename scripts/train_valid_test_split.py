import glob, random
from collections import defaultdict
from shutil import copyfile

FULL_DATA_DIR = "data/full_data"
TRAIN_DATA_DIR = "data/train"
VALID_DATA_DIR = "data/validation"
TEST_DATA_DIR = "data/test"
TRAIN_PERCENT = 0.8
VALID_PERCENT = 0.05
TEST_PERCENT = 1 - TRAIN_PERCENT - VALID_PERCENT
MAX_LINE_POINTS = 800

IGNORE_MAX_LINE_FILES = True
RANDOM_SHUFFLE = False

data_file_paths = glob.glob(FULL_DATA_DIR + '/*')
writer_to_files_map = defaultdict(list)

for file_path in data_file_paths:
  filename = file_path.split('/')[-1]
  filename = filename.split('.')[0] # Remove extension
  line_id, writer_id = filename.split('_')
  if IGNORE_MAX_LINE_FILES:
    with open(file_path) as fp:
      file_data = fp.read()
      file_lines = file_data.split('\n')
      line_text, num_points, line_datapoints = file_lines[0], int(file_lines[1]), file_lines[2:]
      if num_points > MAX_LINE_POINTS:
        continue
  writer_to_files_map[writer_id].append(file_path)

num_files = sum([len(writer_to_files_map[writer_id]) for writer_id in writer_to_files_map])
print(num_files)
num_train_files = int(num_files * TRAIN_PERCENT)
num_valid_files = int(num_files * VALID_PERCENT)
num_test_files = int(num_files * TEST_PERCENT)

writers = list(writer_to_files_map.keys())
if RANDOM_SHUFFLE:
  random.shuffle(writers)
  
train_split_idx = 0
total_train_files = 0
while train_split_idx < len(writers) and total_train_files < num_train_files:
  total_train_files += len(writer_to_files_map[writers[train_split_idx]])
  train_split_idx += 1

valid_split_idx = train_split_idx
total_valid_files = 0
while valid_split_idx < len(writers) and total_valid_files < num_valid_files:
  total_valid_files += len(writer_to_files_map[writers[valid_split_idx]])
  valid_split_idx += 1

train_writers = writers[:train_split_idx]
valid_writers = writers[train_split_idx:valid_split_idx]
test_writers = writers[valid_split_idx:]

train_files = []
for writer in train_writers: train_files += writer_to_files_map[writer]
valid_files = []
for writer in valid_writers: valid_files += writer_to_files_map[writer]
test_files = []
for writer in test_writers: test_files += writer_to_files_map[writer]

for file_path in train_files:
  filename = file_path.split('/')[-1]
  copyfile(file_path, TRAIN_DATA_DIR + '/' + filename)

for file_path in valid_files:
  filename = file_path.split('/')[-1]
  copyfile(file_path, VALID_DATA_DIR + '/' + filename)

for file_path in test_files:
  filename = file_path.split('/')[-1]
  copyfile(file_path, TEST_DATA_DIR + '/' + filename)