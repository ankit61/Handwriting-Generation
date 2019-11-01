import os
import torch
from torch.utils.data import Dataset
from collections import defaultdict

MAX_LINE_POINTS = 800
MINIMUM_CHAR_FREQUENCY = 50

"""
Returns:
  char_count_dict: Dictionary of character to their frequency in data
    (ignores files that contain more datapoints than max_line_points)
"""
def _get_char_frequency_dict(data_dir, max_line_points):
  char_count_dict = defaultdict(int)
  for file_name in os.listdir(data_dir):
    with open(f'{data_dir}/{file_name}') as fp:
      file_lines = fp.read().split('\n')
      num_points = int(file_lines[1])
      if num_points > max_line_points: continue
      for c in file_lines[0]: 
        char_count_dict[c] += 1
  return dict(char_count_dict)

"""
Returns:
  chars_to_ignore: Set of characters to ignore based on their frequency in the data
    and the minimum_char_frequency
"""
def _get_char_set_to_ignore(char_freq_dict, minimum_char_frequency=MINIMUM_CHAR_FREQUENCY):
  chars_to_ignore = [char for char in char_freq_dict if char_freq_dict[char] < minimum_char_frequency]
  return set(chars_to_ignore)

"""
Returns:
  chars_to_ignore: Set of characters whose data should be ignored
  idx_to_char_map: Mapping from index to character for valid (non-ignored) characters
  char_to_idx_map: Mapping from character to index for valid (non-ignored) characters
"""
def _get_char_info_from_data(data_dir, max_line_points):
  char_freq_dict = _get_char_frequency_dict(data_dir, max_line_points)
  line_char_set = set(char_freq_dict.keys())
  # Ignore lines / datafiles whose text contains characters with freq < MINIMUM_CHAR_FREQUENCY
  chars_to_ignore = _get_char_set_to_ignore(char_freq_dict)
  # Get non-ignored characters present in data 
  valid_chars = list(line_char_set - chars_to_ignore)
  valid_chars = sorted(valid_chars)
  # Get mapping from character to index for one-hot transforms
  char_to_idx_map, idx_to_char_map = {}, {}
  for i, char in enumerate(valid_chars): 
    char_to_idx_map[char] = i
    idx_to_char_map[i] = char
  return chars_to_ignore, idx_to_char_map, char_to_idx_map

class HWGANDataset(Dataset):
  """
  * Expects data directory to have files names "line-id_writer-id.txt"
  * Data files should have line_text on line 1, num_data_points on line 2 and 
    x, y, p on each line thereafter
  """

  def __init__(self, data_dir, transform=None, max_line_points=MAX_LINE_POINTS):
    self.transform = transform
    self.data_dir = data_dir
    self.data = []

    # Get characters to ignore and char-and-index mappings based on data
    self.chars_to_ignore, self.idx_to_char_map, self.char_to_idx_map = _get_char_info_from_data(data_dir, max_line_points)

    # Load data
    for file_name in os.listdir(data_dir):
      file_name_parts = file_name.split('.')[0].split('_')
      line_id, writer_id = file_name_parts[0], file_name_parts[1]
      with open(f'{data_dir}/{file_name}') as fp:
        file_data = fp.read()
        file_lines = file_data.split('\n')
        line_text, num_points, line_datapoints = file_lines[0], int(file_lines[1]), file_lines[2:]
        if num_points > max_line_points: continue
        # Check intersection between line text and ignored chars
        if set(line_text) & self.chars_to_ignore: continue
      # Convert string datapoints to float tuples
      datapoints = [None] * num_points
      for i, line in enumerate(line_datapoints):
        split_line = line.split(',')
        datapoints.append((float(split_line[0]), float(split_line[1]), float(split_line[2]))) # (x, y, p)

      self.data.append((writer_id, line_text, datapoints))

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    
    writer_id, line_text, datapoints = self.data[idx]
    sample = {
      'writer_id': writer_id,
      'line_text': line_text,
      'datapoints': datapoints
    }

    if self.transform:
      sample = self.transform(sample)
    return sample
    