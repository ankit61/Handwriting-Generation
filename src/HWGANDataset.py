import os
import torch
from torch.utils.data import Dataset
from utils import get_char_info_from_data
from constants import MINIMUM_CHAR_FREQUENCY, MAX_LINE_POINTS


class HWGANDataset(Dataset):
  """
  * Expects data directory to have files names "line-id_writer-id.txt"
  * Data files should have line_text on line 1, num_data_points on line 2 and 
    x, y, p on each line thereafter
  """

  def __init__(self, data_dir, transforms=None, max_line_points=MAX_LINE_POINTS):
    self.transforms = transforms
    self.data_dir = data_dir
    self.data = []

    # Get characters to ignore and char-and-index mappings based on data
    self.chars_to_ignore, self.idx_to_char_map, self.char_to_idx_map = get_char_info_from_data(data_dir, max_line_points, MINIMUM_CHAR_FREQUENCY)

    # Load data
    points_size = []
    for file_name in os.listdir(data_dir):
      file_name_parts = file_name.split('.')[0].split('_')
      line_id, writer_id = file_name_parts[0], int(file_name_parts[1])
      with open(f'{data_dir}/{file_name}') as fp:
        file_data = fp.read()
        file_lines = file_data.split('\n')
        line_text, num_points, line_datapoints = file_lines[0], int(file_lines[1]), file_lines[2:]
        if num_points > max_line_points: continue
        # Check intersection between line text and ignored chars
        points_size.append(num_points)
        if set(line_text) & self.chars_to_ignore: continue
      # Convert string datapoints to float tuples
      datapoints = [None] * num_points
      for i, line in enumerate(line_datapoints):
        split_line = line.split(',')
        datapoints[i] = (float(split_line[0]), float(split_line[1]), float(split_line[2])) # (x, y, p)

      datapoints = torch.tensor(datapoints, dtype=torch.float)

      sample = {
        'writer_id': writer_id,
        'line_text': line_text,
        'orig_line_text_len': len(line_text),
        'datapoints': datapoints
      }

      if self.transforms:
        if type(self.transforms) == list:
          for transform in self.transforms:
            sample = transform(sample)
      else:
        sample = self.transforms(sample)

      self.data.append(sample)

  def __len__(self):
    return len(self.data)

  def get_data_statistics(self):
    writer_ids = [sample['writer_id'] for sample in self.data]
    max_line_len = max([sample['orig_line_text_len'] for sample in self.data])
    print(f'Max Line Length: {max_line_len}')
    print(f'Num Writers: {len(set(writer_ids))}')

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()

    return self.data[idx]
