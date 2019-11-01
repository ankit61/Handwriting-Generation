import torch
import numpy as np

class CoordinatesToDeltaTransform(object):
  """Converts a list of x, y, p datapoints to their delta counterparts
  * delta x = x_i - x_(i-1)
  """

  def __init__(self):
    pass

  def __call__(self, sample):
    datapoints = sample['datapoints']
    # Assume first point of stroke to be the origin
    delta_datapoints = [(0, 0, 0)]
    for i in range(1, len(datapoints)):
      delta_datapoints.append((datapoints[i][0] - datapoints[i-1][0], datapoints[i][1] - datapoints[i-1][1], datapoints[i][2]))
    
    return {
      'writer_id': sample['writer_id'],
      'line_text': sample['line_text'],
      'datapoints': delta_datapoints
    }

class LineToOneHotMatrix(object):
  """Converts a string of line_text into a one hot matrix
  """

  def __init__(self, char_to_idx_map):
    self.char_to_idx_map = char_to_idx_map
    self.num_valid_chars = len(self.char_to_idx_map)

  def __call__(self, sample):
    line_text = sample['line_text']
    line_onehot_matrix = np.zeros((len(line_text), self.num_valid_chars))
    for i, c in enumerate(line_text):
      line_onehot_matrix[i][self.char_to_idx_map[c]] = 1
    
    return {
      'writer_id': sample['writer_id'],
      'line_matrix': line_onehot_matrix,
      'line_text': sample['line_text'],
      'datapoints': sample['datapoints']
    }
