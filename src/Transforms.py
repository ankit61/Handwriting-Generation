import torch, copy
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
    
    new_sample = copy.deepcopy(sample)
    new_sample['datapoints'] = torch.tensor(delta_datapoints, dtype=torch.float)
    return new_sample

class LineToOneHotMatrixTransform(object):
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
    
    new_sample = copy.deepcopy(sample)
    new_sample['line_matrix'] = line_onehot_matrix
    new_sample.pop('line_text', None)
    return new_sample

class LineTextToIntegerTransform(object):
  """Converts a string of line_text into a list of integers
    * Integers specify their index in the idx_to_char_map
  """

  def __init__(self, char_to_idx_map):
    self.char_to_idx_map = char_to_idx_map

  def __call__(self, sample):
    line_text = sample['line_text']
    line_text_integers = [None] * len(line_text)
    for i, c in enumerate(line_text):
      line_text_integers[i] = self.char_to_idx_map[c]
    
    new_sample = copy.deepcopy(sample)
    new_sample['line_text_integers'] = torch.tensor(line_text_integers, dtype=torch.long)
    new_sample.pop('line_text', None)
    return new_sample

class PadDatapointsTransform(object):
  """Pads the list of datapoints (x, y, p) to max_line_points
  """

  def __init__(self, max_line_points):
    self.max_line_points = max_line_points

  def __call__(self, sample):
    datapoints = sample['datapoints']
    if not torch.is_tensor(datapoints):
      datapoints = torch.tensor(datapoints, dtype=torch.float)

    assert (self.max_line_points - len(datapoints)) >= 0, f'max_line_points ({self.max_line_points}) must be larger or equal to length of datapoints ({len(datapoints)})'

    padding = torch.tensor([(0, 0, 0)] * (self.max_line_points - len(datapoints)), dtype=torch.float)
    padded_datapoints = torch.cat((datapoints, padding), dim=0)
    
    new_sample = copy.deepcopy(sample)
    new_sample['datapoints'] = padded_datapoints
    return new_sample

class PadLineTextTransform(object):
  """Pads the string of line_text into a string of length max_line_text_length
    * The character used for padding is specified by pad_character
  """

  def __init__(self, pad_character, max_line_text_length):
    self.pad_character = pad_character
    self.max_line_text_length = max_line_text_length

  def __call__(self, sample):
    line_text = sample['line_text']

    assert (self.max_line_text_length - len(line_text)) >= 0, f'max_line_text_length ({self.max_line_text_length}) must be larger or equal to line_text_length ({len(line_text)}) of datapoints'
    
    new_line_text = line_text + self.pad_character * (self.max_line_text_length - len(line_text))
    
    new_sample = copy.deepcopy(sample)
    new_sample['line_text'] = new_line_text
    return new_sample