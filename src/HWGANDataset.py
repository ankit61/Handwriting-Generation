import os, threading, copy
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from collections import defaultdict
import numpy as np
from utils import get_char_info_from_data
import constants

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

        sample['datapoints'] = torch.tensor(delta_datapoints[1:], dtype=torch.float)
        return sample

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
        
        sample['line_text_integers'] = torch.tensor(line_text_integers, dtype=torch.long)
        sample.pop('line_text', None)
        return sample

class NormalizeDatapointsTransform(object):
    """Applies Max-Min Normalization to the list of datapoints (x, y, p) to reduce them to (0, 1)"""

    def __init__(self, max_line_points= constants.MAX_LINE_POINTS):
        self.max_line_points = max_line_points

    def __call__(self, sample):
        datapoints = sample['datapoints']
        if not torch.is_tensor(datapoints):
            datapoints = torch.tensor(datapoints, dtype=torch.float)

        assert (self.max_line_points - len(datapoints)) >= 0, f'max_line_points ({self.max_line_points}) must be larger or equal to length of datapoints ({len(datapoints)})'


        x_vals, y_vals = datapoints[:, 0], datapoints[:, 1]
        min_x, max_x, min_y, max_y = min(x_vals), max(x_vals), min(y_vals), max(y_vals)
        datapoints = [((x-min_x)/(max_x-min_x), (y-min_y)/(max_y-min_y), p) for x, y, p in datapoints]

        sample['datapoints'] = torch.tensor(datapoints, dtype=torch.float)
        return sample

class PadDatapointsTransform(object):
    """Pads the list of datapoints (x, y, p) to max_line_points constants."""

    def __init__(self, max_line_points= constants.MAX_LINE_POINTS):
        self.max_line_points = max_line_points

    def __call__(self, sample):
        datapoints = sample['datapoints']
        if not torch.is_tensor(datapoints):
            datapoints = torch.tensor(datapoints, dtype=torch.float)

        assert (self.max_line_points - len(datapoints)) >= 0, f'max_line_points ({self.max_line_points}) must be larger or equal to length of datapoints ({len(datapoints)})'

        padding = torch.tensor([(0, 0, 0)] * (self.max_line_points - len(datapoints)), dtype=torch.float)
        sample['datapoints'] = torch.cat((datapoints, padding), dim=0)
        return sample

class PadLineTextTransform(object):
    """Pads the string of line_text into a string of length max_line_text_length
        * The character used for padding is specified by pad_character
    """

    def __init__(self, pad_character = constants.PAD_CHARACTER, 
        max_line_text_length = constants.MAX_LINE_TEXT_LENGTH):
        self.pad_character = pad_character
        self.max_line_text_length = max_line_text_length

    def __call__(self, sample):
        line_text = sample['line_text']

        assert (self.max_line_text_length - len(line_text)) >= 0, f'max_line_text_length ({self.max_line_text_length}) must be larger or equal to line_text_length ({len(line_text)}) of datapoints'
        
        sample['line_text'] = line_text + self.pad_character * (self.max_line_text_length - len(line_text))
        return sample

class HWGANDataset(Dataset):
    """
    * Expects data directory to have files names "line-id_writer-id.txt"
    * Data files should have line_text on line 1, num_data_points on line 2 and 
        x, y, p on each line thereafter
    """
    
    pos_weight = None

    def __init__(self, data_dir = constants.DATA_BASE_DIR, max_line_points=constants.MAX_LINE_POINTS):
        # Get characters to ignore and char-and-index mappings based on data
        self.chars_to_ignore, self.idx_to_char_map, self.char_to_idx_map = \
            get_char_info_from_data(data_dir, max_line_points, constants.MINIMUM_CHAR_FREQUENCY)

        assert len(self.char_to_idx_map) <= constants.CHARACTER_SET_SIZE, \
            f'''total characters ({len(self.char_to_idx_map)}) need to be smaller than 
                constants.CHARACTER_SET_SIZE ({constants.CHARACTER_SET_SIZE})'''

        self.transforms = transforms.Compose([
            #NormalizeDatapointsTransform(),
            CoordinatesToDeltaTransform(),
            PadLineTextTransform(),
            LineTextToIntegerTransform(self.char_to_idx_map),
            PadDatapointsTransform(),
        ])

        self.data = self.load_data(data_dir, max_line_points)

        pos, neg = 0, 0 
        for d in self.data:
            for point in d['datapoints']:
                pos += point[2].item()
                neg += 1 - point[2].item()
        pos_weight = neg / pos

    def __getattribute__(self, name):
        if name == 'pos_weight' and pos_weight is None:
            raise NotImplementedError('pos_weight is not set, create an object' +
                                        'of this class somewhere in the process' +  
                                        'to initialize it')

        return Dataset.__getattribute__(self, name)
    
    def load_data(self, data_dir, max_line_points):
        data = []
        writer_id_to_int_map = {}
        data_load_threads = []
        files_to_load = list(os.listdir(data_dir))
        
        for _ in range(constants.MAX_DATA_LOAD_THREADS):
            # Appends sample directly to data
            data_load_thread = threading.Thread(target=self.parallel_data_load, 
                    args=(files_to_load, data_dir, writer_id_to_int_map, max_line_points, data))
            data_load_thread.start()
            data_load_threads.append(data_load_thread)
        
        # Wait for transform threads to finish
        for thread in data_load_threads:
            thread.join()

        data.sort(key = lambda sample : sample['orig_datapoints_len'], 
            reverse=True)

        if constants.STANDARDIZE_POINTS : data = self.standardize_datapoints(data)

        return data

    def parallel_data_load(self, file_name_buf, data_dir, writer_id_to_int_map, max_line_points, data_buf):
        while len(file_name_buf) > 0:
            file_name = file_name_buf.pop()
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
                datapoints[i] = [float(split_line[0]), float(split_line[1]), float(split_line[2])] # (x, y, p)

            datapoints = torch.tensor(datapoints, dtype=torch.float)

            writer_id = file_name.split('.')[0].split('_')[1]
            if writer_id not in writer_id_to_int_map:
                writer_id_to_int_map[writer_id] = len(writer_id_to_int_map)
            writer_id = writer_id_to_int_map[writer_id]
            assert writer_id < constants.NUM_WRITERS, 'writer_id > constants.NUM_WRITERS'

            sample = {
                'writer_id': writer_id,
                'line_text': line_text,
                'orig_line_text_len': len(line_text),
                'datapoints': datapoints,
                'orig_datapoints_len': num_points
            }

            sample = self.transforms(sample)
            data_buf.append(sample)

    def standardize_datapoints(self, data):
        delta_xs, delta_ys = [], []
        for sample in data:
            for x, y, _ in sample['datapoints']:
                delta_xs.append(x)
                delta_ys.append(y)
        mean_x, std_x = np.mean(delta_xs), np.std(delta_xs)
        mean_y, std_y = np.mean(delta_ys), np.std(delta_ys)

        for sample in data:
            datapoints = np.array(sample['datapoints'])
            datapoints[:, 0] = (datapoints[:, 0] - mean_x) / std_x
            datapoints[:, 1] = (datapoints[:, 1] - mean_y) / std_y
            sample['datapoints'] = datapoints

        return data

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
