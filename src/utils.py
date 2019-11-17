import torch, os, matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict
import constants

class AverageMeter(object):
    """Computes and stores the average and current value"""
    #taken from https://github.com/pytorch/examples/blob/master/imagenet/main.py
    def __init__(self, name, fmt=':6.3f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    #taken from https://github.com/pytorch/examples/blob/master/imagenet/main.py
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, epoch):
        entries = [self.prefix + ' ' + str(epoch) + ':' + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

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
def _get_char_set_to_ignore(char_freq_dict, minimum_char_frequency):
    chars_to_ignore = [char for char in char_freq_dict if char_freq_dict[char] < minimum_char_frequency]
    return set(chars_to_ignore)

"""
Returns:
    chars_to_ignore: Set of characters whose data should be ignored
    idx_to_char_map: Mapping from index to character for valid (non-ignored) characters
    char_to_idx_map: Mapping from character to index for valid (non-ignored) characters
"""
def get_char_info_from_data(data_dir, max_line_points, minimum_char_frequency):
    char_freq_dict = _get_char_frequency_dict(data_dir, max_line_points)
    line_char_set = set(char_freq_dict.keys())
    # Ignore lines / datafiles whose text contains characters with freq < MINIMUM_CHAR_FREQUENCY
    chars_to_ignore = _get_char_set_to_ignore(char_freq_dict, minimum_char_frequency)
    # Get non-ignored characters present in data 
    # # Get non-ignored characters present in data 
    # Get non-ignored characters present in data 
    valid_chars = list(line_char_set - chars_to_ignore) + [constants.PAD_CHARACTER]
    valid_chars = sorted(valid_chars)
    # Get mapping from character to index for one-hot transforms
    char_to_idx_map, idx_to_char_map = {}, {}
    for i, char in enumerate(valid_chars): 
        char_to_idx_map[char] = i
        idx_to_char_map[i] = char
    return chars_to_ignore, idx_to_char_map, char_to_idx_map

def delta_points_to_image(delta_points, output_dir, file_name):
    cur_point = (0, 0, 0)
    i = 0
    plot_x = []
    plot_y = []

    while i < len(delta_points):
        delta_x, delta_y, p = delta_points[i]
        x, y, _ = cur_point
        cur_point = (x + delta_x, y + delta_y, p)
        plot_x.append(cur_point[0])
        plot_y.append(1 - cur_point[1])
        if p == 1:
            # Plot current stroke and start new stroke
            plt.plot(plot_x, plot_y, 'k')
            plot_x = []
            plot_y = []
        i += 1
    # Plot any remaining points (especially for generated points)
    if len(plot_x) != 0:
        plt.plot(plot_x, plot_y, 'k')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.savefig(f'{output_dir}/{file_name}')
    plt.clf()

def delta_points_to_image_discrete(delta_points, output_dir, file_name, attn_weights=None, orig_text=None):
    cur_point = (0, 0, 0)
    i = 0
    plot_x = []
    plot_y = []
    #plt.xlim(-1, 1)
    #plt.ylim(-1, 1)

    while i < len(delta_points):
        delta_x, delta_y, p = delta_points[i]
        x, y, _ = cur_point
        cur_point = (x + delta_x, y + delta_y, p)
        plot_x.append(cur_point[0])
        plot_y.append(cur_point[1])
        if p == 1:
            # Plot current stroke and start new stroke
            plt.plot(plot_x, plot_y, 'ko')
            plot_x = []
            plot_y = []
        i += 1
    # Plot any remaining points (especially for generated points)
    if len(plot_x) != 0:
        plt.plot(plot_x, plot_y, 'ko')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(f'{output_dir}/{file_name}')
    plt.clf()

def points_to_image_discrete(delta_points, output_dir, file_name):
    cur_point = (0, 0, 0)
    i = 0
    plot_x = [x for x, _, _ in delta_points]
    plot_y = [y for _, y, _ in delta_points]
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.plot(plot_x, plot_y, 'ko')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(f'{output_dir}/{file_name}')
    plt.clf()
