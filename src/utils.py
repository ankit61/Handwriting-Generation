import torch, os, matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict
import constants

PLOT_FLAG_MAP = {
    'continuous': '-',
    'discrete': 'o'
}
POINT_FLIP_THRESHOLD = 10000

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
            text = fp.readline().split('\n')[0]
            num_points = int(fp.readline().split('\n')[0])
            if num_points > max_line_points: continue
            for c in text: 
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

def attention_output(attn_weights, generated_points, original_text, delta_points=False, plot_type='continuous'):
    fig = plt.figure()
    plot_rows = 2
    plot_cols = 1

    heatmap_ax = fig.add_subplot(plot_rows, plot_cols, 1)
    sns.heatmap(attn_weights, ax=heatmap_ax, cbar=False)
    heatmap_ax.set_ylabel(original_text)

    generated_ax = fig.add_subplot(plot_rows, plot_cols, 2)
    plot_points(plt, generated_points, delta_points, plot_flags=f'k{PLOT_FLAG_MAP[plot_type]}')
    # Turn off the frame and axes-ticks for the generated handwriting plot
    generated_ax.spines['top'].set_visible(False)
    generated_ax.spines['right'].set_visible(False)
    generated_ax.spines['bottom'].set_visible(False)
    generated_ax.spines['left'].set_visible(False)
    generated_ax.get_xaxis().set_ticks([])
    generated_ax.get_yaxis().set_ticks([])

    return fig

def unstandardize_points(points, mean, std):
    points = points * np.array(std) + np.array(mean)
    return points

def points_to_image(generated_points, ground_truth_points=None, delta_points=False, plot_type='continuous', generated_plot_title='Generated Output',
        gt_plot_title='Ground Truth Output', save_to_file=False, file_path=None):
    fig = plt.figure()

    plot_rows = 1 if ground_truth_points is None else 2
    plot_cols = 1
    generated_ax = fig.add_subplot(plot_rows, plot_cols, 1)
    #generated_fig = plt.subplot(plot_rows, plot_cols, 1)
    generated_ax.set_title(generated_plot_title)
    plot_points(plt, generated_points, delta_points, plot_flags=f'k{PLOT_FLAG_MAP[plot_type]}')

    if not ground_truth_points is None:
        gt_ax = fig.add_subplot(plot_rows, plot_cols, 2)
        #gt_fig = plt.subplot(plot_rows, plot_cols, 2)
        gt_ax.set_title(gt_plot_title)
        plot_points(plt, ground_truth_points, delta_points, plot_flags=f'k{PLOT_FLAG_MAP[plot_type]}')

    if save_to_file:
        assert file_path != None, 'File path required when saving image to file'
        plt.savefig(file_path)
        plt.clf()
    
    return fig

def plot_points(fig, points, delta_points, plot_flags):
    cur_point = (0, 0, 0)
    i = 1
    plot_x = []
    plot_y = []
    while i < len(points):
        x, y, p = points[i]
        cur_point = (cur_point[0] + x, cur_point[1] + y, p) if delta_points else (x, y, p)
        plot_x.append(cur_point[0])
        plot_y.append(POINT_FLIP_THRESHOLD - cur_point[1]) # Vertically flip image to get understandable output
        if cur_point[2] == 1:
            # Plot current stroke and start new stroke
            fig.plot(plot_x, plot_y, plot_flags)
            plot_x = []
            plot_y = []
        i += 1
    if len(plot_x) != 0:
        fig.plot(plot_x, plot_y, plot_flags)