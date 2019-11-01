from HWGANDataset import HWGANDataset
from torch.utils.data import DataLoader
import torch
import constants

dataset = HWGANDataset()
batch_size = 10
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

for batch in dataloader:
    break

for key in batch.keys():
    assert type(batch[key]) == type(torch.tensor([]))

print('datapoints: ', batch['datapoints'])
print('line_text_integers: ', batch['line_text_integers'])
print('writer_id: ', batch['writer_id'])
print('orig_datapoints_len: ', batch['orig_datapoints_len'])

assert batch['datapoints'].shape == \
    torch.Size([batch_size, constants.MAX_LINE_POINTS, 3])

assert batch['line_text_integers'].shape == torch.Size([batch_size, 
    constants.MAX_LINE_TEXT_LENGTH])

assert batch['writer_id'].shape == torch.Size([batch_size])

assert batch['orig_datapoints_len'].shape == torch.Size([batch_size])

print('All checks successful')