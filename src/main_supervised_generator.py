from SupervisedGeneratorRunner import SupervisedGeneratorRunner
from torch.utils.data import DataLoader
from HWGANDataset import HWGANDataset
import torch

#torch.set_default_tensor_type(torch.cuda.FloatTensor 
#        if torch.cuda.is_available() else torch.FloatTensor)

gen_runner  = SupervisedGeneratorRunner()
data_loader = DataLoader(HWGANDataset(), batch_size=32)

print('Starting to train...')
gen_runner.train(data_loader, 1)
