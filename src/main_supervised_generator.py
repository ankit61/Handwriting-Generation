from SupervisedGeneratorRunner import SupervisedGeneratorRunner
from torch.utils.data import DataLoader
from HWGANDataset import HWGANDataset
import torch
import constants

#torch.set_default_tensor_type(torch.cuda.FloatTensor 
#        if torch.cuda.is_available() else torch.FloatTensor)

gen_runner  = SupervisedGeneratorRunner()
data_loader = DataLoader(HWGANDataset(), batch_size=constants.GEN_BATCH_SIZE)

print('Starting to train...')
gen_runner.train(data_loader, 100)
