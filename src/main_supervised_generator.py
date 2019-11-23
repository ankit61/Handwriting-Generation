from SupervisedGeneratorRunner import SupervisedGeneratorRunner
from torch.utils.data import DataLoader
from HWGANDataset import HWGANDataset
import torch, matplotlib
import constants
matplotlib.use('Agg')

#torch.set_default_tensor_type(torch.cuda.FloatTensor 
#        if torch.cuda.is_available() else torch.FloatTensor)
dataset = HWGANDataset()
gen_runner  = SupervisedGeneratorRunner()
data_loader = DataLoader(dataset, batch_size=constants.GEN_BATCH_SIZE)

print('Starting to train...')
gen_runner.train(data_loader, constants.EPOCHS)
