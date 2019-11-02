from SupervisedGeneratorRunner import SupervisedGeneratorRunner
from torch.utils.data import DataLoader
from HWGANDataset import HWGANDataset

gen_runner  = SupervisedGeneratorRunner()
data_loader = DataLoader(HWGANDataset(), batch_size=32, pin_memory=True)

print('Starting to train...')
gen_runner.train(data_loader, 1)