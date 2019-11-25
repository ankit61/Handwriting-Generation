from SupervisedGeneratorRunner import SupervisedGeneratorRunner
from torch.utils.data import DataLoader
from HWGANDataset import HWGANDataset
import torch, matplotlib
import constants
matplotlib.use('Agg')
import os
import sys
#torch.set_default_tensor_type(torch.cuda.FloatTensor 
#        if torch.cuda.is_available() else torch.FloatTensor)

#get latest model
saved_models = os.listdir(constants.MODELS_BASE_DIR)
saved_models = [model for model in saved_models if model.split('_')[0] == 'GeneratorCell']
if len(saved_models) > 0:
    saved_models.sort(key= lambda m : int(m.split('_')[2].split('.')[0]), reverse=True)
    saved_models = saved_models[:1]
    saved_models[0] = os.path.join(constants.MODELS_BASE_DIR, saved_models[0])
else:
    saved_models = None

dataset = HWGANDataset()
dataset.get_data_statistics()
gen_runner  = SupervisedGeneratorRunner(load_paths=saved_models)
data_loader = DataLoader(dataset, batch_size=constants.GEN_BATCH_SIZE)

print('Starting to train...')
gen_runner.train(data_loader, constants.EPOCHS, validate_on_train=True)
