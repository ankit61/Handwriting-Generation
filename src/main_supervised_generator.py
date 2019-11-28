from SupervisedGeneratorRunner import SupervisedGeneratorRunner
from torch.utils.data import DataLoader
from HWGANDataset import HWGANDataset
import torch, matplotlib
import constants
matplotlib.use('Agg')
import os
import pickle

def get_model_code():
    try:
        with open(constants.MODEL_STR_2_ID_DICT_FILE, 'rb') as f:
            model_code_dict = pickle.load(f)
    except:
        model_code_dict = {}

    model_code = ''
    if constants.MODEL_STR in model_code_dict:
        model_code = model_code_dict[constants.MODEL_STR]
    else:
        model_code = str(len(model_code_dict))
        model_code_dict[constants.MODEL_STR] = model_code

        #save in file
        with open(constants.MODEL_STR_2_ID_DICT_FILE, 'wb') as f:
            pickle.dump(model_code_dict, f)

    return model_code

def get_saved_models(model_code):
    model_name = 'GeneratorCell' + model_code
    saved_models = os.listdir(constants.MODELS_BASE_DIR)
    saved_models = [model for model in saved_models if model.split('_')[0] == model_name]
    if len(saved_models) > 0:
        saved_models.sort(key= lambda m : int(m.split('_')[-1].split('.')[0]), reverse=True)
        saved_models = saved_models[:1]
        saved_models[0] = os.path.join(constants.MODELS_BASE_DIR, saved_models[0])
    else:
        saved_models = None
    
    return saved_models

def main():
    model_code = get_model_code()
    saved_models = get_saved_models(model_code)

    dataset = HWGANDataset()
    dataset.get_data_statistics()
    gen_runner  = SupervisedGeneratorRunner(load_paths=saved_models, model_code=model_code)
    data_loader = DataLoader(dataset, batch_size=constants.GEN_BATCH_SIZE)

    print('Starting to train...')
    gen_runner.train(data_loader, constants.EPOCHS, validate_on_train=True)

main()