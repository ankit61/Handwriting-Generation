import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from constants import STYLE_GEN_DATABASE_DIR, STYLE_GENERATOR_PIC_TYPE
import numpy as np
from skimage import io, transfrom

class StyleGeneratorDataset(Dataset):
    """
    * Expects data directory to have picture as "hand_written_sample_id.jpg" and
    ground truth style vector as "hand_written_sample_id.txt"
    * Style vector will have STYLE_VECTOR_SIZE lines where each line is a value
    """

    def __init__(self, data_dir = STYLE_GEN_DATABASE_DIR):
        self.data = self.load_data(data_dir)

    def load_data(self, data_dir):
        all_files = os.listdir(data_dir)
        txt_files = filter(lambda x: x[-4:] == '.txt', all_files)
        img_files = filter(lambda x: x[-4:] == STYLE_GENERATOR_PIC_TYPE, all_files)
        
        # Sort the files to be in corresponding order
        sorted(txt_files, key=lambda x: int(x.split('.')[0]))
        sorted(img_files, key=lambda x: int(x.split('.')[0]))
        
        # Assert correct order for txt_files and pic_files
        assert(len(txt_files) == len(img_files))
        for i in range(len(txt_files)):
            txt_file = txt_files[i]
            img_file = img_files[i]
            if (txt_file[-4:] != img_file[-4:]):
                raise AssertionError("Picture file does not correspond to Style Vector")

        # Create style vector for each text file
        style_vectors = map(self.read_lines_as_list, txt_files)

        # Create picture object for each picture file
        images = map(self.convert_images, img_files)

        return zip(style_vectors, images)

    def read_lines_as_list(self, file):
        with open(file, 'rt') as fp:
            return np.array([line.rstrip('\n') for line in file])
    
    def convert_images(self, file):
        img = io.imread(file)
        img = img.transpose((2,0,1))

        return torch.from_numpy(img)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.data[idx]
