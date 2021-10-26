import random as rnd
import numpy as np
from torch.utils.data import Dataset

class DummyDataset(Dataset):
    def __init__(self, image_dims=[3,220,220], audio_dims=[42, 101], size=10000):
        self.lang_ids = ["japanese", "english", "hindi"]
        self.image_dims = image_dims
        self.audio_dims = audio_dims
        self.size = size
        print(f"DUMMY_DATALOADER: ")

    def __getitem__(self,index):
        audio_data = {}
        for lang_id in self.lang_ids:
            # frames, spec_dims
            audio_data[lang_id] = {}
            audio_data[lang_id]["lmspecs"] = np.random.normal(size=self.audio_dims)
            audio_data[lang_id]["nframes"] = rnd.randint(1,1000)

        image_data = np.random.normal(size=self.image_dims)
        return image_data, audio_data

    def __len__(self):
        return self.size




