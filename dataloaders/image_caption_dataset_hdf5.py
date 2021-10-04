# Author: Wei-Ning Hsu
import h5py
import io
import json
import numpy as np
import os
import torch
import torch.nn.functional
import torchvision.transforms as transforms
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset
import pprint


def _process_resize_crop_norm_funcs(image_conf: dict):
    crop_size = image_conf.get('crop_size', 224)
    center_crop = image_conf.get('center_crop', False)
    if center_crop:
        ret_image_resize_and_crop = transforms.Compose(
            [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
    else:
        ret_image_resize_and_crop = transforms.Compose(
            [transforms.RandomResizedCrop(crop_size), transforms.ToTensor()])

    RGB_mean = image_conf.get('RGB_mean', [0.485, 0.456, 0.406])
    RGB_std = image_conf.get('RGB_std', [0.229, 0.224, 0.225])
    ret_image_normalize = transforms.Normalize(mean=RGB_mean, std=RGB_std)
    return ret_image_resize_and_crop, ret_image_normalize

def _check_num_samples(english_audios, hindi_audios, j_audios, images):
    assert english_audios['melspec'].shape[0] == images['image'].shape[0]
    assert hindi_audios['melspec'].shape[0]   == images['image'].shape[0]
    assert j_audios['melspec'].shape[0]       == images['image'].shape[0]


class ImageCaptionDatasetHDF5(Dataset):
    def __init__(self, json_path, audio_conf={}, image_conf={}):
        """
        Dataset that manages a set of paired images and audio recordings

        :param audio_hdf5 (str): path to audio hdf5
        :param image_hdf5 (str): path to audio hdf5
        """
        print(f"DATALOADER: loading metadata from {json_path}")
        with open(json_path, 'r') as f:
            data = json.load(f)
        self.english_audio_hdf5_path = data['english_audio_hdf5_path']
        self.hindi_audio_hdf5_path = data['hindi_audio_hdf5_path']
        self.j_audio_hdf5_path = data['japanese_audio_hdf5_path']
        self.image_hdf5_path = data['image_hdf5_path']

        print("DATALOADER: Using the following file paths:")
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(data)
        print("", flush=True)

        # delay creation until __getitem__ is first called
        self.images = None
        self.audios = None # This will eventually store all lang audios

        # audio features are pre-computed with default values.
        self.audio_conf = audio_conf
        self.image_conf = image_conf

        # load image/audio dataset size and check if number matches
        images = h5py.File(self.image_hdf5_path, 'r')
        # audios = h5py.File(self.english_audio_hdf5_path, 'r')

        english_audios = h5py.File(self.english_audio_hdf5_path, 'r')
        hindi_audios = h5py.File(self.hindi_audio_hdf5_path, 'r')
        j_audios = h5py.File(self.j_audio_hdf5_path, 'r')

        # Assure that the number of datapoints match
        _check_num_samples(english_audios, hindi_audios, j_audios, images)
        self.n_samples = english_audios['melspec'].shape[0]


        english_audios.close()
        hindi_audios.close()
        j_audios.close()
        images.close()

        self.image_resize_and_crop, self.image_normalize = _process_resize_crop_norm_funcs(
            image_conf)


    def _get_lang_audio(self, index, lang_key):
        audio = defaultdict(dict)
        audio[lang_key]['lmspecs'] = self.audios[lang_key]['melspec'][index]
        audio[lang_key]['nframes'] = self.audios[lang_key]['melspec_len'][index]
        if self.audio_conf.get('normalize', False):
                n_frames = audio[lang_key]['nframes']
                logspec = audio[lang_key]['lmspecs']
                mean = logspec[:, 0:n_frames].mean()
                std = logspec[:, 0:n_frames].std()
                logspec[:, 0:n_frames].add_(-mean)
                logspec[:, 0:n_frames].div_(std)
        return audio


    def _get_audio(self, index, offset=0):
        if self.audios is None:
            self._LoadAudio()

        audio = defaultdict(dict)
        for lang_id in ['english', 'hindi', 'japanese']:
            # Returns dict: {lang_id: {'lmspecs': np.ndarray[#logmel_feats, max_frame_length], 'nframes', np.ndarray[1]}}
            # nframes is just a scalar for each index
            lang_audio = self._get_lang_audio(index+offset,lang_id)
            audio.update(lang_audio)

        return audio


    def _LoadAudio(self):
        english_audios = h5py.File(self.english_audio_hdf5_path, 'r')
        hindi_audios = h5py.File(self.hindi_audio_hdf5_path, 'r')
        j_audios = h5py.File(self.j_audio_hdf5_path, 'r')
        self.audios = {
            "english"  : english_audios,
            "hindi"    : hindi_audios,
            "japanese" : j_audios
        }



    def _get_image(self, index, offset=0):
        if self.images is None:
            self._LoadImage()
        binary_img = self.images['image'][index+offset]
        img = Image.open(io.BytesIO(binary_img)).convert('RGB')
        img = self.image_resize_and_crop(img)
        img = self.image_normalize(img)
        return img


    def _LoadImage(self):
            self.images = h5py.File(self.image_hdf5_path, 'r')


    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) 
        nframes is an integer
        """
        audio = self._get_audio(index)
        image = self._get_image(index)
        return image, audio

    def __len__(self):
        return self.n_samples


    def _create_dev_set_arg_check(self, n, p, dev_set_confs):
        error_prefix_str = str(self.__class__)+": ARGUMENT ERROR: "
        assert (n is None or p is None) and (n is not None or p is not None), \
                error_prefix_str + "Only one of 'n' and 'p' can be given"

        assert p is None or (isinstance(p, float) and 0.0 < p < 1.0), \
                error_prefix_str + "'p' must be a float between 0.0 and 1.0 exclusive"

        assert n is None or (isinstance(n, int) and 0 < n < self.n_samples), \
                error_prefix_str + f"'n' must be an int greater than 0 and less than n_samples ({self.n_samples})"




    def create_dev_set(self, n:int=None, p:float=None, dev_set_confs = dict()):
        print("DATA_LOADER: Creating dev set")
        self._create_dev_set_arg_check(n, p, dev_set_confs)
        if n is not None:
            dev_n_samples = n
        elif p is not None:
            dev_n_samples = self.n_sample
        else:
            raise ValueError("Either 'n' or 'p' must be given")

        orig_train_samples = self.n_samples
        self.n_samples = self.n_samples - dev_n_samples
        print(f"DATA_LOADER: Dev set size {dev_n_samples}")
        print(f"DATA_LOADER: Train set size reduced from {orig_train_samples} to {self.n_samples}")
        dev_audio_conf = dev_set_confs.get("audio_conf", dict())
        dev_image_conf = dev_set_confs.get("image_conf", dict())

        self._LoadAudio()
        self._LoadImage()
        dev_set = DevImageCaptionDatasetHDF5(self.audios, self.images,
                                             dev_audio_conf, dev_image_conf, self.n_samples)
        return self, dev_set

class DevImageCaptionDatasetHDF5(ImageCaptionDatasetHDF5):
    def __init__(self, audios: dict, images: h5py.File,
                 audio_conf: dict, image_conf: dict, train_n_sample):
        self.audios = audios
        self.images = images

        # audio features are pre-computed with default values.
        self.audio_conf = audio_conf
        self.image_conf = image_conf

        # Assure that the number of datapoints match
        _check_num_samples(self.audios["english"], self.audios["hindi"], self.audios["japanese"], self.images)
        self.n_samples = audios["english"]['melspec'].shape[0] - train_n_sample
        self.train_n_sample = train_n_sample

        self.image_resize_and_crop, self.image_normalize = _process_resize_crop_norm_funcs(
            self.image_conf)

    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) 
        nframes is an integer
        """
        audio = self._get_audio(index, offset=self.train_n_sample)
        image = self._get_image(index, offset=self.train_n_sample)
        return image, audio
