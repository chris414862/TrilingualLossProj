# Author: Wei-Ning Hsu
import h5py
import json
import librosa
import numpy as np
import os
import scipy
import time
from pathlib import Path
from PIL import Image
from torchvision.transforms import transforms
from math import ceil
import pprint

from dataloaders.utils import WINDOWS, compute_spectrogram


def run(json_path, hdf5_json_path, audio_out_prefix, out_image_hdf5_path, audio_conf={}, overwrite:bool=False):
    with open(json_path, 'r') as f:
        data_and_dirs = json.load(f)
        data = data_and_dirs['data']
        hindi_audio_base = data_and_dirs['hindi_base_path']
        english_audio_base = data_and_dirs['english_base_path']
        japanese_audio_base = data_and_dirs['japanese_base_path']
        image_base = data_and_dirs['image_base_path']
    print('Loaded %d data from %s' % (len(data), json_path))

    #TODO: Uncomment this
    run_image(data, image_base, out_image_hdf5_path, overwrite=overwrite)
    print(f"Processing Hindi files stored at {hindi_audio_base}...")
    run_audio(data, hindi_audio_base, audio_out_prefix+"hindi.hdf5", audio_conf, key="hindi_wav", overwrite=overwrite)
    print(f"\nProcessing English files stored at {english_audio_base}...")
    run_audio(data, english_audio_base, audio_out_prefix + "english.hdf5", audio_conf, key="english_wav", overwrite=overwrite)
    print(f"\nProcessing Japanese files stored at {japanese_audio_base}...")
    run_audio(data, japanese_audio_base, audio_out_prefix + "japanese.hdf5", audio_conf, key="japanese_wav", overwrite=overwrite)
    
    Path(os.path.dirname(hdf5_json_path)).mkdir(parents=True, exist_ok=True)

    with open(hdf5_json_path, 'w') as f:
        d = {'image_hdf5_path': out_image_hdf5_path,
             'english_audio_hdf5_path': audio_out_prefix+"english.hdf5", 
             'hindi_audio_hdf5_path': audio_out_prefix+"hindi.hdf5", 
             'japanese_audio_hdf5_path': audio_out_prefix+"japanese.hdf5"
             }

        print("Metadata for hdf5 data:")
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(d)
        print(f"Saving to {hdf5_json_path}")
        json.dump(d, f)


# Solution borrows from https://github.com/h5py/h5py/issues/745
def run_image(data, image_base, image_path, overwrite:bool=False):
    #TODO: Uncomment this later
    if os.path.exists(image_path) and not overwrite:
        print('%s already exists. skip' % image_path)
        return

    print('Dumping image to HDF5 : %s' % image_path)
    n = len(data)
    Path(os.path.dirname(image_path)).mkdir(parents=True, exist_ok=True)
    f = h5py.File(image_path, 'w')
    dt = h5py.special_dtype(vlen=np.dtype('uint8'))
    dset_img = f.create_dataset('image', (n,), dtype=dt)
    
    start = time.time()
    for i, d in enumerate(data):
        with open('%s/%s' % (image_base, d['image']), 'rb') as f_img:
            binary_img = f_img.read()
        dset_img[i] = np.frombuffer(binary_img, dtype='uint8')

        if i % 100 == 0:
            t = time.time() - start
            print('processed %d / %d images (%.fs)' % (i, n, t))


def store_data():
    pass

def auto_detect_target_length(data, audio_base, audio_conf, key=None):
    # automatically determine target length
    cached_data = None
    n = len(data)
    if audio_conf.get('auto_length', None) is not None:
        data_prop_to_keep = audio_conf['auto_length']
        # record all lengths
        lengths = []
        cached_data = []
        print("automatically detecting target length.\n\tdata proportion to preserve:", data_prop_to_keep)
        orig_use_raw_length = audio_conf['use_raw_length']
        audio_conf['use_raw_length'] = True
        start = time.time()
        for i, d in enumerate(data):
            if (i+1) % 100 == 0:
                t = time.time() - start
                print('\tprocessed %d / %d audios (%.fs)\r' % (i+ 1, n, t), end="")
            # Load file and use default sample rate (which is 22050)
            y, sr = librosa.load(str(Path(audio_base)/ d[key]), None)
            cached_data.append((y,sr))
            logspec, n_frames = compute_spectrogram(y, sr, audio_conf)
            lengths.append(logspec.shape[1])

        print()
        lengths = np.array(lengths)
        sorted_args = np.argsort(lengths)
        cutoff_idx = int(ceil(lengths.shape[0]*data_prop_to_keep))
        new_target_length = lengths[sorted_args[cutoff_idx]]
        print(f"\tNew target length: {new_target_length}")
        audio_conf['use_raw_length'] = orig_use_raw_length
        return new_target_length, cached_data

def run_audio(data, audio_base, hdf5_audio_path, audio_conf, key=None, overwrite:bool=False):
    if os.path.exists(hdf5_audio_path) and not overwrite:
        print('%s already exists. skip' % hdf5_audio_path)
        return
    DEFAULT_TARGET_LENGTH = 2048
    print('Dumping audio to HDF5: %s' % hdf5_audio_path)
    print('  audio_conf : %s' % audio_conf)

    audio_conf['num_mel_bins'] = audio_conf.get('num_mel_bins', 40)


    audio_conf['target_length'] = audio_conf.get('target_length', 2048)
    audio_conf['use_raw_length'] = audio_conf.get('use_raw_length', False)
    assert(not audio_conf['use_raw_length'])

    # Chris Crabtree added this next block
    audio_conf['auto_length'] = audio_conf.get('auto_length', .95)
    if audio_conf.get('auto_length', None) is not None:
        target_length, cached_data = auto_detect_target_length(data, audio_base, audio_conf, key=key)
    else:
        target_length = DEFAULT_TARGET_LENGTH
        cached_data = None

    audio_conf['target_length'] = target_length
   
    # dump audio
    n = len(data)
    Path(os.path.dirname(hdf5_audio_path)).mkdir(parents=True, exist_ok=True)
    f = h5py.File(hdf5_audio_path, 'w')
    dset_mel_shape = (n, audio_conf['num_mel_bins'],
                      audio_conf['target_length'])
    dset_mel = f.create_dataset('melspec', dset_mel_shape, dtype='f')
    dset_len = f.create_dataset('melspec_len', (n,), dtype='i8')


    start = time.time()
    trunc_cnt = 0
    for i, d in enumerate(data):
        if audio_conf.get('auto_length', None) is None:
            # Load file and use default sample rate (which is 22050)
            y, sr = librosa.load(str(Path(audio_base)/ d[key]), None)
        else:
            assert not cached_data is None
            # Only cached the librosa output bc wasn't sure if anything done in 
            # compute_spectrogram was dependent on the number of frames
            y, sr = cached_data[i]

        logspec, n_frames, trunc_cnt = compute_spectrogram(y, sr, audio_conf, trunc_cnt=trunc_cnt)
        dset_mel[i, :, :] = logspec
        dset_len[i] = n_frames

        if i % 100 == 0:
            t = time.time() - start
            print('processed %d / %d audios (%.fs)' % (i, n, t))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('inp_json_path', type=str, help='input JSON file')
    parser.add_argument('out_json_path', type=str, help='path to save output json')
    parser.add_argument('audio_h5_prefix', type=str, help='prefix to save the HDF5 audio file for each language')
    parser.add_argument('out_image_h5_path', type=str, help='path to save HDF5 image file')
    parser.add_argument('--overwrite', action="store_true", help='flag to overwrite files if they exist.')
    args = parser.parse_args()
    print(args)

    run(args.inp_json_path, args.out_json_path,
        args.audio_h5_prefix, args.out_image_h5_path, overwrite=args.overwrite)
