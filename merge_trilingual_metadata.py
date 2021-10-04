# Author: Wei-Ning Hsu
import h5py
import json
import librosa
import numpy as np
import os
import scipy
import time
import sys
import re
from collections import Counter

from pathlib import Path
from PIL import Image
from torchvision.transforms import transforms

from dataloaders.utils import WINDOWS, compute_spectrogram


"""
This script merges the metadata info for spoken captions 
from three languages: English, Hindi, and Japanese. The English and
Hindi metadata are combined in one file, while the Japanese metadata
is separate and has a different format. This script is highly tailored
to the datasets used by this project and not meant for general use.
"""


# Default values for arguments
INP_JP_TXT = "./data/PlacesAudioJpn_100k/PlacesAudioJpn_100k_image2audio_train.txt"
INP_HINDI_METADATA_DIR ="/home/harwath/data/hindi_places_100k/"
INP_ENGLISH_METADATA_DIR ="/home/harwath/data/PlacesAudio_400k_distro/metadata/"
IMAGE_BASE_PATH ="/home/harwath/data/PlacesAudio_400k_distro/images/"
OUT_JSON_PATH ="./data/TrilingualData/metadata/trilingual_train.json"

def get_dict_from_json_file(hindi_json_path):
    with open(hindi_json_path, 'r') as f:
        data_and_dirs = json.load(f)

    return data_and_dirs

def check_duplicate(dataset):
    seen = set()
    num_overlaping = 0
    for datum in dataset:
        if datum["image"] not in seen:
            seen.add(datum["image"])
        else:
            num_overlaping += 1
    return num_overlaping


def check_overlap(dataset1, dataset2):
    second_set = [datum["image"] for datum in dataset2]
    num_overlaping = 0
    for datum in dataset1:
        if datum["image"] in second_set:
            num_overlaping += 1
    return num_overlaping

def report_num_absent(num_absent, json_path):
    if num_absent > 0:
        print(f"\tWARNING: {num_absent} entries could not be found in the {json_path}", file=sys.stderr)
        
def report_duplicates(data, path:str="", num_duplicates:int=None):
    if num_duplicates is None:
        num_duplicates = check_duplicate(data)
    if num_duplicates > 0:
        print(f"\tWARNING: Entries with duplicate images contained WITHIN {path}", file=sys.stderr) 
        print(f"\t\t{num_duplicates} duplicates found out of a total of {len(data)}. Leaving them to be resolved later.", file=sys.stderr)

def report_overlap(dataset1, dataset2):
    num_overlaping = check_overlap(dataset1, dataset2)
    if num_overlaping > 0:
        print(f"\tWARNING: There are data entries with duplicate images now contained in the merged dataset.", file=sys.stderr)
        print(f"\t\t{num_overlaping} duplicates found out of a total of {len(dataset1)+len(dataset2)}.", end=" ", file=sys.stderr)
        print("Leaving them to be resolved later.", file=sys.stderr)

def report_consistency(base_path1, base_path2):
    if  base_path1 != base_path2:
        print("\tWARNING: Something is amiss with the metadata base paths", file=sys.stderr)
        print(f"\t\t{base_path2} has a key value does not match the rest.", file=sys.stderr)


def merge_metadata_splits(english_metadata_dir, split_json_regex:str=None, check_key_for_consistency:str= None):
    metadata_path_list = [path for path in os.listdir(english_metadata_dir) if re.search(split_json_regex, path) is not None]
    print("Splits found:", sorted(metadata_path_list))

    data_complete = None
    for split_path in metadata_path_list:
        data_and_dirs = get_dict_from_json_file(os.path.join(english_metadata_dir,split_path))
        print(f"\nProcessing: {split_path}\n\tcontains {len(data_and_dirs['data'])} entries")
        # Make sure all data entries are unique within dataset
        report_duplicates(data_and_dirs["data"], split_path)

        if data_complete is None:
            print("\tMerge beginning with:", split_path)
            data_complete = data_and_dirs
        else:
            if check_key_for_consistency is not None:
                report_consistency(data_and_dirs[check_key_for_consistency], data_complete.get(check_key_for_consistency, ""))
            print("\tMerging", split_path, "with previous")

            # Make sure all data entries are unique in the aggregated set
            report_overlap(data_complete["data"], data_and_dirs["data"])
            data_complete["data"].extend(data_and_dirs["data"])
            
        print("Current number of total entries:", len(data_complete["data"]))

    print("Total merged data entries:", len(data_complete["data"]))
    return data_complete, data_complete["data"]


def get_jp_metadata(jp_txt_path):
    # read Japanese txt file and create data dict
    with open(jp_txt_path) as f:
        lines = [line.strip() for line in f.readlines()]
    
    jp_metadata = dict()
    num_duplicates = 0
    for line in lines:
        img_path, wav_path = line.split()
        wav_path = wav_path.strip().lstrip("/")
        img_path = img_path.strip().lstrip("/")
        if img_path in jp_metadata.keys():
            num_duplicates += 1
        jp_metadata[img_path] = {"japanese_wav":wav_path}
    
    report_duplicates(jp_metadata, path=jp_txt_path, num_duplicates=num_duplicates)
    return jp_metadata


def check_present(img_path_query, hindi_img_paths, english_img_paths):
    hindi_img_absent, english_img_absent = 0, 0
    if img_path_query not in hindi_img_paths:
        hindi_img_absent += 1

    if img_path_query not in english_img_paths:
        english_img_absent += 1

    return hindi_img_absent, english_img_absent


def prepend_dict_keys(d, prepend_str, exceptions:list=[]):
    new_dict = dict()
    for key, val in d.items():
        if key in exceptions:
            new_dict[key] = val
            continue
        new_key = prepend_str+key
        new_dict[new_key] = val
    return new_dict


def my_inner_join(hindi_json_data, english_json_data, jp_metadata):
    '''
    'inner' merge info in hindi json w/ J data dict using image_path as key
    '''
    # create image_path->index mapping for hindi json
    hindi_img2idx = {datum["image"]:i  for i, datum in enumerate(hindi_json_data)}
    # create image_path->index mapping for english json
    english_img2idx = {datum["image"]:i  for i, datum in enumerate(english_json_data)}

    merge_info = Counter()
    new_data = list()
    for img_path in jp_metadata.keys():
        hindi_img_absent, english_img_absent = check_present(img_path, hindi_img2idx.keys(), english_img2idx.keys())
        merge_info['hindi_num_imgs_absent'] += hindi_img_absent 
        merge_info['english_num_imgs_absent'] += english_img_absent
        if hindi_img_absent + english_img_absent > 0:
            continue
        
        hindi_idx = hindi_img2idx[img_path]
        new_datum = hindi_json_data[hindi_idx]

        english_idx = english_img2idx[img_path]
        new_english_datum = prepend_dict_keys(english_json_data[english_idx], prepend_str="english_", exceptions=['image'])
        
        new_datum.update(new_english_datum) 
        new_datum.update(jp_metadata[img_path])

        new_data.append(new_datum) 
    return new_data, merge_info
        
def create_metadata_dict(english_metadata_dir, hindi_metadata_dir, jp_txt_path, image_base_path):

    # this is the only base path in the hindi data metadata file
    # it does not seem to correspond to the 'english_wav' keys,
    # but I use it just as a check in case some other file got
    # caught by the regex. I.e. if it's not there then there is 
    # some problem that occured.
    check_key_for_consistency = "english_base_path"
    print("-"*30)
    print("Merginging all Hindi metadata splits...\n")
    split_json_regex = r"hindi_english_100k[^./]*\.json$"
    hindi_data_and_dirs, hindi_json_data = merge_metadata_splits(hindi_metadata_dir, split_json_regex=split_json_regex,
                                                                 check_key_for_consistency=check_key_for_consistency)

    check_key_for_consistency = "audio_base_path"
    print("-"*30)
    print("Merginging all English metadata splits...\n")
    split_json_regex = r".*2020\.json$"
    english_data_and_dirs, english_json_data = merge_metadata_splits(english_metadata_dir, split_json_regex=split_json_regex, 
                                                                     check_key_for_consistency=check_key_for_consistency)
    print("-"*30)
    print(f"Loading Japanese metadata at {jp_txt_path}") 
    jp_metadata = get_jp_metadata(jp_txt_path)
    print("Data entries in Japanese metadata:", len(jp_metadata)) 
    
    print("-"*30)
    print(f"(Inner) Merging metadata files...")
    new_data, merge_info = my_inner_join(hindi_json_data, english_json_data, jp_metadata)
    report_num_absent(merge_info['hindi_num_imgs_absent'], hindi_metadata_dir)
    report_num_absent(merge_info['english_num_imgs_absent'], english_metadata_dir)
    report_duplicates(new_data, "result of inner join operation")
    print(f"{len(new_data)} entries successfully merged.")
    print("-"*30)

    new_metadata = {
                    "hindi_base_path": str((Path(hindi_metadata_dir)/ "hindi_wavs").resolve()),
                    "english_base_path":str(Path(english_metadata_dir).parent.resolve()),
                    "japanese_base_path":str(Path(jp_txt_path).parent.resolve()),
                    "image_base_path":str(Path(image_base_path).resolve()),
                }
    
    new_metadata['data'] = new_data
    return new_metadata
        

def run(english_metadata_dir, hindi_metadata_dir, jp_txt_path, image_base_path, json_path_out ):

    new_metadata_dict = create_metadata_dict(english_metadata_dir, hindi_metadata_dir, jp_txt_path, image_base_path)
    print("\nNew metadata info:")
    print("\tBase paths:")
    for key, val in new_metadata_dict.items():
        if key == "data":
            continue
        print(f"\t\t'{key}': {val}")

    print("\tNumber of entries:",len(new_metadata_dict['data']))
    print(f"Saving results to {json_path_out}\n")
    with open(json_path_out, 'w') as f:
       json.dump(new_metadata_dict, f, indent=4, ensure_ascii=False)




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp_hindi_metadata_dir', default=INP_HINDI_METADATA_DIR, type=str, help='directory containing input Hindi metadata files.')
    parser.add_argument('--inp_english_metadata_dir', default=INP_ENGLISH_METADATA_DIR, type=str, help='directory containing input English metadata files.')
    parser.add_argument('--inp_jp_txt', default=INP_JP_TXT, type=str, help='input Japanese txt metadata file')
    parser.add_argument('--image_base_path', default=IMAGE_BASE_PATH, type=str, help='path to directory containing images')
    parser.add_argument('--out_json_path', default=OUT_JSON_PATH, type=str, help='path to save output json')
    args = parser.parse_args()
    print(args)
    

    run(args.inp_english_metadata_dir, args.inp_hindi_metadata_dir, args.inp_jp_txt, args.image_base_path, args.out_json_path)


