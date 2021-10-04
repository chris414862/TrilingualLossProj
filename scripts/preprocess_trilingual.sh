#!/bin/bash 

# Author: Wei-Ning Hsu
# Updated by: Chris Crabtree, Summer 2021


# python merge_trilingual_metadata.py \
#     --inp_hindi_metadata_dir /home/harwath/data/hindi_places_100k/ \
#     --inp_english_metadata_dir /home/harwath/data/PlacesAudio_400k_distro/metadata/ \
#     --inp_jp_txt ./data/PlacesAudioJpn_100k/PlacesAudioJpn_100k_image2audio_train.txt \
#     --image_base_path /home/harwath/data/PlacesAudio_400k_distro/images/ \
#     --out_json_path ./data/TrilingualData/metadata/trilingual_train.json \
#
# python merge_trilingual_metadata.py \
#     --inp_hindi_metadata_dir /home/harwath/data/hindi_places_100k/ \
#     --inp_english_metadata_dir /home/harwath/data/PlacesAudio_400k_distro/metadata/ \
#     --inp_jp_txt ./data/PlacesAudioJpn_100k/PlacesAudioJpn_100k_image2audio_valid.txt \
#     --image_base_path /home/harwath/data/PlacesAudio_400k_distro/images/ \
#     --out_json_path ./data/TrilingualData/metadata/trilingual_valid.json \
#

python dump_hdf5_dataset.py \
  "./data/TrilingualData/metadata/trilingual_train.json" \
  "./data/TrilingualData/hdf5/metadata/trilingual_train_HDF5.json" \
  "./data/TrilingualData/hdf5/audio/trilingual_train_audio_" \
  "./data/TrilingualData/hdf5/images/trilingual_train_images.hdf5" \
  "--overwrite"

# python3 dump_hdf5_dataset.py \
#   "./data/TrilingualData/metadata/trilingual_valid.json" \
#   "./data/TrilingualData/hdf5/metadata/trilingual_valid_HDF5.json" \
#   "./data/TrilingualData/hdf5/audio/trilingual_valid_audio_" \
#   "./data/TrilingualData/hdf5/images/trilingual_valid_images.hdf5" \
#   "--overwrite"

