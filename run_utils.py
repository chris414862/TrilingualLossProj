# Author: Wei-Ning Hsu
import numpy as np
import os
import pickle
import random
import torch

from models.AudioModels import  ResDavenet
#,ResDavenetVQ,
from models.ImageModels import Resnet50


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_audio_model(args, lang_ids):
        # audio_model_name, feat_dim, 
        #                # VQ_sizes, 
        #                layer_widths, layer_depths,
        #                # VQ_turnon, 
        #                convsize, 
        #                # VQ_commitment_cost, jitter, init_ema_mass, init_std, nonneg_init, 
        #                output_head, 
        #                mh_dropout,
        #                no_scale_pe,
        #                lang_embed_type,
        #                lang_ids,
        #                lang_embed_dim):
    layer_widths = [int(w) for w in args.layer_widths.split(',')]
    layer_depths = [int(w) for w in args.layer_depths.split(',')]

    # Load Models
    # if audio_model_name == 'ResDavenetVQ':
    #     vq_sizes = [int(s) for s in VQ_sizes.split(',')]
    #     vqs_enabled = [int(w) for w in VQ_turnon.split(',')]
    #     audio_model = ResDavenetVQ(layers=layer_depths,
    #                                layer_widths=layer_widths,
    #                                convsize=convsize,
    #                                codebook_Ks=vq_sizes,
    #                                commitment_cost=VQ_commitment_cost,
    #                                jitter_p=jitter,
    #                                vqs_enabled=vqs_enabled,
    #                                init_ema_mass=init_ema_mass,
    #                                init_std=init_std,
    #                                nonneg_init=nonneg_init,
    #                                output_head=output_head)
    if args.audio_model == 'ResDavenet':
        audio_model = ResDavenet(feat_dim=args.audio_feature_dim,
                                 layers=layer_depths,
                                 layer_widths=layer_widths,
                                 convsize=args.convsize,
                                 output_head=args.audio_output_head,
                                 mh_dropout=args.mh_dropout,
                                 scale_pe=not args.no_scale_pe,
                                 lang_embed_type=args.lang_embed_type,
                                 lang_ids=lang_ids,
                                 lang_embed_dim=args.lang_embed_dim,
                                 use_cls=not args.dont_use_cls,
                                 args=args)

    else:
        raise ValueError('Unknown audio model: %s' % audio_model_name)

    return audio_model


def create_image_model(args):
    #image_model_name, pretrained_image_model, output_head, mh_dropout, no_scale_pe, edim):
    #.image_model, args.pretrained_image_model, args.image_output_head, args.mh_dropout, args.no_scale_pe, args.edim)

    if args.image_model == 'Resnet50':
        image_model = Resnet50(pretrained=args.pretrained_image_model, output_head=args.image_output_head,
                                 mh_dropout=args.mh_dropout,
                                 scale_pe=not args.no_scale_pe, embedding_dim=args.edim,
                                 use_cls=not args.dont_use_cls,
                                 args=args)
    else:
        raise ValueError('Unknown image model: %s' % args.image_model)

    return image_model


def load_state_dict(model, states):
    """
    1) Take care of DataParallel/nn.Module state_dict
    2) Show keys that are not loaded due to size mismatch or not found in model
    """
    new_states = model.state_dict()
    loaded_keys = []
    for k, v in states.items():
        k = k[7:] if k.startswith('module') else k
        if k in new_states and new_states[k].size() == v.size():
            new_states[k] = v
            loaded_keys.append(k)
        else:
            print('Ignoring %s due to not existing or size mismatch' % k)

    non_loaded_keys = set(new_states.keys()) - set(loaded_keys)
    if non_loaded_keys:
        print('\nModel states that do not exist in the seed_dir:')
        for k in sorted(non_loaded_keys):
            print('  %s' % k)

    model.load_state_dict(new_states)


def load_audio_model_and_state(state_path='', arg_path='', exp_dir=''):
    '''
    Load model and state based on state_path (primary) or exp_dir
    '''
    if bool(state_path):
        if not bool(arg_path):
            exp_dir = os.path.dirname(os.path.dirname(state_path))
            arg_path = '%s/args.pkl' % exp_dir
    elif bool(exp_dir):
        state_path = '%s/models/best_audio_model.pth' % exp_dir
        arg_path = '%s/args.pkl' % exp_dir
    else:
        raise ValueError('Neither state_path or exp_dir is given')

    with open(arg_path, 'rb') as f:
        args = pickle.load(f)

    audio_model = create_audio_model(
            args.audio_model, args.VQ_sizes, args.layer_widths,
            args.layer_depths, args.VQ_turnon, args.convsize,
            args.VQ_commitment_cost, args.jitter, args.init_ema_mass,
            args.init_std, args.nonneg_init)

    if torch.cuda.is_available():
        audio_states = torch.load(state_path)
    else:
        audio_states = torch.load(state_path, map_location='cpu')
    load_state_dict(audio_model, audio_states)

    return audio_model
