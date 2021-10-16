# Author: David Harwath, Wei-Ning Hsu
import argparse
import numpy as np
import os
import pickle
import shutil
import sys
import time
import torch
import warnings
from collections import OrderedDict

import dataloaders
from steps.traintest import train, validate
from run_utils import str2bool, set_seeds, create_audio_model, create_image_model, load_state_dict
from run_display_utils import my_model_summary


def load_args(old_args, exp_dir):
    """
    If resuming, load args.pkl from the experiment directory, and overwrite
    `data_train`/`data_val`/`resume`.
    """
    print('loading arguments from %s/args.pkl' % exp_dir)
    with open('%s/args.pkl' % exp_dir, 'rb') as f:
        tmp_args = pickle.load(f)
    for k in vars(old_args):
        if hasattr(tmp_args, k):
            setattr(old_args, k, getattr(tmp_args, k))
        else:
            print('...missing arg: %s' % k)
    return old_args


def load_dataloaders(data_train, data_val, batch_size, num_workers, args: argparse.Namespace):
    train_dset = dataloaders.ImageCaptionDatasetHDF5(data_train)
    
    if args.dev_eval is not None:
        dev_set_size = args.dev_eval
        train_dset, val_dset = train_dset.create_dev_set(dev_set_size, dev_set_confs={'image':{'center_crop':True}})

    else: # Use test set
        val_dset = dataloaders.ImageCaptionDatasetHDF5(
                data_val, image_conf={'center_crop':True})
        
    train_loader = torch.utils.data.dataloader.DataLoader(
            train_dset, batch_size=batch_size, shuffle=True, 
            num_workers=num_workers, pin_memory=False)#True)
    val_loader = torch.utils.data.dataloader.DataLoader(
            val_dset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=False)#True) Chris Crabtree: pin_memory=True caused annoying warnings and I did not see a slow down when =False

    return train_loader, val_loader, train_dset, val_dset


def load_state_dicts(audio_models, image_model, seed_dir, seed_epoch):
    audio_states = dict()
    if seed_epoch > 0:
        for lang_id in audio_models.keys():
            audio_states[lang_id] = torch.load(
                    '%s/models/%s_audio_model.e%d.pth' % (seed_dir, lang_id,seed_epoch))
        image_states = torch.load(
                '%s/models/image_model.e%d.pth' % (seed_dir, seed_epoch))
    else:
        for lang_id in audio_models.keys():
            audio_states = torch.load(
                    '%s/models/best_%s_audio_model.pth' % (seed_dir, lang_id,seed_epoch))
        # audio_states = torch.load(
        #         '%s/models/best_audio_model.pth' % seed_dir)
        image_states = torch.load(
                '%s/models/best_image_model.pth' % seed_dir)

    for lang_id in audio_models.keys():
        load_state_dict(audio_models[lang_id], audio_states[lang_id])
    load_state_dict(image_model, image_states)
    print('loaded parameters from %s/models/' % seed_dir)


def get_default_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # ResDavenet args
    parser.add_argument('--audio-model', type=str, default='ResDavenet', 
            choices=['ResDavenetVQ', 'ResDavenet'], help='audio model architecture')
    parser.add_argument('--image-model', type=str, default='Resnet50', 
            choices=['Resnet50'], help='image model architecture')
    parser.add_argument('--freeze-image-model', type=str2bool, default=False,
            help='Freeze image model parameters.')
    parser.add_argument('--audio-feature-dim', type=int, default=40, 
            help='Number of raw spectrogram features embedding dimension')
    parser.add_argument('--edim', type=int, default=1024, 
            help='Shared embedding dimension')
    parser.add_argument('--enorm', type=str, default='none',
            choices=['none', 'norm', 'clip'])
    parser.add_argument('--eproj', type=str, default='none',
            choices=['none', 'linear', 'relu', 'tanh'])         
    parser.add_argument('--pretrained-image-model', type=str2bool, default=True, 
            help='Use an image network pretrained on ImageNet')
    parser.add_argument('--seed-dir', type=str, default='',
            help=('Load image and audio model weights from a seed model. Overrides' 
                  ' using an image model pretrained on ImageNet'))
    parser.add_argument('--seed-epoch', type=int, default=-1, 
            help='Load snapshot from this epoch')
    parser.add_argument('--margin', type=float, default=1.0, 
            help='Margin paramater for margin losses (triplet and masked margin sm)')
    parser.add_argument('--layer-widths', type=str, default='128,256,256,512,1024', 
            help='ResDavenet layer/block sizes')
    parser.add_argument('--layer-depths', type=str, default='2,2,2,2', 
            help='ResDavenet depth of each residual block')
    parser.add_argument('--convsize', type=int, default=9,
            help='ResDavenet 1-D convolution width')
    parser.add_argument('--seed', type=int, default=8675309, help='Random seed')
    
    # VQ args
    # parser.add_argument('--VQ-commitment-cost', default=1, type=float, 
    #         help='VQ commitment cost')
    # parser.add_argument('--VQ-turnon', type=str, default='0,0,0,0,0', 
    #         help=('Comma-separated list of integers representing which VQ layers' 
    #               ' are turned on.'))
    # parser.add_argument('--VQ-sizes', type=str, default='1024,1024,1024,1024,1024', 
    #         help=('Comma-separated list of integers representing the codebook sizes' 
    #               ' for the quantization layers.'))
    # parser.add_argument('--nonneg-init', type=str2bool, default=False,
    #         help='Clamp the initial VQ codebooks at 0')
    # parser.add_argument('--init-std', default=1, type=float, 
    #         help='VQ codebook initialization standard deviation')
    # parser.add_argument('--init-ema-mass', default=1, type=float,
    #         help='EMA mass for the initial codebook')
    # parser.add_argument('--jitter', type=float, default=0.12, 
    #         help='Temporal jitter probability (equal for both left and right)')
    parser.add_argument('--image-output-head', default="avg", 
            choices=['avg', 'mh_attn', 'custom_self_attn'],
            help='Head layer to use to get single vector representation of the image model. '+
                 'Options: ["avg", "mh_attn", "custom_self_attn"]. '+
                 'Default: "avg"')
    parser.add_argument('--audio-output-head', default="avg", 
            choices=['avg', 'mh_attn', 'custom_self_attn'],
            help='Head layer to use to get single vector representation of the audio model. '+
                 'Options: ["avg", "mh_attn", "custom_self_attn"]. '+
                 'Default: "avg"')
    parser.add_argument('--no-scale-pe', action="store_true",
            help="Don't scale (by sqrt(d_model))in the positional embeddings layer")
    parser.add_argument('--norm-outputs-in-loss', action="store_true",
            help="Normalize all model outputs before loss computation. Places all ouputs on the 'hypersphere'. "+
                 "If --loss=hypersheric is set, this will automatically be set to true.")



    return parser
    

def get_train_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # training and optimization args
    parser.add_argument('--optim', type=str, default='adam',
            help='training optimizer', choices=['sgd', 'adam'])
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', 
            help='mini-batch size')
    parser.add_argument('--lr', type=float, default=2e-4, metavar='LR', 
            help='initial learning rate')

    parser.add_argument('--lr-ramp', default="0.0", metavar='LRRAMP',
            help=('Ramp up learning rate. If int, will ramp over this many steps. '+
                  'If float between 0 and 1, will ramp over percent of total steps.' ))
    parser.add_argument('--lr-decay', type=int, default=50, metavar='LRDECAY',
            help=('Multiply the learning rate by lr-decay-multiplier every lr-decay'
                  ' number of training steps'))
    parser.add_argument('--lr-decay-multiplier', type=float, default=0.99,
            metavar='LRDECAYMULT',
            help='Multiply the learning rate by this factor every lr-decay epochs')

    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
            help='momentum')
    parser.add_argument('--weight-decay', type=float, default=5e-7, metavar='W', 
            help='weight decay')
    parser.add_argument('--force-start-epoch', type=int, default=0, 
            metavar='force_start_epoch', 
            help=('Start on this epoch number (for controlling the position in the'
                  ' learning rate schedule)'))
    parser.add_argument('--n-epochs', type=int, default=150,
            help='number of maximum training epochs')
    parser.add_argument('--n-print-steps', type=int, default=100,
            help='number of steps to print statistics')
    parser.add_argument('--save-every', type=int, default=10,
            help=('Keep a model checkpoint every this many epochs. Set to -1 to'
                  ' deactivate'))
    parser.add_argument('--dev-eval', type=int,
            help='Use dev set for evaluation, rather than test set. '+
            f'Must give integer to specify dev set size.')
    parser.add_argument('--loss', type=str, default='multiview_coding',
            choices=['triplet', 'triplet_w_hardneg','multiview_coding','hyperspheric', 'masked_margin_sm'],
            help='Loss function to use')
    parser.add_argument('--full-graph', action="store_true",
            help='Use every modality pair for contrastive loss (rather than using images as the anchor)')
    parser.add_argument('--validate-full-graph', action="store_true",
            help='Use every modality pair in the validation output')
    parser.add_argument('--use-custom-hsphere', action="store_true",
            help='Do not use the multiview coding framework with hyperspheric loss')
    parser.add_argument('--sim-measure', default="dot", choices=['cos', 'dot'],
            help='Similarity measure to use for loss function '+
                 'Options: ["cos", "dot"]. Default: "dot"')
    parser.add_argument('--temperature', type=float, default=1.0,
            help='Temperature parameter for InfoNCE loss')
    parser.add_argument('--hsphere-alpha', type=float, default=2.0,
            help='Alpha to use in hyperspheric loss. Default: 2.0')
    parser.add_argument('--hsphere-t', type=float, default=2.0,
            help='t to use in hyperspheric loss. Default: 2.0')
    parser.add_argument('--hsphere-uniform-weight', type=float, default=1.0,
            help='adjust weighting to use in hyperspheric uniform sub-loss. Default: 1.0')
    parser.add_argument('--hsphere-align-weight', type=float, default=1.0,
            help='adjust weighting to use in hyperspheric align sub-loss. Default: 1.0')

    return parser


def get_run_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # I/O args
    parser.add_argument('--data-train', type=str, default='',
            help='training data json')
    parser.add_argument('--data-val', type=str, default='',
            help='validation data json')
    parser.add_argument('--exp-dir', type=str, default='',
            help='directory to dump experiments')
    parser.add_argument('--langs', type=str, default='english',
            help='languages to use. Must be in ["english", "hindi", "japanese"] and separated by commas')
    parser.add_argument('--resume', type=str2bool, default=False,
            help='load from exp_dir if True')
    parser.add_argument('--mode', type=str, default='eval',
            choices=['train', 'eval'],
            help='Train the model; otherwise, perform validation')
    parser.add_argument('--num-workers', type=int, default=8,
            help='number of dataloading workers')
    # For model summary
    parser.add_argument('--print-summary', action='store_true', 
            help='print model summaries')
    parser.add_argument('--mock-train4mem-stats', action="store_true",
            help='Perform mock training loop to get memory usage stats.')
    parser.add_argument('--no-pbar', action='store_true', 
            help="Don't display progress bar")

    return parser


if __name__ == '__main__':
    print('I am process %s, running on %s: starting (%s)' % (
            os.getpid(), os.uname()[1], time.asctime()))
    
    parser = get_default_parser()
    parser = get_train_parser(parser)
    parser = get_run_parser(parser)
    
    args = parser.parse_args()
    set_seeds(args.seed)
    if args.loss == "hyperspheric":
        print("RUNSCRIPT: Hyperspheric loss detected. Setting '--norm_ouputs_in_loss' flag to True.", )
        args.norm_outputs_in_loss = True

    def get_and_del_attr(name):
        val = getattr(args, name)
        delattr(args, name)
        return val

    exp_dir = get_and_del_attr('exp_dir')
    resume = get_and_del_attr('resume')
    data_train = get_and_del_attr('data_train')
    data_val = get_and_del_attr('data_val')
    mode = get_and_del_attr('mode')
    if resume:
        args = load_args(args, exp_dir)

    for k in vars(args):
        print('%-40s : %s' % (k, getattr(args, k)), flush=True)
    

    train_loader, val_loader, train_dset, val_dset = load_dataloaders(
            data_train, data_val, args.batch_size, args.num_workers, args)


    lang_ids = [lang.strip().lower() for lang in args.langs.split(",")]
    audio_models = dict()
    for lang_id in lang_ids:
        audio_models[lang_id] = create_audio_model(
                args.audio_model, args.audio_feature_dim, 
                # args.VQ_sizes, 
                args.layer_widths, args.layer_depths,
                # args.VQ_turnon, 
                args.convsize, 
                # args.VQ_commitment_cost,args.jitter, args.init_ema_mass, args.init_std, args.nonneg_init,
                args.audio_output_head,
                args.no_scale_pe)
    image_model = create_image_model(args.image_model, args.pretrained_image_model, args.image_output_head, args.no_scale_pe)
    
    image_model_input, audio_model_input_dict = train_dset.__getitem__(0)
    audio_model_input_shape = audio_model_input_dict["english"]["lmspecs"].shape #this doesn't include batch dimension
    image_model_input_shape = image_model_input.shape 
    if args.print_summary: # prints info about each layer and expected memory requirements
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",category=UserWarning)

            audio_m = [am for am in audio_models.values()][0]
            my_model_summary(audio_m, audio_model_input_shape, batch_size=args.batch_size,
                             mock_train4_mem_stats=args.mock_train4mem_stats, model_name="Audio Model")
            my_model_summary(image_m, image_model_input_shape, batch_size=args.batch_size,
                             mock_train4_mem_stats=args.mock_train4mem_stats, model_name="Image Model")


    # Start Training
    if mode == 'train':
        if args.seed_dir:
            load_state_dicts(audio_models, image_model,
                             args.seed_dir, args.seed_epoch)
    
        if not resume:
            print('RUNSCRIPT: Creating experiment directory: %s' % exp_dir)
            os.makedirs('%s/models' % exp_dir, exist_ok=True)
            with open('%s/args.pkl' % exp_dir, 'wb') as f:
                pickle.dump(args, f)
    
        train(audio_models, image_model, train_loader, val_loader,
              args, exp_dir, resume)
    elif mode == 'eval':
        load_state_dicts(audio_models, image_model, exp_dir, -1)
        validate(audio_models, image_model, val_loader, args)
    else:
        raise ValueError('Unsupported mode %s' % mode)



