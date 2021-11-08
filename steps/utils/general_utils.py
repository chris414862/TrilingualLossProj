# Author: David Harwath
import math
import numpy as np
import pickle
import random
import torch
import torch.nn as nn
import sys
from collections import defaultdict, OrderedDict

###################
#### Constants ####
###################
EPSILON=1e-15
MAX_GRAD = 1e10


#############################
#### Parameter Functions ####
#############################

def get_param_norm(optimizer):
    params = []
    for param_group in optimizer.param_groups:
        params.extend([t.flatten() for t in param_group["params"] if t is not None])
        
    flat_params = torch.cat(params, dim=-1)

    return flat_params.norm(p=2, dim=-1)


def get_trainable_params(optimizer):
    params = []
    for param_group in optimizer.param_groups:
        params.extend(param_group["params"])
    return params
        

#######################
### Gradient Funcs ####
#######################

def check_gradient(optimizer, epoch_step, args):
    grad_norm = collect_gradient_from_opt(optimizer, normalize=True)
    warning_size = 100
    if grad_norm > warning_size or grad_norm > args.clip_grad:
        print(f"TRAINER: WARNING: (epoch step {epoch_step+1}) Gradient norm has become very large({grad_norm:.3f})).", end=" ")
        # Clip Gradient
        if args.clip_grad < MAX_GRAD:
            print(f"Clipping to {args.clip_grad:.2}.")
            params = get_trainable_params(optimizer)
            torch.nn.utils.clip_grad_norm_(params, args.clip_grad)
        else:
            print()


def collect_gradient_from_models(audio_models, image_model, normalize=True):
    model_gradients = OrderedDict()
    # gradient.extend([t.grad.flatten() for t in param_group["params"] if t is not None and t.grad is not None])
    model_gradients["img"] = collect_gradient_from_iter(image_model.parameters(), normalize=normalize)
    for lang_id, lang_encoder in audio_models.items():
        gradient = collect_gradient_from_iter(lang_encoder.parameters(), normalize=normalize)
        # gradient.extend([t.grad.flatten() for t in param_group["params"] if t is not None and t.grad is not None])

        model_gradients[lang_id] = gradient

    return model_gradients

def collect_grad_norm(optimizer):
    gradient = collect_gradient(optimizer)
    grad_norm = gradient.norm(p=2, dim=-1)

    return grad_norms

def collect_gradient_from_iter(iterable, normalize=False):
    gradient = [t.grad.flatten() for t in iterable if t is not None and t.grad is not None]
    gradient = torch.cat(gradient, dim=-1)
    if normalize:
        gradient = gradient.norm(p=2, dim=-1)
    return gradient

def collect_gradient_from_opt(optimizer, normalize=False):
    gradient = []
    for param_group in optimizer.param_groups:
        gradient.append(collect_gradient_from_iter(param_group["params"], normalize=normalize))
    #[t.grad.flatten() for t in param_group["params"] if t is not None and t.grad is not None])

    if normalize:
        ret = torch.tensor(gradient).sum()
    else:
        ret = torch.cat(gradient, dim=-1)

    return ret


##############################################
#### Model Input/Output Calculation Funcs ####
##############################################


def compute_avg_views(model_outputs:iter):
    avg_tens = None
    for tens in model_outputs:
        avg_tens = tens if avg_tens is None else avg_tens + tens
    avg_tens = avg_tens/len(model_outputs)
    return avg_tens

def get_model_outputs(image_model, image_input, audio_models:dict, target_audio_input:dict, args):
    model_outputs = dict() # save outputs for explicit deletion later
    image_output = image_model(image_input)
    # image_output dims: [batch, embed_dim]
    
    model_outputs['img'] = image_output

    # Get output for each language
    lang_ids = [k for k in target_audio_input.keys()]
    for lang_id in lang_ids:
        audio_input = target_audio_input[lang_id]['lmspecs'], target_audio_input[lang_id]['nframes']
        audio_output = audio_models[lang_id](*audio_input, view_id=lang_id)
        # audio dims: [batch, embed_dim]
        model_outputs[lang_id] = audio_output


    view_ids = ["img"] + lang_ids
    if args.use_avg_anchor or args.use_avg_others_contrast:
        assert not (args.use_avg_anchor and args.use_avg_others_contrast), "Should only set either --use-avg-anchor or --use-avg-others-contrast, not both"
        model_outputs["avg"] = compute_avg_views(model_outputs.values())


    if args.norm_outputs_in_loss:
        for k in model_outputs.keys():
            # EPSILON is to prevent division by zero. Located in utils.py
            model_outputs[k] = model_outputs[k]/(model_outputs[k].norm(p=2, dim=-1, keepdim=True) + EPSILON)

    return model_outputs, view_ids


def get_target_multiling_data(full_audio_input, device, args):
    """
    Removes languages that will not be used in training

    Returns:
        model_outputs - (dict):
            Dictionary containing tensors to be input into audio model(s). 
            Structure:
            {
                lang_id1:{
                    'lmspecs': torch.Tensor  #shape: [batch, max_seq_len, spectrogram_dim] 
                    'nframes': torch.Tensor  #shape: [batch], Contains length of un-padded data
                },
                lang_id2:{
                ....
            }
    """
    assert args.langs is not None
    langs = [lang.strip().lower() for lang in args.langs.split(",")]
    for lang in langs:
        assert lang in ["english", "hindi", "japanese"]

    target_audio_input = defaultdict(dict)
    for lang in langs:
        lmspecs = full_audio_input[lang]['lmspecs'].to(device).type(torch.float32)
        # lmspecs dims: [batch, raw_audio_dim, time_steps]
        nframes = full_audio_input[lang]['nframes'].to(device).type(torch.float32)
        # nframes dims: [batch, ]

        target_audio_input[lang].update({'lmspecs':lmspecs, 'nframes':nframes})

    return target_audio_input

#######################################
#### Learning Rate Scheduler Funcs ####
#######################################


def get_lr_steps_from_str(lr, lr_ramp, total_steps):
    try:
        lr_ramp_steps = int(lr_ramp)
        return lr_ramp_steps
    except ValueError as e:
        pass

    try:
        lr_ramp_pct = float(lr_ramp)
        if 0.0 <= lr <= 1.0:
            return int(lr_ramp_pct*total_steps)

    except ValueError as e:
        raise ValueError("--lr-ramp must either be a positive integer or a float between .0 and 1.0.")


def adjust_learning_rate(base_lr, lr_ramp, lr_decay, lr_decay_multiplier, optimizer, global_step, total_steps):
    """Sets the learning rate to the initial LR decayed every lr_decay epochs"""
    lr_ramp_steps = get_lr_steps_from_str(base_lr, lr_ramp, total_steps)

    if global_step < lr_ramp_steps:
        lr = base_lr * (global_step / lr_ramp_steps)
    else:
        lr = base_lr * (lr_decay_multiplier ** ((global_step - lr_ramp_steps) // lr_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

####################
#### Misc Funcs ####
####################

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def free_mem(loss, aux_losses, model_outputs):
    # Free up VRAM memory explicitly
    del loss
    if aux_losses is not None:
        for al in aux_losses.values():
            if isinstance(al, dict):
                for sub_al in al.values():
                    del sub_al
            else:
                del al
    for output in model_outputs.values():
        del output




