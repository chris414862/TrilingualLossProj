# Author: David Harwath, Wei-Ning Hsu

import datetime
import numpy as np
import pickle
import shutil
import time
import torch
import torch.nn as nn
import sys
import os
import re
import pprint
import pandas as pd
import signal

from models.quantizers import compute_perplexity
from .util import *
from .util2 import InfoNCE_loss
from math import ceil
from collections import defaultdict, Counter, OrderedDict


# Made this global so it could be used in a signal handler
# But pytorch doesn't pass along signal handlers to sub-processes in DataParallel (it seems)
CURR_NUM_STAT_LINES = 0


def flprint(*args, **kwargs):
    print(*args, flush=True, **kwargs)

def map_skip_none(fn, it):
    """
    emulate list(map(fn, it)) but leave None as it is. 
    """
    ret = []
    for x in it:
        if x is None:
            ret.append(None)
        else:
            ret.append(fn(x))
    return ret


def numbers_to_str(nums, precision=3):
    msg = '('
    num_tmp = '%%.%df' % precision
    num_to_str = lambda x: (str(x) if x is None else num_tmp % x)
    for num in nums[:-1]:
        msg += num_to_str(num)
        msg += ', '
    msg += num_to_str(nums[-1])
    msg += ')'
    return msg

def can_report_mem_usage():
    version_pieces = torch.__version__.split(".")
    if len(version_pieces) < 2:
        print("Unexpected version formatting from Pytorch. Disabling memory usage display in progress bar")
        return False
    try:
        major = int(version_pieces[0])
        minor = int(version_pieces[0])
    except ValueError as e:
        print("Unexpected version formatting from Pytorch. Disabling memory usage display in progress bar")
        return False

    if major >= 1 and minor >=4:
        return True
    else:
        return False


def pbar_update(i, updates_per_epoch, loss_meter, update_every=1, bar_parts=50, aux_losses=None, report_mem_usage=False, cur_lr=None, optimizer=None):
    if  i % update_every  == 0:
        global CURR_NUM_STAT_LINES
        # This is an ANSI CSI (Control Sequence Introducer) Sequences.
        # On Unix-like system's \x1B is ESC. "[J" clears from the cursor to the end of the screen
        print(f"\x1B[J",end="")
        cols = 100
        prefix_str = f"{(i+1):>7}/{updates_per_epoch} "
        stat_lines = [f"{prefix_str}| Ep.Loss avg: {loss_meter.avg:<9.3f} cur: {loss_meter.val:<9.3f} "]
        if cur_lr is not None:
            stat_lines[0] += f"| lr: {cur_lr:<.2e} "

        if optimizer is not None:
            gradient = collect_gradient(optimizer)
            grad_norm = gradient.norm(p=2, dim=-1)
            param_norm = get_param_norm(optimizer)
            stat_lines.append(" "*len(prefix_str)+f"| param norm: {param_norm:.3f}  grad_norm: {grad_norm:.3f}")

        if aux_losses is not None:
            for view_pair_key, loss_dict in aux_losses.items():
                for loss_type, loss_val in loss_dict.items():
                    curr_stat_line = stat_lines.pop(-1)
                    new_str = f"| {view_pair_key+'_'+loss_type+':':<12} {loss_val.item():8.3f} "
                    if len(curr_stat_line + new_str) > cols or loss_type.strip() == "total":
                        stat_lines.append(curr_stat_line)
                        stat_lines.append(" "*len(prefix_str)+new_str)
                    else:
                        stat_lines.append(curr_stat_line+new_str)

        if report_mem_usage:
            curr_stat_line = stat_lines.pop(-1)
            new_str = f" | mem: {memr:5.2f}gb"
            if len(curr_stat_line + new_str) > cols:
                stat_lines.append(curr_stat_line)
                stat_lines.append(" "*len(prefix_str)+new_str)
            else:
                stat_lines.append(curr_stat_line+new_str)

        for stat_line in stat_lines:
            print(stat_line)
        parts_done = int((i+1)/updates_per_epoch*bar_parts)
        parts_togo = int((updates_per_epoch-i-1)/updates_per_epoch*bar_parts)
        print(" "*len(prefix_str)+"|"+"-"*parts_done+">"+" "*parts_togo+"|", flush=True)

        # +1 for the status bar.
        CURR_NUM_STAT_LINES = len(stat_lines)+1

        # "[(number)A" moves cursor up (number) spaces.
        print(f"\x1B[{CURR_NUM_STAT_LINES}F",end="", flush=True )


def mid_epoch_training_report(epoch, batches_per_epoch, loss_meter,
                              epoch_loss_meter, i, batch_timer,
                              epoch_time_elapsed, tot_time, cur_lr, 
                              args):

    # \x1B is "ESCAPE" key, "[(number)E" moves cursor down (number) spaces.
    print(f"\x1B[{CURR_NUM_STAT_LINES}E", end="")
    # print(f"\x1B[J",end="")
    print('Epoch: [{0}][{1}/{2}]'
          '  Bat time={bt.val:.1f} ({bt.avg:.1f})'
          '  Ep time={et}s'
          '  Cur loss: {loss.val:.3f}'
          '  Tot loss avg: {loss.avg:.3f}'
          '  Avg loss for ep.: {epoch_loss.avg:.3f}'
          '  Cur lr: {cur_lr:.8}'
          '  Langs=({langs})'.format(
                       epoch, (i+1), batches_per_epoch,
                       bt         = batch_timer,
                       et         = int(epoch_time_elapsed),
                       loss       = loss_meter,
                       epoch_loss = epoch_loss_meter,
                       cur_lr     = cur_lr,
                       langs      = args.langs
                       ),
           flush = True)
    # recalls = validate(audio_model, image_model, test_loader, args)


def load_state(exp_dir, audio_models, image_model, device):
    (progress, epoch, global_step, best_epoch,
     best_acc) = load_progress("%s/progress.pkl" % exp_dir)
    print("\nResume training from:")
    print("  epoch = %s" % epoch)
    print("  global_step = %s" % global_step)
    print("  best_epoch = %s" % best_epoch)
    print("  best_acc = %.4f" % best_acc)
    if epoch != 0:
        # Models' state
        for lang_id in audio_models.keys():
            audio_models[lang_id].load_state_dict(
                    torch.load("%s/models/%s_audio_model.iter.pth" % (exp_dir, lang_id)))
        image_model.load_state_dict(
                torch.load("%s/models/image_model.iter.pth" % (exp_dir)))
        print("loaded parameters from epoch %d" % epoch)

        # Optimizer state
        optimizer.load_state_dict(
                torch.load("%s/models/optim_state.iter.pth" % (exp_dir)))
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print("loaded state dict from epoch %d" % epoch)
    return progress, epoch, global_step, best_epoch, best_acc


def prepare_models(image_model, audio_models, device, args):
    if not isinstance(image_model, torch.nn.DataParallel):
        image_model = nn.DataParallel(image_model)

    image_model = image_model.to(device)

    for lang_id in audio_models.keys():
        if not isinstance(audio_models[lang_id], torch.nn.DataParallel):
            audio_models[lang_id] = nn.DataParallel(audio_models[lang_id])
        audio_models[lang_id] = audio_models[lang_id].to(device)

    return image_model, audio_models


def setup_optimizer(image_model, audio_models, args):
    # Gather trainable parameters
    audio_trainables = list()
    for lang_id in audio_models.keys():
        audio_trainables.extend([p for p in audio_models[lang_id].parameters() if p.requires_grad])
    if args.freeze_image_model:
        image_trainables = [p for n, p in image_model.named_parameters() \
                            if n.startswith('embedder')]
    else:
        image_trainables = [p for p in image_model.parameters() if p.requires_grad]
    trainables = audio_trainables + image_trainables
    print('TRAINER: Total %d trainable parameters' % len(trainables))

    # Instantiate optimizer type
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(trainables, args.lr,
                                 momentum=args.momentum,
                                 weight_decay=args.weight_decay)
        print('TRAINER: Using %s optimizer w/ lr: %f, momentum: %d, weight_decay: %f' % (args.optim, args.lr, args.momentum, args.weight_decay))
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(trainables, args.lr,
                                weight_decay=args.weight_decay)
        print('TRAINER: Using %s optimizer w/ lr: %f, weight_decay: %f' % (args.optim, args.lr, args.weight_decay))
    else:
        raise ValueError('Optimizer %s is not supported' % args.optim)

    return optimizer, trainables

def report_initial_info(batch_size, tot_size, updates_per_epoch, args):
    print('TRAINER: Found %d GPUs' % torch.cuda.device_count())
    print(f"TRAINER: batch size: {batch_size}, dataset size: {tot_size}, updates per epoch: {updates_per_epoch}")
    print(f"TRAINER: Training with {args.langs} dataset only")
    print("TRAINER: Starting training...")


def report_epoch_info(global_step, epoch, cur_lr):
    print("TRAINER: Current #steps=%s, #epochs=%s" % (global_step, epoch))
    print('TRAINER: Learning rate @ %d is %.8f' % (epoch, cur_lr))


def save_state_and_progress(exp_dir, image_model, audio_models, optimizer, epoch, progress, is_best_acc:bool, args):

    # Save optimizer and models' state
    for lang_id in audio_models.keys():
        torch.save(audio_models[lang_id].state_dict(),
                "%s/models/%s_audio_model.iter.pth" % (exp_dir, lang_id))
    torch.save(image_model.state_dict(),
            "%s/models/image_model.iter.pth" % (exp_dir))
    torch.save(optimizer.state_dict(),
            "%s/models/optim_state.iter.pth" % (exp_dir))

    # Record models if best seen so far
    if is_best_acc:
        for lang_id in audio_models.keys():
            shutil.copyfile("%s/models/%s_audio_model.iter.pth" % (exp_dir, lang_id),
                "%s/models/best_%s_audio_model.pth" % (exp_dir, lang_id))
        shutil.copyfile("%s/models/image_model.iter.pth" % (exp_dir),
            "%s/models/best_image_model.pth" % (exp_dir))

    # Record models periodically according to args.save_every
    if args.save_every > 0 and epoch % args.save_every == 0:
        for lang_id in audio_models.keys():
            shutil.copyfile("%s/models/%s_audio_model.iter.pth" % (exp_dir, lang_id),
                            "%s/models/%s_audio_model.e%d.pth" % (exp_dir, lang_id, epoch))
        shutil.copyfile("%s/models/image_model.iter.pth" % (exp_dir),
            "%s/models/image_model.e%d.pth" % (exp_dir, epoch))

    progress.to_pickle("%s/progress.pkl" % exp_dir)

def get_target_multiling_data(full_audio_input, device, args):
    assert args.langs is not None
    langs = [lang.strip().lower() for lang in args.langs.split(",")]
    for lang in langs:
        assert lang in ["english", "hindi", "japanese"]

    target_audio_input = defaultdict(dict)
    for lang in langs:

        lmspecs = full_audio_input[lang]['lmspecs'].to(device)
        # lmspecs dims: [batch, raw_audio_dim, time_steps]
        nframes = full_audio_input[lang]['nframes'].to(device)

        target_audio_input[lang].update({'lmspecs':lmspecs, 'nframes':nframes})

    return target_audio_input



def get_model_outputs(image_model, image_input, audio_models:dict, target_audio_input:dict, args):
    model_outputs = dict() # save outputs for explicit deletion later
    image_output = image_model(image_input)
    # image_output dims: [batch, embed_dim]
    
    model_outputs['image'] = image_output

    # Get output for each language
    lang_ids = [k for k in target_audio_input.keys()]
    for lang_id in lang_ids:
        audio_input = target_audio_input[lang_id]['lmspecs'], target_audio_input[lang_id]['nframes']
        audio_output = audio_models[lang_id](*audio_input, view_id=lang_id)
        # audio dims: [batch, embed_dim]
        model_outputs[lang_id] = audio_output

    if args.norm_outputs_in_loss:
        for k in model_outputs.keys():
            model_outputs[k] = model_outputs[k]/(model_outputs[k].norm(p=2, dim=-1, keepdim=True) + EPSILON)

    return model_outputs, lang_ids


def compute_view_pair_loss(model_output1, model_output2, loss_func,  loss_record, view_pair_id, loss_weight=1.0, **extra_loss_kwargs):
    loss, aux_losses = loss_func(model_output1, model_output2, view_pair_id=view_pair_id, **extra_loss_kwargs)#, debug=True)
    loss = loss * loss_weight
    loss_record[view_pair_id] = OrderedDict()
    loss_record[view_pair_id]["total"] = loss
    if aux_losses is not None:
        assert isinstance(aux_losses, dict)
        loss_record[view_pair_id].update(aux_losses)
    return loss


def custom_hsphere_loss_computation(image_model, image_input, audio_models:dict, target_audio_input,  device, args, **kwargs):
    
    model_outputs, lang_ids = get_model_outputs(image_model, image_input, audio_models, target_audio_input, args)

    # compute audio-image pairs
    tot_loss = 0.0

    # record loss for display purposes
    loss_record = OrderedDict() #if len(lang_ids) >= 1 else None

    # get alignment loss
    loss, loss_record, _ = multiling_contrastive_computation(image_model, image_input, audio_models, target_audio_input,  device, args, 
                                      loss_func=hsphere_align_loss, loss_weight=args.hsphere_align_weight, model_outputs=model_outputs,
                                      lang_ids=lang_ids, view_pair_suffix="_al")
    # aligin weight applied above
    tot_loss = tot_loss + loss

    # get uniform losses
    img_loss, _ = hsphere_uniformity_loss(model_outputs['image'], t=args.hsphere_t, view_id="image")
    tot_loss = tot_loss + args.hsphere_uniform_weight*img_loss
    loss_record["img_u"] = {"total": args.hsphere_uniform_weight*img_loss}
    for i in range(len(lang_ids)):
        lang_loss, _ = hsphere_uniformity_loss(model_outputs[lang_ids[i]], t=args.hsphere_t, view_id=lang_ids[i])#, debug=True)
        lang_loss = args.hsphere_uniform_weight*lang_loss
        tot_loss = tot_loss + lang_loss
        loss_record[lang_ids[i][:3]+"_u"] = {"total": lang_loss}

    return tot_loss, loss_record, model_outputs



def multiling_contrastive_computation(image_model, image_input, audio_models:dict, target_audio_input:dict, device, args,
                                      loss_weight=1.0, loss_func=None, model_outputs=None, lang_ids=None, view_pair_suffix="", **extra_loss_kwargs):
    '''
       Logic for multilingual contrastive loss computation. We assume the image modality is the anchor
       unless the '--full-graph' argument is set
       Parameters:
       Return:
            batch_loss: torch.Tenser - loss for the batch. Size is []. I.e the loss is a scalar
    '''

    if model_outputs is None: 
        model_outputs, lang_ids = get_model_outputs(image_model, image_input, audio_models, target_audio_input, args)

    # compute audio-image pairs
    tot_loss = 0.0
    loss_record = OrderedDict() #if len(lang_ids) >= 1 else None
    for i in range(len(lang_ids)):
        view_pair_id = "img_"+lang_ids[i][:3]+view_pair_suffix
        loss = compute_view_pair_loss(model_outputs['image'], model_outputs[lang_ids[i]], loss_func, loss_record, view_pair_id,
                                      loss_weight=loss_weight, **extra_loss_kwargs)
        tot_loss = tot_loss + loss

    # Get language to language contrastive losses
    if args.full_graph:
        for i in range(len(lang_ids)):
            for j in range(i+1, len(lang_ids)):
                view_pair_id = lang_ids[i][:3]+"_"+lang_ids[j][:3]+view_pair_suffix
                loss = compute_view_pair_loss(model_outputs[lang_ids[i]], model_outputs[lang_ids[j]], loss_func, loss_record, view_pair_id,
                                              loss_weight=loss_weight, **extra_loss_kwargs)
                tot_loss = tot_loss + loss

    return tot_loss, loss_record, model_outputs


def get_loss_function(args):
    if args.loss == "multiview_coding":
        return MultiViewCodingLoss(temperature=args.temperature, sim_measure=args.sim_measure)
    elif args.loss == 'triplet':
        return TripletLoss(margin=args.margin) 
    elif args.loss == 'triplet_w_hardneg':
        return TripletLoss(margin=args.margin, use_hard_neg=True)
    elif args.loss == "hyperspheric" and not args.use_custom_hsphere:
        return HypersphericLoss(alpha=args.hsphere_alpha, t=args.hsphere_t, align_weight=args.hsphere_align_weight, 
                uniform_weight=args.hsphere_uniform_weight)
    elif args.loss == "hyperspheric" and args.use_custom_hsphere:
        return None
    elif args.loss == "masked_margin_sm":
        raise NotImplementedError()
    else:
        raise ValueError(f"Could not recognize loss function {args.loss}")


def get_loss_function_wrapper(args):
    if args.loss == "hyperspheric" and args.use_custom_hsphere:
        return custom_hsphere_loss_computation
    else:
        return multiling_contrastive_computation

def init_progress(progress:dict):
    progress.update({
        "epoch": [],
        "global_step": [],
        "best_epoch": [],
        "best_acc": [],
        "tot_avg_loss": [],
        "avg_ep_loss": [],
        "avg_epoch_time": [],
        "avg_batch_time": [],
        "total_time": []
    })


def update_progress(progress: dict,
                    epoch=None,
                    global_step=None,
                    best_epoch=None,
                    best_acc=None,
                    loss_meter=None,
                    epoch_loss_meter=None,
                    epoch_timer=None,
                    batch_timer=None,
                    total_time_elapsed=None,
                    recalls=None):

    progress['epoch'].append(epoch)
    progress['global_step'].append(global_step)
    progress['best_epoch'].append(best_epoch)
    progress['best_acc'].append(best_acc)
    progress['tot_avg_loss'].append(loss_meter.avg)
    progress['avg_ep_loss'].append(epoch_loss_meter.avg)
    progress['avg_epoch_time'].append(epoch_timer.avg)
    progress['avg_batch_time'].append(batch_timer.avg)
    progress['total_time'].append(total_time_elapsed)
    for recall_stat in recalls.keys():
        progress[recall_stat].append(recalls[recall_stat])


def collect_gradient(optimizer):
    gradient = []
    for param_group in optimizer.param_groups:
        gradient.extend([t.grad.flatten() for t in param_group["params"] if t is not None and t.grad is not None])

    gradient = torch.cat(gradient, dim=-1)
    return gradient


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


def train(audio_models, image_model, train_loader, test_loader, args, exp_dir, resume):
    # Initialize all of the statistics we want to keep track of
    batch_timer = AverageMeter() # tracks batch computation time
    epoch_timer = AverageMeter() # tracks epoch computation time
    loss_meter = AverageMeter()
    start_time = time.time()
    best_epoch, best_acc = 0, -np.inf
    global_step, epoch = 0, 0
    progress = defaultdict(list)
    init_progress(progress)

    # Set device and maybe load snapshot
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(True)

    # Setup the optimizer and models
    optimizer, trainables = setup_optimizer(image_model, audio_models, args)
    image_model, audio_models = prepare_models(image_model, audio_models, device, args)

    # Create/Load experiment
    if resume:
        progress, epoch, global_step, best_epoch, best_acc = load_state(exp_dir, audio_models, image_model, device)
    else:
        for lang_id in audio_models.keys():
            torch.save(audio_models[lang_id].state_dict(),
                       "%s/models/%s_audio_model.e%d.pth" % (exp_dir,lang_id, epoch))
        torch.save(image_model.state_dict(),
                   "%s/models/image_model.e%d.pth" % (exp_dir, epoch))

    # Get loss function
    loss_func = get_loss_function(args)
    loss_func_wrapper = get_loss_function_wrapper(args)

    # Alias convenient variables
    batch_size = train_loader.batch_size
    tot_size = len(train_loader.dataset)
    batches_per_epoch = len(train_loader)
    report_mem_usage = can_report_mem_usage()

    # Report initial status to user
    epoch += 1
    report_initial_info(batch_size, tot_size, batches_per_epoch, args)

    
    cur_lr = adjust_learning_rate(args.lr, args.lr_ramp, args.lr_decay,
                                  args.lr_decay_multiplier,
                                  optimizer, global_step+1, batches_per_epoch*args.n_epochs) # +1 to show non-zero lr on first iter
    while epoch <= args.n_epochs:
        epoch_start_time = time.time()
        torch.cuda.empty_cache()


        report_epoch_info(global_step, epoch, cur_lr)

        # setup epoch stats
        epoch_time = epoch_start_time
        epoch_loss_meter = AverageMeter()
        # prep models for training
        for audio_model in audio_models.values():
            audio_model.train()
        image_model.train()
        aux_losses = None
        for i, (image_input, audio_input) in enumerate(train_loader):
            batch_start_time = time.time()

            ### Prepare input
            image_input = image_input.to(device)
            target_audio_input = get_target_multiling_data(audio_input, device, args)

            # Compute loss
            loss, aux_losses, model_outputs = loss_func_wrapper(
                                                            image_model, image_input, audio_models, target_audio_input,
                                                             device, args, loss_func=loss_func)

            # Update parameters
            optimizer.zero_grad()
            loss.backward()

            # Check Gradient size
            gradient = collect_gradient(optimizer)
            grad_norm = gradient.norm(p=2, dim=-1)
            warning_size = 100
            if grad_norm > warning_size:
                print(f"TRAINER: WARNING: (epoch step {i+1}) Gradient norm has become very large({grad_norm:.3f})).", end=" ")
                # Clip Gradient
                if args.clip_grad < 1e100:
                    print(f"Clipping to {args.clip_grad:.2}.")
                    params = get_trainable_params(optimizer)
                    torch.nn.utils.clip_grad_norm_(params, clip_to)
                else:
                    print()

            # Make update
            optimizer.step()


            # Update statistics
            loss_meter.update(loss.item(), image_input.size(0))  #Averages over entire training run
            epoch_loss_meter.update(loss.item(), image_input.size(0)) #Only for single epoch
            batch_timer.update(time.time() - batch_start_time)
            global_step += 1
            # Display current progress
            if not args.no_pbar:
                pbar_update( i,
                         batches_per_epoch,
                         epoch_loss_meter,
                         aux_losses=aux_losses,
                         report_mem_usage=report_mem_usage,
                         cur_lr=cur_lr,
                         optimizer=optimizer)

            # Optional mid epoch report
            if i % args.n_print_steps == 0:
                epoch_time = time.time()-epoch_start_time
                tot_time = time.time()-start_time
                mid_epoch_training_report(epoch, batches_per_epoch, loss_meter,
                                          epoch_loss_meter, i, batch_timer,
                                          epoch_time, tot_time, cur_lr, 
                                          args)

            # Chech if training went off the rails
            if np.isnan(loss_meter.avg):
                # preserve progress display
                print(f"\x1B[{CURR_NUM_STAT_LINES}E", end="")
                print("TRAINER: training diverged...", flush=True)
                return

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

            # Increment learning rate. Return value is just for display purposes. Increment happens in function
            cur_lr = adjust_learning_rate(args.lr, args.lr_ramp, args.lr_decay,
                                          args.lr_decay_multiplier,
                                          optimizer, global_step, batches_per_epoch*args.n_epochs)


        # validate
        recalls, best_r10 = validate(image_model, audio_models, test_loader, device, args)
        if best_r10 > best_acc:
            best_epoch = epoch
            best_acc = best_r10

        # Save info
        epoch_timer.update(time.time()-epoch_start_time) #
        total_time_elapsed = time.time()-start_time
        update_progress(progress,
                        epoch=epoch,
                        global_step=global_step,
                        best_epoch=best_epoch,
                        best_acc=best_acc,
                        loss_meter=loss_meter,
                        epoch_loss_meter=epoch_loss_meter,
                        epoch_timer=epoch_timer,
                        batch_timer=batch_timer,
                        total_time_elapsed=total_time_elapsed,
                        recalls=recalls)
        progress_df = pd.DataFrame(progress)
        save_state_and_progress(exp_dir, image_model, audio_models, optimizer, epoch, progress_df,
                   is_best_acc=(epoch == best_epoch), args=args)

        print('TRAINER: Finished epoch %d. Time elapsed in epoch = %.fs. Average epoch time = %.fs. '
              'Total time elapsed = %.fs. Current Time = %s' % (
              epoch, epoch_timer.val, epoch_timer.avg, total_time_elapsed, datetime.datetime.now()))
        epoch += 1

    print('TRAINER: Finished training. best_epoch = %s, best_acc = %s'
          % (best_epoch, best_acc))

def curate_recalls(recalls_record, recalls4display, sim_type):
    avg_recalls = Counter()
    view_ids = [k for k in recalls4display.keys()]
    for view_id in view_ids: # "view2_id->view1_id" and "view1_id->view2_id"
        for recall_width in recalls4display[view_id].keys(): #r1, r5, and r10

            # Store in record dict
            recalls_record[view_id+"_"+sim_type+"_"+recall_width] = recalls4display[view_id][recall_width]
            # sum the scores
            avg_recalls[recall_width] += recalls4display[view_id][recall_width]

    #id to display in progress record for average of both directions
    avg_id = re.sub(r"->", "&", view_ids[0])
    for recall_width in avg_recalls.keys(): #r1, r5, and r10
        # Normalize by number of views
        avg_recalls[recall_width] /= len(view_ids)

        # Organize average scores in record and display dicts
        recalls_record[avg_id+'_'+sim_type+"_avg_"+recall_width] = avg_recalls[recall_width]
        recalls4display["avg"][recall_width] = avg_recalls[recall_width]


def print_recalls(recalls, title_str, view1_id, view2_id):
    # Heading
    # \x1B is "ESCAPE" key, "[(number)E" moves cursor down (number) spaces.
    print(f"\x1B[{CURR_NUM_STAT_LINES}E", end="")
    print(f"\x1B[J",end="")
    print(f"{title_str+',':<30} view1: {view1_id} view2: {view2_id}")

    # Each line of recall scores
    for recall_id in recalls.keys():# "view2_id->view1_id", "view1_id->view2_id", and "avg"
        recall_widths = sorted([(k, int(re.sub(r"\D", "",k))) for k in recalls[recall_id].keys()], key=lambda x: x[1]) # r1, r5, and r10
        recall_widths = [k[0] for k in recall_widths]
        print(f"{recall_id+':':<20}", end=" ")
        [print(f"| {rec_width}: {recalls[recall_id][rec_width]:6.2%} ", end="") for rec_width in recall_widths]
        print(flush=True)


def curate_and_print_results(view1_output, view2_output, recalls_record, view1_id="", view2_id="", best_r10=0.):

    # Get similarity measures
    dot_S = dot_sim_matrix(view1_output, view2_output)
    cos_S = cosine_sim_matrix(view1_output, view2_output)
    norm_dot_S =dot_sim_matrix(view1_output/(view1_output.norm(p=2, dim=-1, keepdim=True) + EPSILON),
                               view2_output/(view1_output.norm(p=2, dim=-1, keepdim=True) + EPSILON)) 

    # Calculate recall scores
    dot_recalls = calc_recalls(dot_S, view1=view1_id, view2=view2_id)
    cos_recalls = calc_recalls(cos_S, view1=view1_id, view2=view2_id)
    norm_dot_recalls = calc_recalls(norm_dot_S, view1=view1_id, view2=view2_id)

    # Organize the keys and structure of the record and display dicts
    curate_recalls(recalls_record, dot_recalls, sim_type="dot")
    curate_recalls(recalls_record, cos_recalls, sim_type="cos")
    curate_recalls(recalls_record, norm_dot_recalls, sim_type="norm_dot")

    # Display results
    print()
    print_recalls(dot_recalls, "Dot Product Retrieval", view1_id, view2_id)
    print_recalls(cos_recalls, "Cosine Retrieval", view1_id, view2_id)
    print_recalls(norm_dot_recalls, "Normalized Dot Product Retrieval", view1_id, view2_id)

    # Record the best seen r10 score. Easier to do here with structure of display dict (which is discarded)
    if dot_recalls['avg']['r10'] > best_r10:
        best_r10 = dot_recalls['avg']['r10']
    if cos_recalls['avg']['r10'] > best_r10:
        best_r10 = cos_recalls['avg']['r10']
    if norm_dot_recalls['avg']['r10'] > best_r10:
        best_r10 = norm_dot_recalls['avg']['r10']

    return best_r10



def validate(image_model,audio_models, val_loader, device, args):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for lang_id in audio_models.keys():
        if not isinstance(audio_models[lang_id], torch.nn.DataParallel):
            audio_models[lang_id] = nn.DataParallel(audio_models[lang_id])
        audio_models[lang_id] = audio_models[lang_id].to(device)
        audio_models[lang_id].eval()
    if not isinstance(image_model, torch.nn.DataParallel):
        image_model = nn.DataParallel(image_model)
    image_model = image_model.to(device)
    image_model.eval()

    batch_time = time.time()
    N_examples = len(val_loader.dataset)
    I_embeddings = []
    A_embeddings = defaultdict(list)
    all_recalls = defaultdict(list)
    with torch.no_grad():
        for i, (image_input, audio_input) in enumerate(val_loader):
            # Prep batch. Audio is put on 'device' in method
            target_audio_input  = get_target_multiling_data(audio_input, device, args)
            image_input = image_input.to(device)

            # compute output
            image_output = image_model(image_input)
            image_output = image_output.to('cpu').detach()
            I_embeddings.append(image_output)
            for lang_id in audio_models.keys():
                audio_output = audio_models[lang_id](target_audio_input[lang_id]['lmspecs'], target_audio_input[lang_id]['nframes'])
                audio_output = audio_output.to('cpu').detach()
                A_embeddings[lang_id].append(audio_output)


        image_output = torch.cat(I_embeddings)
        lang_ids = [k for k in audio_models.keys()]

        best_r10 = 0.0
        for i in range(len(lang_ids)):
            audio_output = torch.cat(A_embeddings[lang_ids[i]])
            best_r10 = curate_and_print_results(image_output, audio_output, all_recalls, view1_id="image", view2_id=lang_ids[i], best_r10=best_r10)
            if args.validate_full_graph: # Compute all pairs
                for j in range(i+1, len(lang_ids)):
                    audio_output2 = torch.cat(A_embeddings[lang_ids[j]])
                    best_r10 = curate_and_print_results(image_output, audio_output2, all_recalls, view1_id=lang_ids[i], view2_id=lang_ids[j], best_r10=best_r10)

    return all_recalls, best_r10
