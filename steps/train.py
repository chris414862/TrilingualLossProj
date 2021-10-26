# Author: David Harwath, Wei-Ning Hsu
import numpy as np
import pickle
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
# from .util import *
# from .util2 import InfoNCE_loss
from .utils.setup_utils import get_loss_function, get_loss_framework, prepare_models, setup_optimizer
from .utils.report_utils import (report_initial_info, report_epoch_info, mid_epoch_training_report,
                                 epoch_fin_report, training_fin_report)
from .utils.general_utils import (check_gradient, AverageMeter, adjust_learning_rate, 
                                  get_target_multiling_data, free_mem, MAX_GRAD)
from .utils.load_save_utils import load_state, init_progress, update_progress, save_state_and_progress
from .utils.pbar import pbar_update
from .validation import validate


from math import ceil
from collections import defaultdict, Counter, OrderedDict


# Made this global so it could be used in a signal handler
# But pytorch doesn't pass along signal handlers to sub-processes in DataParallel (it seems)


# def flprint(*args, **kwargs):
#     print(*args, flush=True, **kwargs)
#
# def map_skip_none(fn, it):
#     """
#     emulate list(map(fn, it)) but leave None as it is.
#     """
#     ret = []
#     for x in it:
#         if x is None:
#             ret.append(None)
#         else:
#             ret.append(fn(x))
#     return ret
#
#
# def numbers_to_str(nums, precision=3):
#     msg = '('
#     num_tmp = '%%.%df' % precision
#     num_to_str = lambda x: (str(x) if x is None else num_tmp % x)
#     for num in nums[:-1]:
#         msg += num_to_str(num)
#         msg += ', '
#     msg += num_to_str(nums[-1])
#     msg += ')'
#     return msg




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

    device = torch.device("cuda" if torch.cuda.is_available() and not args.use_cpu else "cpu")
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
    loss_framework_func = get_loss_framework(args)

    # Alias convenient variables
    batch_size = train_loader.batch_size
    tot_size = len(train_loader.dataset)
    batches_per_epoch = len(train_loader)

    # Report initial status to user
    epoch += 1
    report_initial_info(batch_size, tot_size, batches_per_epoch, device, args)


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
            if i > 5:
                break
            batch_start_time = time.time()

            ### Prepare input
            image_input = image_input.to(device).type(torch.float32)
            target_audio_input = get_target_multiling_data(audio_input, device, args)

            # Compute loss
            loss, aux_losses, model_outputs = loss_framework_func(
                                                            image_model, image_input, audio_models, target_audio_input,
                                                             device, args, loss_func=loss_func)

            # Update parameters
            optimizer.zero_grad()
            loss.backward()

            # Check Gradient size
            check_gradient(optimizer, i, args)

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
                         cur_lr=cur_lr,
                         optimizer=optimizer,
                         audio_models=audio_models,
                         image_model=image_model)

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
            free_mem(loss, aux_losses, model_outputs)

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

        epoch_fin_report(epoch, epoch_timer, total_time_elapsed)
        epoch += 1

    training_fin_report(best_epoch, best_acc)











