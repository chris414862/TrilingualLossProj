# Author: David Harwath, Wei-Ning Hsu

import datetime
import numpy as np
import pickle
import shutil
import time
import torch
import torch.nn as nn
import sys

from models.quantizers import compute_perplexity
from .util import *


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


def get_target_monoling_data(audio_input, args):
    # english_input, hindi_input, j_input = audio_input["lmspecs"] #log-mel spectrograms
    # english_nframes, hindi_nframes, j_nframes = audio_input["nframes"]
    if args.monoling.lower() == "english":
        return audio_input["english"]["lmspecs"], audio_input["english"]["nframes"]
        # return english_input, english_nframes
    elif args.monoling.lower() == "hindi":
        return audio_input["hindi"]["lmspecs"], audio_input["hindi"]["nframes"]
        # return hindi_input,  hindi_nframes
    elif args.monoling.lower() == "japanese":
        return audio_input["japanese"]["lmspecs"], audio_input["japanese"]["nframes"]
        # return j_input, j_nframes
    else:
        print(f"TRAINER: Did not recognize --monoling argument: {args.monoling}")
        print("TRAINER: Only use one of {'english', 'hindi', 'japanese'}")
        sys.exit()


def get_target_multiling_data(audio_input, args):
    english_input, hindi_input, japanese_input = audio_input["lmspecs"] #log-mel spectrograms
    english_nframes, hindi_nframes, japanese_nframes = audio_input["nframes"]
    assert args.multiling is not None
    langs = [lang.strip().lower() for lang in args.multiling.split(",")]
    target_audio_input = dict()
    target_audio_nframes = dict()
    if "english" in langs:
        target_audio_input['english'] = audio_input["english"]["lmspecs"]
        target_audio_nframes['english'] = audio_input["english"]["nframes"]
    elif "hindi" in langs:
        target_audio_input['hindi'] = audio_input["hindi"]["lmspecs"]
        target_audio_nframes['hindi'] = audio_input["hindi"]["nframes"]
        # target_audio_input['hindi'] = hindi_input
        # target_audio_nframes['hindi'] = hindi_nframes
    elif "japanese" in langs:
        target_audio_input['japanese'] = audio_input["japanese"]["lmspecs"]
        target_audio_nframes['japanese'] = audio_input["japanese"]["nframes"]
        # target_audio_input['japanese'] = japanese_input
        # target_audio_nframes['japanese'] = japanese_nframes

    target_audio_input = target_audio_input.to(device)

    return target_audio_input, target_audio_nframes


def progress_update(i, updates_per_epoch, loss_meter, update_freq=500, bar_parts=50):
    memr = torch.cuda.max_memory_reserved('cuda')/(2**20)
    if (i+1) % (updates_per_epoch//update_freq) == 0:
        print(f"{(i+1):>7}/{updates_per_epoch} | Ep.Loss avg: {loss_meter.avg:<5.3f} cur: {loss_meter.val:<5.3f} | {memr:10.3f}", end="  |")
        parts_done = int((i+1)/updates_per_epoch*bar_parts)
        parts_togo = int((updates_per_epoch-i-1)/updates_per_epoch*bar_parts)
        print("-"*parts_done+">"+" "*parts_togo+"|", end="\r", flush=True)


def mid_epoch_training_report(epoch, batches_per_epoch,loss_meter, epoch_loss_meter, i, batch_timer, args):
    print('Epoch: [{0}][{1}/{2}]'
          '  Time={bt.val:.3f} ({bt.avg:.3f})'
          # '  Data load time={dt.val:.3f} ({dt.avg:.3f})'
          # '  Loss={lt.val:.3f} ({lt.avg:.3f})'
          # '  Bwd={bwdt.val:.3f} ({bwdt.avg:.3f})'
          '  Current loss: {loss.val:.4f}'
          '  Total loss avg: {loss.avg:.4f}'
          '  Avg. loss for ep.: {epoch_loss.avg:.4f}'
          # '  QLoss={qloss_str:s}  Perplexity={ppl_str:s}'
          # '  MultiLing={is_multiling} ({multilangs})'
          '  MonoLing={is_monoling} ({monolang})'.format(
                       epoch, (i+1), batches_per_epoch, 
                       bt=batch_timer,
                       # dt=data_timer, 
                       # lt=loss_timer, 
                       # bwdt         = bwd_timer,
                       loss         = loss_meter,
                       epoch_loss   = epoch_loss_meter,
                       # ppl_str      = perplexities_str,
                       # qloss_str    = quant_losses_str,
                       is_monoling  = args.monoling is not None,
                       # is_multiling = args.multiling is not None,
                       monolang     = args.monoling if args.monoling is not None else "N/A",
                       # multilangs   = args.multiling if args.multiling is not None else "N/A"
                       ),
           flush = True)
    # recalls = validate(audio_model, image_model, test_loader, args)


def load_state(exp_dir, audio_model, image_model, device):
    (progress, epoch, global_step, best_epoch, 
     best_acc) = load_progress("%s/progress.pkl" % exp_dir)
    print("\nResume training from:")
    print("  epoch = %s" % epoch)
    print("  global_step = %s" % global_step)
    print("  best_epoch = %s" % best_epoch)
    print("  best_acc = %.4f" % best_acc)
    if epoch != 0:
        # Models' state
        audio_model.load_state_dict(
                torch.load("%s/models/audio_model.iter.pth" % (exp_dir)))
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


def prepare_models(image_model, audio_model, device):
    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    if not isinstance(image_model, torch.nn.DataParallel):
        image_model = nn.DataParallel(image_model)

    audio_model = audio_model.to(device)
    image_model = image_model.to(device)
    return image_model, audio_model


def setup_optimizer(image_model, audio_model, args):
    # Gather trainable parameters
    audio_trainables = [p for p in audio_model.parameters() if p.requires_grad]
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
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(trainables, args.lr,
                                weight_decay=args.weight_decay)
    else:
        raise ValueError('Optimizer %s is not supported' % args.optim)

    return optimizer, trainables

def report_initial_info(batch_size, tot_size, updates_per_epoch, args):
    print('TRAINER: Found %d GPUs' % torch.cuda.device_count())
    print(f"TRAINER: batch size: {batch_size}, dataset size: {tot_size}, updates per epoch: {updates_per_epoch}")
    if args.monoling is not None:
        print(f"TRAINER: Training with {args.monoling} dataset only")
    print("TRAINER: Starting training...")
    

def report_epoch_info(global_step, epoch, cur_lr):
    print("TRAINER: Current #steps=%s, #epochs=%s" % (global_step, epoch))
    print('TRAINER: Learning rate @ %5d is %f' % (epoch, cur_lr))


def save_state_and_progress(exp_dir, image_model, audio_model, optimizer, epoch, progress, is_best_acc:bool, args):

    # Save optimizer and models' state
    torch.save(audio_model.state_dict(),
            "%s/models/audio_model.iter.pth" % (exp_dir))
    torch.save(image_model.state_dict(),
            "%s/models/image_model.iter.pth" % (exp_dir))
    torch.save(optimizer.state_dict(), 
            "%s/models/optim_state.iter.pth" % (exp_dir))
    
    # Record models if best seen so far
    if is_best_acc:
        shutil.copyfile("%s/models/audio_model.iter.pth" % (exp_dir), 
            "%s/models/best_audio_model.pth" % (exp_dir))
        shutil.copyfile("%s/models/image_model.iter.pth" % (exp_dir), 
            "%s/models/best_image_model.pth" % (exp_dir))

    # Record models periodically according to args.save_every
    if args.save_every > 0 and epoch % args.save_every == 0:
        shutil.copyfile("%s/models/audio_model.iter.pth" % (exp_dir), 
            "%s/models/audio_model.e%d.pth" % (exp_dir, epoch))
        shutil.copyfile("%s/models/image_model.iter.pth" % (exp_dir), 
            "%s/models/image_model.e%d.pth" % (exp_dir, epoch))

    # Save progress
    with open("%s/progress.pkl" % exp_dir, "wb") as f:
        pickle.dump(progress, f)


def run_validation(image_model, audio_model, test_loader, epoch, epoch_time, device, args):

    recalls = validate(audio_model, image_model, test_loader, device, args)
    avg_acc = (recalls['A_r10'] + recalls['I_r10']) / 2
    print('TRAINER: Finished epoch %d. Time elapsed = %.fs. Current Time = %s' % (
          epoch, time.time() - epoch_time, datetime.datetime.now()))

    return recalls, avg_acc


def orig_loss_aggregation(image_output,  target_audio_output, target_audio_nframes, target_audio_input):
    '''
        Original logic for aggregating all loss functions 
    '''
    
    ### We won't be using quantization
    # flat_inputs = [v if v is None else v.detach() for v in flat_inputs]
    # flat_onehots = [v if v is None else v.detach() for v in flat_onehots]
    # audio_model.module.ema_update(flat_inputs, flat_onehots)

    # quant_losses = map_skip_none(torch.mean, quant_losses)
    # perplexities = map_skip_none(compute_perplexity, flat_onehots)
    # quant_losses_str = numbers_to_str(quant_losses)
    # perplexities_str = numbers_to_str(perplexities)
    pooling_ratio = round(target_audio_input.size(-1) / target_audio_output.size(-1))
    target_audio_nframes = target_audio_nframes.type(torch.float).div(pooling_ratio).type(torch.long)
    S = compute_pooldot_similarity_matrix(
            image_output, target_audio_output, target_audio_nframes)
    I2A_sampled_loss = sampled_triplet_loss_from_S(S, args.margin)
    A2I_sampled_loss = sampled_triplet_loss_from_S(S.t(), args.margin)
    I2A_hardneg_loss = semihardneg_triplet_loss_from_S(S, args.margin)
    A2I_hardneg_loss = semihardneg_triplet_loss_from_S(S.t(), args.margin)

    loss = I2A_sampled_loss + A2I_sampled_loss + \
           I2A_hardneg_loss + A2I_hardneg_loss

    ### We won't be using quantization
    # qloss = [l for l in quant_losses if l is not None]
    # qloss = torch.sum(torch.stack(qloss)) if bool(qloss) else None
    # if qloss is not None:
    #     loss = loss + qloss
    aux_losses = (I2A_sampled_loss, A2I_sampled_loss, I2A_hardneg_loss, A2I_hardneg_loss)
    return loss, aux_losses

def monolingual_loss_computation(image_model, image_input, audio_model, target_audio_input, target_audio_nframes, args):
    '''
       Logic for simple monolingual loss computation 
       Parameters:
            image_model: torch.nn.Module (subclass) - image network

            image_input: torch.Tensor - tensor of size [batch, channel, height, width]
                                      - representing a batch of images
                                       Tensor size should be:
                                           [batch_size, channels, height, width]

            audio_model: torch.nn.Module (subclass) - Base audio model for this project

            target_audio_input: dict - a dictionary containing a mapping of all language captions
                                       for each image. Structure is:
                                          {"lang_id": torch.Tensor[,"lang_id": torch.Tensor]*}
                                       Tensor size for each language should be:
                                           [batch_size, log_mel_feats, nframes]

            target_audio_nframes: dict - a dictionary containing a mapping of language id to total
                                         number of audio frames for a single image:
                                            {"lang_id": torch.Tensor[,"lang_id": torch.Tensor]*}
                                         Tensor size for each language should be:
                                            [batch_size] 

            args: argparse.Namespace - arguments from the calling script

        Return:
            batch_loss: torch.Tenser - loss for the batch. Size is []. I.e the loss is a scalar
    '''
    # B = target_audio_input.size(0)
    # T = target_audio_input.size(-1)

    # print("image input:", image_input.size())
    # print("audio input:", target_audio_input.size())
    # print("audio nframes:", target_audio_nframes.size())

    


    # if args.loss is None:
    #     loss_func = orig_loss_aggregation
    # elif

    loss, aux_losses = loss_fun(image_output, target_audio_input, target_audio_output, 
                                 target_audio_nframes, args)
    # print("loss: ", loss.shape)
    # print(loss)
    # sys.exit()
    model_outputs = (image_output, target_audio_output)
    return loss, aux_losses, model_outputs


# def multilingual_loss_computation():

def get_loss_function(args):
    if args.loss is None or args.loss == "orig":
        return orig_loss_aggregation
    elif args.loss == "multiview_coding":
        return MultiViewCodingLoss()

def train(audio_model, image_model, train_loader, test_loader, args, exp_dir, resume):
    # Initialize all of the statistics we want to keep track of
    batch_timer = AverageMeter()
    data_timer = AverageMeter()
    loss_timer = AverageMeter()
    bwd_timer = AverageMeter()
    loss_meter = AverageMeter()
    progress = []
    best_epoch, best_acc = 0, -np.inf
    global_step, epoch = 0, 0
    batch_time = time.time()

    # Set device and maybe load snapshot
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(True)

    # Setup the optimizer and models
    optimizer, trainables = setup_optimizer(image_model, audio_model, args)
    image_model, audio_model = prepare_models(image_model, audio_model, device)

    # Create/Load experiment
    if resume:
        progress, epoch, global_step, best_epoch, best_acc = load_state(exp_dir, audio_model, image_model, device)
    else: 
        torch.save(audio_model.state_dict(), 
                   "%s/models/audio_model.e%d.pth" % (exp_dir, epoch))
        torch.save(image_model.state_dict(), 
                   "%s/models/image_model.e%d.pth" % (exp_dir, epoch))

    # Get loss function
    loss_func = get_loss_function(args)

    # Alias convenient variables
    batch_size = train_loader.batch_size
    tot_size = len(train_loader.dataset)
    updates_per_epoch = tot_size//batch_size

    # Report initial status to user
    epoch += 1
    report_initial_info(batch_size, tot_size, updates_per_epoch, args)

    # Start training
    while epoch <= args.n_epochs:
        torch.cuda.empty_cache()
        cur_lr = adjust_learning_rate(args.lr, args.lr_decay, 
                                      args.lr_decay_multiplier,
                                      optimizer, epoch)

        report_epoch_info(global_step, epoch, cur_lr)

        epoch_time = time.time()
        batch_time = time.time()

        batches_per_epoch = len(train_loader)

        audio_model.train()
        image_model.train()
        epoch_loss_meter = AverageMeter()
        for i, (image_input, audio_input) in enumerate(train_loader):
            # if i > 10:
            #     break
            # Display current progress 
            progress_update(i, updates_per_epoch, epoch_loss_meter)

            # Update tracking variables
            data_timer.update(time.time() - batch_time)
            start_time = time.time()

            ### Prepare input
            image_input = image_input.to(device)
            actual_batch_size = image_input.size(0) # might be less than batch_size argument on last iteration
            # For Monolingual models
            if args.monoling is not None:
                target_audio_input, target_audio_nframes = get_target_monoling_data(audio_input, args)            
                loss_computation = monolingual_loss_computation
            # For multilingual
            elif args.multiling is not None:
                target_audio_input, target_audio_nframes = get_target_multiling_data(audio_input, args)            
                loss_computation = multilingual_loss_computation
            # Default to Monolingual
            else:
                # Simple take the first language in the monolingual setting (should be english in our metadata json)
                target_audio_input, target_audio_nframes = audio_input["lmspecs"][0], audio_input["nframes"][0]
                loss_computation = monolingual_loss_computation


            image_output = image_model(image_input)
            # image dims: [batch, embed_dim]
            (target_audio_output, quant_losses, flat_inputs, 
             flat_onehots) = audio_model(target_audio_input, target_audio_nframes)
            # audio dims: [batch, embed_dim]



            # Compute loss and update parameters
            loss, aux_losses = loss_func(image_output,  target_audio_output, target_audio_nframes, target_audio_input)
                    # image_model, image_input, audio_model, 
                    #                 target_audio_input, target_audio_nframes, args)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update statistics
            loss_meter.update(loss.item(), actual_batch_size)  #Averages over entire training run
            epoch_loss_meter.update(loss.item(), actual_batch_size) #Only for single epoch
            loss_timer.update(time.time() - start_time) #Chris Crabtree: Not sure why this is necessary
            start_time = time.time()
            batch_timer.update(time.time() - batch_time)
            batch_time = time.time()
            global_step += 1
            
            # Optional mid epoch report
            if (global_step) % args.n_print_steps == 0:
                mid_epoch_training_report(epoch, batches_per_epoch,loss_meter, epoch_loss_meter, i, batch_timer, args)

            # Chech if training went off the rails
            if np.isnan(loss_meter.avg):
                print("TRAINER: training diverged...")
                return

            # Free up VRAM memory explicitly
            # I2A_sampled_loss, A2I_sampled_loss, I2A_hardneg_loss, A2I_hardneg_loss = aux_losses
            # image_output, target_audio_output, quant_losses, flat_inputs, flat_onehots = model_outputs
            del (#I2A_sampled_loss, A2I_sampled_loss, I2A_hardneg_loss, 
                 #A2I_hardneg_loss, 
                 #qloss, 
                 loss, 
                 # S, 
                 image_output, target_audio_output, 
                 quant_losses, 
                 # perplexities, 
                 flat_inputs, flat_onehots
                 )

        recalls, avg_acc = run_validation(image_model, audio_model, test_loader, epoch, epoch_time, device, args)
        if avg_acc > best_acc:
            best_epoch = epoch
            best_acc = avg_acc

        # Save info
        progress.append([epoch, global_step, best_epoch, best_acc, time.time() - batch_time])
        save_state_and_progress(exp_dir, image_model, audio_model, optimizer, epoch, progress, 
                   is_best_acc=(epoch == best_epoch), args=args)

        epoch += 1

    print('TRAINER: Finished training. best_epoch = %s, best_acc = %s'
          % (best_epoch, best_acc))


def validate(audio_model, image_model, val_loader, device, args):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    if not isinstance(image_model, torch.nn.DataParallel):
        image_model = nn.DataParallel(image_model)
    audio_model = audio_model.to(device)
    image_model = image_model.to(device)
    
    batch_time = time.time()
    N_examples = len(val_loader.dataset)
    I_embeddings = [] 
    A_embeddings = [] 
    frame_counts = []
    image_model.eval()
    audio_model.eval()
    with torch.no_grad():
        for i, (image_input, audio_input) in enumerate(val_loader):
            # english_input, hindi_input, j_input = audio_input["lmspecs"] #log-mel spectrograms
            # english_nframes, hindi_nframes, j_nframes = audio_input["nframes"]
            # image_input = image_input.to(device)
            # audio_input = audio_input.to(device)

            # TODO: Fix eval for multilingual
            # For Monolingual models
            if args.monoling is not None:
                target_audio_input, target_audio_nframes = get_target_monoling_data(audio_input, args)            
            else:
                target_audio_input, target_audio_nframes = audio_input["lmspecs"][0], audio_input["nframes"][0]

            # For testing
            image_input = image_input.to(device)
            target_audio_input = target_audio_input.to(device)

            # target_audio_input = english_input
            # target_audio_nframes = english_nframes

            # compute output
            image_output = image_model(image_input)
            target_audio_output, _, _, _ = audio_model(target_audio_input)

            image_output = image_output.to('cpu').detach()
            target_audio_output = target_audio_output.to('cpu').detach()

            I_embeddings.append(image_output)
            A_embeddings.append(target_audio_output)
            
            pooling_ratio = round(target_audio_input.size(-1) / target_audio_output.size(-1))
            # nframes.div_(pooling_ratio)
            target_audio_nframes = target_audio_nframes.type(torch.float).div(pooling_ratio).type(torch.long)

            frame_counts.append(target_audio_nframes.cpu())

        image_output = torch.cat(I_embeddings)
        audio_output = torch.cat(A_embeddings)
        nframes = torch.cat(frame_counts)
        S = compute_pooldot_similarity_matrix(
                image_output, audio_output, nframes)
        recalls = calc_recalls(S)
        A_r10 = recalls['A_r10']
        I_r10 = recalls['I_r10']
        M_r10 = (A_r10 + I_r10) / 2.
        A_r5 = recalls['A_r5']
        I_r5 = recalls['I_r5']
        M_r5 = (A_r5 + I_r5) / 2.
        A_r1 = recalls['A_r1']
        I_r1 = recalls['I_r1']
        M_r1 = (A_r1 + I_r1) / 2.

    print(' * Audio R@10 {A_r10:.3f} / Image R@10 {I_r10:.3f}'
          ' / Mean R@10 {M_r10:.3f} over {N:d} validation pairs'.format(
          A_r10=A_r10, I_r10=I_r10, M_r10=M_r10, N=N_examples), flush=True)
    print(' * Audio R@5 {A_r5:.3f} / Image R@5 {I_r5:.3f}'
          ' / Mean R@5 {M_r5:.3f} over {N:d} validation pairs'.format(
          A_r5=A_r5, I_r5=I_r5, M_r5=M_r5, N=N_examples), flush=True)
    print(' * Audio R@1 {A_r1:.3f} / Image R@1 {I_r1:.3f}'
          ' / Mean R@1 {M_r1:.3f} over {N:d} validation pairs'.format(
          A_r1=A_r1, I_r1=I_r1, M_r1=M_r1, N=N_examples), flush=True)

    return recalls
