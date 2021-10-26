import datetime

from losses.funcs.loss_func_utils import dot_sim_matrix, cosine_sim_matrix



def report_initial_info(batch_size, tot_size, updates_per_epoch, device, args):
    print('TRAINER: Using device type %s' % device)
    if device == "cuda":
        print('TRAINER: Found %d GPUs' % torch.cuda.device_count())

    print(f"TRAINER: batch size: {batch_size}, dataset size: {tot_size}, updates per epoch: {updates_per_epoch}")
    print(f"TRAINER: Training with {args.langs} dataset only")
    print("TRAINER: Starting training...", flush=True)


def report_epoch_info(global_step, epoch, cur_lr):
    print("TRAINER: Current #steps=%s, #epochs=%s" % (global_step, epoch))
    print('TRAINER: Learning rate @ %d is %.8f' % (epoch, cur_lr), flush=True)

def mid_epoch_training_report(epoch, batches_per_epoch, loss_meter,
                              epoch_loss_meter, i, batch_timer,
                              epoch_time_elapsed, tot_time, cur_lr, 
                              args):
    from .pbar import CURR_NUM_STAT_LINES

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


def epoch_fin_report(epoch, epoch_timer, total_time_elapsed):
    print('TRAINER: Finished epoch %d. Time elapsed in epoch = %.fs. Average epoch time = %.fs. '
          'Total time elapsed = %.fs. Current Time = %s' % (
          epoch, epoch_timer.val, epoch_timer.avg, total_time_elapsed, datetime.datetime.now()))



def training_fin_report(best_epoch, best_acc):
    print('TRAINER: Finished training. best_epoch = %s, best_acc = %s'
          % (best_epoch, best_acc))













