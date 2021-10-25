
from .pbar import CURR_NUM_STAT_LINES
from losses.funcs.loss_func_utils import dot_sim_matrix, cosine_sim_matrix

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
        # make sure recalls are ordered r1, r5, r10. Must sort by string value and not int value
        recall_widths = sorted([(k, int(re.sub(r"\D", "",k))) for k in recalls[recall_id].keys()], key=lambda x: x[1]) # r1, r5, and r10
        recall_widths = [k[0] for k in recall_widths]
        print(f"{recall_id+':':<20}", end=" ")
        [print(f"| {rec_width}: {recalls[recall_id][rec_width]:6.2%} ", end="") for rec_width in recall_widths]
        print(flush=True)


def curate_and_print_recalls(view1_output, view2_output, recalls_record, view1_id="", view2_id="", best_r10=0.):

    # Get similarity measures
    dot_S = dot_sim_matrix(view1_output, view2_output)
    cos_S = cosine_sim_matrix(view1_output, view2_output)

    # Calculate recall scores
    dot_recalls = calc_recalls(dot_S, view1=view1_id, view2=view2_id)
    cos_recalls = calc_recalls(cos_S, view1=view1_id, view2=view2_id)
    # norm_dot_recalls = calc_recalls(norm_dot_S, view1=view1_id, view2=view2_id)

    # Organize the keys and structure of the record and display dicts
    curate_recalls(recalls_record, dot_recalls, sim_type="dot")
    curate_recalls(recalls_record, cos_recalls, sim_type="cos")
    # curate_recalls(recalls_record, norm_dot_recalls, sim_type="norm_dot")

    # Display results
    print()
    print_recalls(dot_recalls, "Dot Product Retrieval", view1_id, view2_id)
    print_recalls(cos_recalls, "Cosine Retrieval", view1_id, view2_id)
    # print_recalls(norm_dot_recalls, "Normalized Dot Product Retrieval", view1_id, view2_id)

    # Record the best seen r10 score. Easier to do here with structure of display dict (which is discarded)
    if dot_recalls['avg']['r10'] > best_r10:
        best_r10 = dot_recalls['avg']['r10']
    if cos_recalls['avg']['r10'] > best_r10:
        best_r10 = cos_recalls['avg']['r10']

    return best_r10


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
