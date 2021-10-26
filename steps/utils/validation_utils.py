import torch
import re
from collections import defaultdict, Counter

from losses.funcs.loss_func_utils import dot_sim_matrix, cosine_sim_matrix
from .general_utils import AverageMeter


def calc_recalls(S, view1="", view2=""):
    """
    Computes recall at 1, 5, and 10 given a similarity matrix S.
    By convention, rows of S are assumed to correspond to images (view1) and columns are captions (view2).
    """
    assert(S.dim() == 2)
    assert(S.size(0) == S.size(1))
    if isinstance(S, torch.autograd.Variable):
        S = S.data
    n = S.size(0)
    v1_v2_scores, v1_v2_ind = S.topk(10, 1)#along the v2 dimension
    v2_v1_scores, v2_v1_ind = S.topk(10, 0)#along the v1 dimension
    v1_r1 = AverageMeter()
    v1_r5 = AverageMeter()
    v1_r10 = AverageMeter()
    v2_r1 = AverageMeter()
    v2_r5 = AverageMeter()
    v2_r10 = AverageMeter()
    for i in range(n):
        v1_foundind = -1
        v2_foundind = -1
        for ind in range(10):
            if v2_v1_ind[ind, i] == i:
                v1_foundind = ind
            if v1_v2_ind[i, ind] == i:
                v2_foundind = ind
        # do r1s
        if v2_foundind == 0:
            v2_r1.update(1)
        else:
            v2_r1.update(0)
        if v1_foundind == 0:
            v1_r1.update(1)
        else:
            v1_r1.update(0)
        # do r5s
        if v2_foundind >= 0 and v2_foundind < 5:
            v2_r5.update(1)
        else:
            v2_r5.update(0)
        if v1_foundind >= 0 and v1_foundind < 5:
            v1_r5.update(1)
        else:
            v1_r5.update(0)
        # do r10s
        if v2_foundind >= 0 and v2_foundind < 10:
            v2_r10.update(1)
        else:
            v2_r10.update(0)
        if v1_foundind >= 0 and v1_foundind < 10:
            v1_r10.update(1)
        else:
            v1_r10.update(0)

    recalls = defaultdict(dict)
    recalls.update({view2+"->"+view1:{'r1':v2_r1.avg, 'r5':v2_r5.avg, 'r10':v2_r10.avg},
           view1+"->"+view2:{'r1':v1_r1.avg, 'r5':v1_r5.avg, 'r10':v1_r10.avg}})
                #'A_meanR':A_meanR.avg, 'I_meanR':I_meanR.avg}

    return recalls


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
    from .pbar import CURR_NUM_STAT_LINES
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
