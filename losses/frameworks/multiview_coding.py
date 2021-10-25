from steps.utils.general_utils import get_model_outputs
from collections import OrderedDict


def compute_view_pair_loss(model_output1, model_output2, loss_func,  loss_record, view_pair_id, loss_weight=1.0, **extra_loss_kwargs):
    loss, aux_losses = loss_func(model_output1, model_output2, view_pair_id=view_pair_id, **extra_loss_kwargs)#, debug=True)
    loss = loss * loss_weight
    loss_record[view_pair_id] = OrderedDict()
    loss_record[view_pair_id]["total"] = loss
    if aux_losses is not None:
        assert isinstance(aux_losses, dict)
        loss_record[view_pair_id].update(aux_losses)
    return loss

def multiview_contrastive_computation(image_model, image_input, audio_models:dict, target_audio_input:dict, device, args,
                                      loss_weight=1.0, loss_func=None, model_outputs=None, view_ids=None,
                                      view_pair_suffix="", **extra_loss_kwargs):
    '''
       Logic for multilingual contrastive loss computation. We assume the image modality is the anchor
       unless the '--full-graph' argument is set
       Parameters:
       Return:
            batch_loss: torch.Tenser - loss for the batch. Size is []. I.e the loss is a scalar
    '''

    if model_outputs is None: 
        model_outputs, view_ids = get_model_outputs(image_model, image_input, audio_models, target_audio_input, args)

    if args.use_avg_anchor:
        anchor_view_id = "avg"
    elif args.use_img_anchor:
        anchor_view_id = "img"
    else: 
        anchor_view_id = "img"


    # compute audio-image view pairs
    tot_loss = 0.0
    loss_record = OrderedDict() 
    view_ids = list(set([anchor_view_id]+view_ids))
    n_views = len(view_ids)
    for i in range(len(view_ids)):
        if view_ids[i] == anchor_view_id and not args.use_avg_others_contrast:
            continue

        if args.use_avg_others_contrast:
            base_contrast_view_id = "avothers"
            # remove current view from average
            avg_others_view = (model_outputs["avg"]-model_outputs[view_ids[i]]/n_views)*n_views/(n_views-1)
            base_contrast_view = avg_others_view

        else:
            base_contrast_view_id = anchor_view_id
            base_contrast_view = model_outputs[anchor_view_id]

        view_pair_id = base_contrast_view_id[:3]+"_"+view_ids[i][:3]+view_pair_suffix # Suffix added for naming flexibility

        # Detach anchor from graph
        if args.detach_anchor:
            base_contrast_view = base_contrast_view.detach()

        loss = compute_view_pair_loss(base_contrast_view, model_outputs[view_ids[i]], loss_func, loss_record, view_pair_id,
                                      loss_weight=loss_weight, **extra_loss_kwargs)
        tot_loss = tot_loss + loss

        # Get all pairwise contrastive losses
        if args.full_graph:
            assert not args.use_avg_anchor, "ERROR: --full-graph and --use-avg-anchor should not be used together"

            for j in range(i+1, len(view_ids)):
                if view_ids[j] == anchor_view_id:
                    continue
                view_pair_id = view_ids[i][:3]+"_"+view_ids[j][:3]+view_pair_suffix
                loss = compute_view_pair_loss(model_outputs[view_ids[i]], model_outputs[view_ids[j]], loss_func, loss_record, view_pair_id,
                                              loss_weight=loss_weight, **extra_loss_kwargs)
                tot_loss = tot_loss + loss

    if args.custom_unif_loss != "na":
        assert "avg" in model_outputs.keys(), "Custom unif loss should be used with either --use_avg_anchor or --use_avg_others_contrast"
        anchor_view = model_outputs["avg"]  
        ul = custom_unif_loss(anchor_view, args)
        tot_loss = tot_loss + ul
        loss_record["cust_unif"] = {"total":ul}


    return tot_loss, loss_record, model_outputs
