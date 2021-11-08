from steps.utils.general_utils import get_model_outputs
from collections import OrderedDict
from steps.utils.aux_nets_utils import get_aux_nets_computation_func


def compute_view_pair_loss(model_output1, model_output2, loss_func,  loss_record, view_pair_id, loss_weight=1.0, **extra_loss_kwargs):
    """
        Responsible for computing the contrastive loss and recording results.
    """
    loss, aux_losses = loss_func(model_output1, model_output2, view_pair_id=view_pair_id, **extra_loss_kwargs)#, debug=True)
    loss = loss * loss_weight
    loss_record[view_pair_id] = OrderedDict()
    loss_record[view_pair_id]["total"] = loss

    # aux_losses not related to aux_nets. aux_losses are just sub-components of the full loss for the pair (recording purposes only)
    if aux_losses is not None:
        assert isinstance(aux_losses, dict)
        loss_record[view_pair_id].update(aux_losses)
    return loss


def compute_contrast(
                     base_view_id=None, base_view=None, args=None,
                     current_view_id=None, current_view=None,
                     loss_func=None, loss_record=None, loss_weight=None, 
                     aux_nets=None, extra_loss_kwargs=None, view_pair_suffix=None,
                     ):
    """
        This is basically a wrapper around compute_view_pair_loss. It is mainly responsible for computing
        any additional transformations on the views before the actual contrasting.
    """

    # Pair id is for display purposes
    view_pair_id = base_view_id[:3]+"_"+current_view_id[:3]+view_pair_suffix # Suffix added for naming flexibility

    # Get output from appropriate projection (and maybe prediction) layers, for this pair of views
    aux_net_comp_func = get_aux_nets_computation_func(args.loss)
    if aux_net_comp_func is not None:
        base_view, current_view = aux_net_comp_func(
                                                base_view_id        = base_view_id,
                                                base_view_tensor    = base_view,
                                                current_view_id     = current_view_id,
                                                current_view_tensor = current_view,
                                                aux_nets            = aux_nets,
                                                )

    loss = compute_view_pair_loss(base_view, current_view, loss_func, loss_record, view_pair_id,
                                  loss_weight=loss_weight, **extra_loss_kwargs)
    return loss


def get_base_view(args, model_outputs, current_contrasting_view):
    """
        Returns the base/anchor view according to user cli arguments
    """

    if args.use_avg_others_contrast:
        # Average all views that are not the current one

        base_contrast_view_id = "avothers"
        # remove current view from average
        assert "avg" in model_outputs.keys(), "Logic error: 'avg' view should be included in model_outputs if --use_avg_others_contrast=True"
        avg_others_view = (model_outputs["avg"]-current_contrasting_view/n_views)*n_views/(n_views-1)
        base_contrast_view = avg_others_view

    else:
        # Simply use the user defined anchor
        if args.use_avg_anchor:
            assert "avg" in model_outputs.keys(), "Logic error: 'avg' view should be included in model_outputs if --use_avg_anchor=True"
            anchor_view_id = "avg"
        elif args.use_img_anchor:
            anchor_view_id = "img"
        else: # default
            anchor_view_id = "img"

        base_contrast_view_id = anchor_view_id
        base_contrast_view = model_outputs[anchor_view_id]

    return base_contrast_view, base_contrast_view_id



def multiview_contrastive_computation(image_model, image_input, audio_models:dict, target_audio_input:dict, device, args,
                                      aux_nets=None, loss_weight=1.0, loss_func=None, model_outputs=None, view_ids=None,
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

    # compute audio-image view pairs
    tot_loss = 0.0
    loss_record = OrderedDict() 
    # view_ids = list(set([anchor_view_id]+view_ids))
    n_views = len(view_ids)
    for i in range(len(view_ids)):
        current_contrasting_view_id = view_ids[i]
        current_contrasting_view = model_outputs[current_contrasting_view_id]

        # Get the base/anchor view and id. This could potentially change with each iteration (i.e. average anchoring)
        base_contrast_view, base_contrast_view_id =  get_base_view(args, model_outputs, current_contrasting_view)

        # Don't have the base/anchor view contrast with itself. With use_avg_others_contrast, the anchor view doesn't matter
        if current_contrasting_view_id == base_contrast_view_id and not args.use_avg_others_contrast:
            continue

        # Detach anchor from graph
        if args.detach_anchor:
            base_contrast_view = base_contrast_view.detach()

        loss = compute_contrast(
                     base_view_id=base_contrast_view_id, base_view=base_contrast_view, args=args,
                     current_view_id=current_contrasting_view_id, current_view=current_contrasting_view,
                     loss_func=loss_func, loss_record=loss_record, loss_weight=loss_weight, 
                     aux_nets=aux_nets, extra_loss_kwargs=extra_loss_kwargs,view_pair_suffix=view_pair_suffix,
                     )

        tot_loss = tot_loss + loss

        # Get all pairwise contrastive losses. Contrast the current view w/ all remaining views (not the base/anchor)
        if args.full_graph:
            assert not args.use_avg_anchor, "ERROR: --full-graph and --use-avg-anchor should not be used together"
            remainder_base_view_id = current_contrasting_view_id
            remainder_base_view = current_contrasting_view

            for j in range(i+1, len(view_ids)):
                current_contrasting_view_id = view_ids[j]
                current_contrasting_view = model_outputs[current_contrasting_view_id]

                # Don't contrast with the base/anchor view from the outter loop. It was already done in the outer loop.
                if current_contrasting_view_id == base_contrast_view_id:
                    continue

                loss = compute_contrast(
                     base_view_id=remainder_base_view_id, base_view=remainder_base_view, args=args,
                     current_view_id=current_contrasting_view_id, current_view=current_contrasting_view,
                     loss_func=loss_func, loss_record=loss_record, loss_weight=loss_weight, 
                     aux_nets=aux_nets, extra_loss_kwargs=extra_loss_kwargs,view_pair_suffix=view_pair_suffix,
                     )
                tot_loss = tot_loss + loss

    # This is experimental
    if args.custom_unif_loss != "na":
        assert "avg" in model_outputs.keys(), "Custom unif loss should be used with either --use_avg_anchor or --use_avg_others_contrast"
        anchor_view = model_outputs["avg"]  
        ul = custom_unif_loss(anchor_view, args)
        tot_loss = tot_loss + ul
        loss_record["cust_unif"] = {"total":ul}


    return tot_loss, loss_record, model_outputs




