from .multiview_coding import multiview_contrastive_computation
from steps.utils.general_utils import get_model_outputs
from losses.funcs.hyperspheric import hsphere_uniformity_loss, hsphere_align_loss
from collections import OrderedDict


def custom_hsphere_loss_computation(image_model, image_input, audio_models:dict, target_audio_input,  device, args, **kwargs):
    
    model_outputs, view_ids = get_model_outputs(image_model, image_input, audio_models, target_audio_input, args)

    # compute audio-image pairs
    tot_loss = 0.0

    # record loss for display purposes
    loss_record = OrderedDict() #if len(lang_ids) >= 1 else None

    # get alignment loss
    loss, loss_record, _ = multiview_contrastive_computation(image_model, image_input, audio_models, target_audio_input,  device, args, 
                                      loss_func=hsphere_align_loss, loss_weight=args.hsphere_align_weight, model_outputs=model_outputs,
                                      view_ids=view_ids, view_pair_suffix="_al")
    # aligin weight applied above
    tot_loss = tot_loss + loss

    # get uniform losses
    # img_loss, _ = hsphere_uniformity_loss(model_outputs['img'], t=args.hsphere_t, view_id="img")
    # tot_loss = tot_loss + args.hsphere_uniform_weight*img_loss
    # loss_record["img_u"] = {"total": args.hsphere_uniform_weight*img_loss}
    for i in range(len(view_ids)):
        view_loss, _ = hsphere_uniformity_loss(model_outputs[view_ids[i]], t=args.hsphere_t, view_id=view_ids[i])#, debug=True)
        view_loss = args.hsphere_uniform_weight*view_loss
        tot_loss = tot_loss + view_loss
        loss_record[view_ids[i][:3]+"_u"] = {"total": view_loss}

    return tot_loss, loss_record, model_outputs
