
from steps.utils.general_utils import get_model_outputs
from losses.funcs.mse import mse


def split_target_online_views(model_outputs, view_ids, args):
    target_output = model_outputs.pop(args.target_view)
    target_id = args.target_view
    online_outputs = model_outputs
    online_ids = [k for k in model_outputs.keys()]
    return target_output, target_id, online_outputs, online_ids
    

def byol_computation(image_model, image_input, audio_models:dict, target_audio_input:dict, device, args, 
                     projection_nets=None, prediction_nets=None, **extra_loss_kwargs):
                                      # loss_weight=1.0, loss_func=None, model_outputs=None, view_ids=None,
                                      # view_pair_suffix=""
    model_outputs, view_ids = get_model_outputs(image_model, image_input, audio_models, target_audio_input, args)
    target_rep, target_id, online_reps, online_ids = split_target_online_views(model_outputs, view_ids, args)

    ### get projections ###
    target_proj = projection_nets[target_id](target_rep)
    online_projs = {}
    for online_id in online_ids:
        online_projs[online_id] = projection_nets[online_id](online_reps[online_id])

    ### get predictions (online only) ###
    online_preds = {}
    for online_id in online_ids:
        online_preds[online_id] = prediction_nets[online_id](online_projs[online_id])


    ### get losses ###
    tot_loss = 0.0
    loss_record = {}
    for online_id in online_ids:
        loss = mse(target_proj, online_reps[online_id])
        loss_record[target_id[:3]+"_"+online_id[:3]] = loss
        tot_loss = tot_loss + loss

    model_outputs.update({"target_proj":target_proj})
    model_outputs.update(online_projs)
    model_outputs.update(online_preds)
    return tot_loss, loss_record, model_outputs


        
        





