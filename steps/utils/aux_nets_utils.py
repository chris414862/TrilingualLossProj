from collections import defaultdict

def get_aux_nets_computation_func(loss_type):
    if loss_type == "byol":
        return compute_byol_aux_nets
    else:
        return None


def compute_byol_aux_nets(
                    base_view_id        = None,
                    base_view_tensor    = None,
                    current_view_id     = None,
                    current_view_tensor = None,
                    aux_nets            = None,
                    ):
    """
        Computes the projection and prediction layers for two views according to the BYOL framework
    """
    # Target view
    assert base_view_id in aux_nets["target"].keys(), "Logic error: in BYOL loss the base/anchor view should be the target view"
    base_view_proj_tensor = aux_nets["target"][base_view_id]["proj"](base_view_tensor)

    # Online views
    assert current_view_id in aux_nets["online"].keys(), "Logic error: in BYOL loss the current_contrasting_view should be an online view"
    current_view_proj_tensor = aux_nets["online"][current_view_id]["proj"](current_view_tensor)
    current_view_pred_tensor = aux_nets["online"][current_view_id]["pred"](current_view_proj_tensor)

    return base_view_proj_tensor, current_view_pred_tensor


def prep_byol_aux_nets(view_ids):
    """
        Returns the projection and prediction layers for each view for the byol loss.
        Structure of return dict:
            {
                "target":
                    {
                        target_view_id : {
                            "proj": ProjectionLayer
                        }
                    }
                "online": 
                    {
                        online_view_id_one : {
                            "proj": ProjectionLayer
                            "pred": PredictionLayer
                        },
                        online_view_id_two : {
                            "proj": ProjectionLayer
                            "pred": PredictionLayer
                        },
                        ....
                    }
            }

        Note that the target view does not get a prediction layer
    """
    aux_nets = defaultdict(dict)
    byol_layer_sizes = get_byol_layer_sizes(args)

    for view_id in view_ids:
        view_type = "target" if view_id == args.byol_target_view else "online"
        aux_nets[view_type][view_id] = {
                                             #get_final_layer_size returns the embed dim for the view (image and lang models might have diff sizes)
                "proj": BYOL_Layer(input_size=get_final_layer_size(view_id, args), layer_sizes=byol_layer_sizes)
                }

        # Prediction layers for all but target view
        if view_id != args.byol_target_view:
            aux_nets[view_type][view_id] = {
                    "pred": BYOL_Layer(input_size=get_final_layer_size(view_id, args), layer_sizes=byol_layer_sizes)
                    }

    return aux_nets 





