

def param_update_step(optimizer, args):
    if args.loss == "byol":
        byol_step(optimizer, args)

    else:
        optimizer.step()


def byol_step(optimizer, args):

    
    for param_group in optimizer.param_groups: 
        assert type(param_group) == dict
        assert "view_id" in param_group.keys() and "params" in params_group.keys()
        # Update target view. The target view's parameters are frozen to the optimizer
        if param_group["view_id"] == args.byol_target_view:
            ema_updata(param_group["params"], args)
        
    # Update everything else
    optimizer.step()
                

def ema_update(params, args):
    """
        params is a list of tensors
    """
    pass 




        

        
