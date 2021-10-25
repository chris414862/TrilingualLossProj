import torch
from .general_utils import collect_gradient_from_models, collect_gradient_from_opt, get_param_norm


CURR_NUM_STAT_LINES = 0

def can_report_mem_usage():
    version_pieces = torch.__version__.split(".")
    if len(version_pieces) < 2:
        print("Unexpected version formatting from Pytorch. Disabling memory usage display in progress bar")
        return False
    try:
        major = int(version_pieces[0])
        minor = int(version_pieces[0])
    except ValueError as e:
        print("Unexpected version formatting from Pytorch. Disabling memory usage display in progress bar")
        return False

    if major >= 1 and minor >=4:
        return True
    else:
        return False

REPORT_MEM_USAGE = can_report_mem_usage()

def pbar_update(i, updates_per_epoch, loss_meter, update_every=1, bar_parts=50, aux_losses=None, report_mem_usage=False, cur_lr=None, optimizer=None,
                audio_models=None, image_model=None):
    if  i % update_every  == 0:
        global CURR_NUM_STAT_LINES
        # This is an ANSI CSI (Control Sequence Introducer) Sequences.
        # On Unix-like system's \x1B is ESC. "[J" clears from the cursor to the end of the screen
        print(f"\x1B[J",end="")
        cols = 150
        prefix_str = f"{(i+1):>7}/{updates_per_epoch} "
        stat_lines = [f"{prefix_str}| Ep.Loss avg: {loss_meter.avg:<9.3f} cur: {loss_meter.val:<9.3f} "]
        if cur_lr is not None:
            stat_lines[0] += f"| lr: {cur_lr:<.2e} "

        # if optimizer is not None:
        #     grad_norm = collect_grad_norms()
        #     param_norm = get_param_norm(optimizer)
        #     stat_lines.append(" "*len(prefix_str)+f"| tot param norm: {param_norm:.3f}  grad_norm: {grad_norm:.3f}")

        if audio_models is not None and image_model is not None:
            # Display gradient norms for each model
            grad_norms = collect_gradient_from_models(audio_models, image_model, normalize=True)
            param_norm = get_param_norm(optimizer)
            new_line = " "*len(prefix_str)+"| grad norms: "
            for model_id, grad_norm in grad_norms.items():
                new_line += f"| {model_id+':'} {grad_norm:10.3f} "
            
            tot_grad_norm = collect_gradient_from_opt(optimizer, normalize=True)
            new_line += f"| tot: {tot_grad_norm:10.3f}"
            stat_lines.append(new_line)

            # Display parameter norm TODO: display for each model
            new_line = " "*len(prefix_str)+f"| param norm: {param_norm:10.3f} "
            stat_lines.append(new_line)


        if aux_losses is not None:
            for view_pair_key, loss_dict in aux_losses.items():
                for loss_type, loss_val in loss_dict.items():
                    curr_stat_line = stat_lines.pop(-1)
                    new_str = f"| {view_pair_key+'_'+loss_type+':':<12} {loss_val.item():10.3f} "
                    if len(curr_stat_line + new_str) > cols or loss_type.strip() == "total":
                        stat_lines.append(curr_stat_line)
                        stat_lines.append(" "*len(prefix_str)+new_str)
                    else:
                        stat_lines.append(curr_stat_line+new_str)

        if REPORT_MEM_USAGE:
            curr_stat_line = stat_lines.pop(-1)
            new_str = f" | mem: {memr:5.2f}gb"
            if len(curr_stat_line + new_str) > cols:
                stat_lines.append(curr_stat_line)
                stat_lines.append(" "*len(prefix_str)+new_str)
            else:
                stat_lines.append(curr_stat_line+new_str)

        for stat_line in stat_lines:
            print(stat_line)
        parts_done = int((i+1)/updates_per_epoch*bar_parts)
        parts_togo = int((updates_per_epoch-i-1)/updates_per_epoch*bar_parts)
        print(" "*len(prefix_str)+"|"+"-"*parts_done+">"+" "*parts_togo+"|", flush=True)

        # +1 for the status bar.
        CURR_NUM_STAT_LINES = len(stat_lines)+1

        # "[(number)A" moves cursor up (number) spaces.
        print(f"\x1B[{CURR_NUM_STAT_LINES}F",end="", flush=True )
