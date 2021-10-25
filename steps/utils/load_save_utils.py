
def load_state(exp_dir, audio_models, image_model, device):
    (progress, epoch, global_step, best_epoch,
     best_acc) = load_progress("%s/progress.pkl" % exp_dir)
    print("\nResume training from:")
    print("  epoch = %s" % epoch)
    print("  global_step = %s" % global_step)
    print("  best_epoch = %s" % best_epoch)
    print("  best_acc = %.4f" % best_acc)
    if epoch != 0:
        # Models' state
        for lang_id in audio_models.keys():
            audio_models[lang_id].load_state_dict(
                    torch.load("%s/models/%s_audio_model.iter.pth" % (exp_dir, lang_id)))
        image_model.load_state_dict(
                torch.load("%s/models/image_model.iter.pth" % (exp_dir)))
        print("loaded parameters from epoch %d" % epoch)

        # Optimizer state
        optimizer.load_state_dict(
                torch.load("%s/models/optim_state.iter.pth" % (exp_dir)))
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print("loaded state dict from epoch %d" % epoch)
    return progress, epoch, global_step, best_epoch, best_acc

def init_progress(progress:dict):
    progress.update({
        "epoch": [],
        "global_step": [],
        "best_epoch": [],
        "best_acc": [],
        "tot_avg_loss": [],
        "avg_ep_loss": [],
        "avg_epoch_time": [],
        "avg_batch_time": [],
        "total_time": []
    })


def update_progress(progress: dict,
                    epoch=None,
                    global_step=None,
                    best_epoch=None,
                    best_acc=None,
                    loss_meter=None,
                    epoch_loss_meter=None,
                    epoch_timer=None,
                    batch_timer=None,
                    total_time_elapsed=None,
                    recalls=None):

    progress['epoch'].append(epoch)
    progress['global_step'].append(global_step)
    progress['best_epoch'].append(best_epoch)
    progress['best_acc'].append(best_acc)
    progress['tot_avg_loss'].append(loss_meter.avg)
    progress['avg_ep_loss'].append(epoch_loss_meter.avg)
    progress['avg_epoch_time'].append(epoch_timer.avg)
    progress['avg_batch_time'].append(batch_timer.avg)
    progress['total_time'].append(total_time_elapsed)
    for recall_stat in recalls.keys():
        progress[recall_stat].append(recalls[recall_stat])

def save_state_and_progress(exp_dir, image_model, audio_models, optimizer, epoch, progress, is_best_acc:bool, args):

    # Save optimizer and models' state
    for lang_id in audio_models.keys():
        torch.save(audio_models[lang_id].state_dict(),
                "%s/models/%s_audio_model.iter.pth" % (exp_dir, lang_id))
    torch.save(image_model.state_dict(),
            "%s/models/image_model.iter.pth" % (exp_dir))
    torch.save(optimizer.state_dict(),
            "%s/models/optim_state.iter.pth" % (exp_dir))

    # Record models if best seen so far
    if is_best_acc:
        for lang_id in audio_models.keys():
            shutil.copyfile("%s/models/%s_audio_model.iter.pth" % (exp_dir, lang_id),
                "%s/models/best_%s_audio_model.pth" % (exp_dir, lang_id))
        shutil.copyfile("%s/models/image_model.iter.pth" % (exp_dir),
            "%s/models/best_image_model.pth" % (exp_dir))

    # Record models periodically according to args.save_every
    if args.save_every > 0 and epoch % args.save_every == 0:
        for lang_id in audio_models.keys():
            shutil.copyfile("%s/models/%s_audio_model.iter.pth" % (exp_dir, lang_id),
                            "%s/models/%s_audio_model.e%d.pth" % (exp_dir, lang_id, epoch))
        shutil.copyfile("%s/models/image_model.iter.pth" % (exp_dir),
            "%s/models/image_model.e%d.pth" % (exp_dir, epoch))

    progress.to_pickle("%s/progress.pkl" % exp_dir)


def load_progress(prog_pkl, quiet=False):
    """
    load progress pkl file
    Args:
        prog_pkl(str): path to progress pkl file
    Return:
        progress(list):
        epoch(int):
        global_step(int):
        best_epoch(int):
        best_avg_r10(float):
    """
    def _print(msg):
        if not quiet:
            print(msg)

    with open(prog_pkl, "rb") as f:
        prog = pickle.load(f)
        epoch, global_step, best_epoch, best_avg_r10, _ = prog[-1]

    _print("\nPrevious Progress:")
    msg =  "[%5s %7s %5s %7s %6s]" % ("epoch", "step", "best_epoch", "best_avg_r10", "time")
    _print(msg)
    return prog, epoch, global_step, best_epoch, best_avg_r10

