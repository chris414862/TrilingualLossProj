import torch

from .utils.report_utils import curate_and_print_recalls



def validate(image_model,audio_models, val_loader, device, args):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for lang_id in audio_models.keys():
        if not isinstance(audio_models[lang_id], torch.nn.DataParallel):
            audio_models[lang_id] = nn.DataParallel(audio_models[lang_id])
        audio_models[lang_id] = audio_models[lang_id].to(device)
        audio_models[lang_id].eval()
    if not isinstance(image_model, torch.nn.DataParallel):
        image_model = nn.DataParallel(image_model)
    image_model = image_model.to(device)
    image_model.eval()

    batch_time = time.time()
    N_examples = len(val_loader.dataset)
    I_embeddings = []
    A_embeddings = defaultdict(list)
    all_recalls = defaultdict(list)
    with torch.no_grad():
        # Get all model outputs
        for i, (image_input, audio_input) in enumerate(val_loader):
            # Prep batch. Audio is put on 'device' in method
            target_audio_input  = get_target_multiling_data(audio_input, device, args)
            image_input = image_input.to(device)

            # compute audio output
            image_output = image_model(image_input)
            image_output = image_output.to('cpu').detach()
            # image_output dims: [batch, embed_dim]
            I_embeddings.append(image_output)
            for lang_id in audio_models.keys():
                audio_output = audio_models[lang_id](target_audio_input[lang_id]['lmspecs'], target_audio_input[lang_id]['nframes'])
                audio_output = audio_output.to('cpu').detach()
                # audio_output dims: [batch, embed_dim]
                A_embeddings[lang_id].append(audio_output)


        #default dim is 0 for torch.cat
        #TODO: make this explicit
        image_output = torch.cat(I_embeddings)  
        # image_output dims: [val dataset size, embed_dim]
        lang_ids = [k for k in audio_models.keys()]

        best_r10 = 0.0
        for i in range(len(lang_ids)):
            audio_output = torch.cat(A_embeddings[lang_ids[i]])
            # audio_output dims: [val dataset size, embed_dim]
            best_r10 = curate_and_print_recalls(image_output, audio_output, all_recalls, view1_id="image", view2_id=lang_ids[i], best_r10=best_r10)
            if args.validate_full_graph: # Compute all pairs
                for j in range(i+1, len(lang_ids)):
                    audio_output2 = torch.cat(A_embeddings[lang_ids[j]])
                    best_r10 = curate_and_print_results(audio_output, audio_output2, all_recalls, view1_id=lang_ids[i], view2_id=lang_ids[j], best_r10=best_r10)

    return all_recalls, best_r10


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
