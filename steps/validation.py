import torch
import torch.nn as nn
import time
from collections import defaultdict

from .utils.validation_utils import curate_and_print_recalls, curate_and_print_recalls
from .utils.general_utils import get_target_multiling_data



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
            image_input = image_input.to(device).type(torch.float32)

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
                    best_r10 = curate_and_print_recalls(audio_output, audio_output2, all_recalls, view1_id=lang_ids[i], view2_id=lang_ids[j], best_r10=best_r10)

    return all_recalls, best_r10



