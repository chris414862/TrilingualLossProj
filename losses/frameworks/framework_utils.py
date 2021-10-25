
# def get_model_outputs(image_model, image_input, audio_models:dict, target_audio_input:dict, args):
#     model_outputs = dict() # save outputs for explicit deletion later
#     image_output = image_model(image_input)
#     # image_output dims: [batch, embed_dim]
#     
#     model_outputs['img'] = image_output
#
#     # Get output for each language
#     lang_ids = [k for k in target_audio_input.keys()]
#     for lang_id in lang_ids:
#         audio_input = target_audio_input[lang_id]['lmspecs'], target_audio_input[lang_id]['nframes']
#         audio_output = audio_models[lang_id](*audio_input, view_id=lang_id)
#         # audio dims: [batch, embed_dim]
#         model_outputs[lang_id] = audio_output
#
#
#     view_ids = ["img"] + lang_ids
#     if args.use_avg_anchor or args.use_avg_others_contrast:
#         assert not (args.use_avg_anchor and args.use_avg_others_contrast), "Should only set either --use-avg-anchor or --use-avg-others-contrast, not both"
#         model_outputs["avg"] = compute_avg_views(model_outputs.values())
#
#
#     if args.norm_outputs_in_loss:
#         for k in model_outputs.keys():
#             # EPSILON is to prevent division by zero. Located in utils.py
#             model_outputs[k] = model_outputs[k]/(model_outputs[k].norm(p=2, dim=-1, keepdim=True) + EPSILON)
#
    return model_outputs, view_ids
