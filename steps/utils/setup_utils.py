import torch
import torch.nn as nn
from losses.funcs.InfoNCE import InfoNCE
from losses.funcs.triplet import TripletLoss
from losses.frameworks.hyperspheric import custom_hsphere_loss_computation
from losses.frameworks.multiview_coding import multiview_contrastive_computation


def get_loss_function(args):
    if args.loss == "info_nce":
        return InfoNCE(temperature=args.temperature, sim_measure=args.sim_measure)
    elif args.loss == 'triplet':
        return TripletLoss(margin=args.margin, use_hard_neg=args.use_hard_neg) 
    elif args.loss == 'triplet_w_hardneg':
        return TripletLoss(margin=args.margin, use_hard_neg=True)
    # elif args.loss == "hyperspheric" and not args.use_custom_hsphere:
    #     return HypersphericLoss(alpha=args.hsphere_alpha, t=args.hsphere_t, align_weight=args.hsphere_align_weight, 
    #             uniform_weight=args.hsphere_uniform_weight)
    elif args.loss == "hyperspheric":
        assert args.use_custom_hsphere, "--use-custom-hsphere should be set when loss is hyperspheric. This is to maintain consistency with prev experiments"
        # This has a separate computational function
        return None
    elif args.loss == "masked_margin_sm":
        raise NotImplementedError()
    else:
        raise ValueError(f"Could not recognize loss function {args.loss}")


def get_loss_framework(args):
    if args.loss == "hyperspheric":
        assert args.use_custom_hsphere, "--use-custom-hsphere should be set when loss is hyperspheric. This is to maintain consistency with prev experiments"
        return custom_hsphere_loss_computation
    else:
        return multiview_contrastive_computation


def prepare_models(image_model, audio_models, device, args):
    if not isinstance(image_model, torch.nn.DataParallel) and not args.use_cpu and not args.dummy_data:
        image_model = nn.DataParallel(image_model)

    image_model = image_model.to(device)

    for lang_id in audio_models.keys():
        if not isinstance(audio_models[lang_id], torch.nn.DataParallel) and not args.use_cpu and not args.dummy_data:
            audio_models[lang_id] = nn.DataParallel(audio_models[lang_id])
        audio_models[lang_id] = audio_models[lang_id].to(device)

    return image_model, audio_models


def setup_optimizer(image_model, audio_models, args):
    # Gather trainable parameters
    audio_trainables = list()
    for lang_id in audio_models.keys():
        audio_trainables.extend([p for p in audio_models[lang_id].parameters() if p.requires_grad])
    if args.freeze_image_model:
        image_trainables = [p for n, p in image_model.named_parameters() \
                            if n.startswith('embedder')]
    else:
        image_trainables = [p for p in image_model.parameters() if p.requires_grad]
    trainables = audio_trainables + image_trainables
    print('TRAINER: Total %d trainable parameters' % len(trainables))

    # Instantiate optimizer type
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(trainables, args.lr,
                                 momentum=args.momentum,
                                 weight_decay=args.weight_decay)
        print('TRAINER: Using %s optimizer w/ lr: %f, momentum: %d, weight_decay: %f' % (args.optim, args.lr, args.momentum, args.weight_decay))
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(trainables, args.lr,
                                weight_decay=args.weight_decay)
        print('TRAINER: Using %s optimizer w/ lr: %f, weight_decay: %f' % (args.optim, args.lr, args.weight_decay))
    else:
        raise ValueError('Optimizer %s is not supported' % args.optim)

    return optimizer, trainables
