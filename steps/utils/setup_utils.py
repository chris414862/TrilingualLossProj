import torch
import torch.nn as nn
from losses.funcs.InfoNCE import InfoNCE
from losses.funcs.triplet import TripletLoss
from losses.frameworks.hyperspheric import custom_hsphere_loss_computation
from losses.frameworks.multiview_coding import multiview_contrastive_computation
from models.CommonLayers import BYOL_Layer
from .aux_nets_utils import prep_byol_aux_nets


def get_loss_function(args):
    if args.loss == "info_nce":
        return InfoNCE(temperature=args.temperature, sim_measure=args.sim_measure)
    elif args.loss == 'masked_margin_sm':
        return InfoNCE(temperature=args.temperature, sim_measure=args.sim_measure, masked_margin=True)
    elif args.loss == 'sched_masked_margin_sm':
        return InfoNCE(temperature=args.temperature, sim_measure=args.sim_measure, masked_margin=True, scheduler=True)
    elif args.loss == 'adapt_masked_margin_sm':
        return InfoNCE(temperature=args.temperature, sim_measure=args.sim_measure, masked_margin=True, scheduler=True, adaptive=True)

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


def get_byol_layer_sizes(args):
    return [int(num) for num in args.byol_layer_sizes.split(",")]

def get_final_layer_size(view_id, args):
    if view_id == "img":
        return int(args.edim)
    else:
        return int(args.layer_widths.split(",")[-1])


def prepare_models(image_model, audio_models, device, args):
    if not isinstance(image_model, torch.nn.DataParallel) and not args.use_cpu and not args.dummy_data:
        image_model = nn.DataParallel(image_model)

    image_model = image_model.to(device)

    for lang_id in audio_models.keys():
        if not isinstance(audio_models[lang_id], torch.nn.DataParallel) and not args.use_cpu and not args.dummy_data:
            audio_models[lang_id] = nn.DataParallel(audio_models[lang_id])
        audio_models[lang_id] = audio_models[lang_id].to(device)

    # Prepare auxillary layers/networks
    if args.loss == "byol":
        view_ids = ["img"]+[k for k in audio_models.keys()]
        aux_nets = prep_byol_aux_nets(view_ids)
        
    else:
        aux_nets = None

    return image_model, audio_models, aux_nets

class Tracker():
    def __init__(self, layers_dict, params_dict, layers_tot, params_tot):
        self.layers_dict = layers_dict
        self.params_dict = params_dict
        self.layers_tot = layers_tot
        self.params_tot = params_tot

    def __call__(self,values, key):

        tmp = len(values)
        self.layers_dict[key] = tmp
        self.layers_tot += tmp
        
        tmp = sum([torch.prod(torch.tensor(p.shape)) for p in values])
        self.params_dict[key] = tmp
        self.params_tot += tmp

        return self.layers_dict, self.params_dict, self.layers_tot, self.params_tot




def get_all_model_params(audio_models, image_model, aux_nets, args):
    audio_trainables = list()
    trainable_layers_dict = dict()  
    trainable_params_dict = dict() 
    tot_trainable_layers = 0
    tot_trainable_params = 0
    tracker = Tracker(trainable_layers_dict, trainable_params_dict, tot_trainable_layers, tot_trainable_params)

    
    # Audio encoders
    if args.shared_audio_encoder == "na":
        for lang_id in audio_models.keys():
            t_params = [p for p in audio_models[lang_id].parameters() if p.requires_grad]
            audio_trainables.extend(t_params)

            (trainable_layers_dict, trainable_params_dict, 
                    tot_trainable_layers, tot_trainable_params) = tracker(t_params, lang_id)

            # tmp = sum([torch.prod(torch.tensor(p.shape)) for p in t_params])
            # num_trainable_params_dict[lang_id] = tmp
            # tot_trainable_params += tmp

    else:
        # Get params from only one (of any) audio models. They are all the same model
        any_key = [k for k in audio_models.keys()][0]
        t_params = [p for p in audio_models[any_key].parameters() if p.requires_grad]
        audio_trainables.extend(t_params)

        (trainable_layers_dict, trainable_params_dict, 
                tot_trainable_layers, tot_trainable_params) = tracker(t_params, "shared_enc")
        # tmp = len(t_params)
        # len_trainable_layers_dict["shared_enc"] = len(t_params)
        # tot_trainable_layers += tmp
        # tmp = sum([torch.prod(torch.tensor(p.shape)) for p in t_params])
        # num_trainable_params_dict["shared_enc"] = tmp
        # tot_trainable_params += tmp


    # Image encoder
    if args.freeze_image_model:
        image_trainables = [p for n, p in image_model.named_parameters() \
                            if n.startswith('embedder')]
    else:
        image_trainables = [p for p in image_model.parameters() if p.requires_grad]


    (trainable_layers_dict, trainable_params_dict, 
            tot_trainable_layers, tot_trainable_params) = tracker(image_trainables, "image")
    # tmp = len(image_trainables)
    # len_trainable_layers_dict["shared_enc"] = tmp
    # tot_trainable_layers += tmp
    #
    # tmp = sum([torch.prod(torch.tensor(p.shape)) for p in image_trainables])
    # num_trainable_params_dict["shared_enc"] = tmp
    # tot_trainable_params += tmp



    # Auxillary networks TODO: Finish this later
    if aux_nets is not None:
        pass

    trainables = audio_trainables + image_trainables
    return (trainables, tot_trainable_layers, tot_trainable_params, 
                trainable_layers_dict, trainable_params_dict)

def get_partitioned_model_params(audio_models, image_model, aux_nets, args):
    """
        Returns a dict of model params to create an optimizer with separate param groups.
        Also returns global info about all params.

        Returns:
            trainables - (list): 
                A list of dicts. Each dict contains the keys: "view_id" and "params"
            num_trainable_layers - (int):
                Total number of modules/layers in all models
            num_trainable_params - (int):
                Total number of parameters/floats in all models
    """
    trainables = list()
    num_trainable_layers = 0
    num_trainable_params = 0

    # Language encoders 
    for lang_id in audio_models.keys():
        lang_trainables = [p for p in audio_models[lang_id].parameters() if p.requires_grad]
        num_trainable_layers += len(lang_trainables)
        num_trainable_params += sum([torch.prod(torch.tensor(p.shape)) for p in lang_trainables])
        trainables.append({"params": lang_trainables, "view_id":lang_id})

    # Image model parameters
    if args.freeze_image_model:
        img_trainables = [p for n, p in image_model.named_parameters() \
                            if n.startswith('embedder')]
    else:
        img_trainables = [p for p in image_model.parameters() if p.requires_grad]

    trainables.append({"params":img_trainables, "view_id":"img"})
    num_trainable_layers += len(img_trainables)
    num_trainable_params += sum([torch.prod(torch.tensor(p.shape)) for p in img_trainables])

    # Auxillary networks (e.g. projection and prediction layers)
    for view_type, view_type_dict in aux_nets.items():
        for aux_net_view_id, aux_net_view_dict in view_type_dict.items():
            # get view's param list in trainables
            views_trainables = [d["params"] for d in trainables if d["view_id"] == aux_net_view_id][0]

            # add view's params from auxillary networks to list
            assert type(aux_net_view_dict["params"]) == list
            views_trainables.extend(aux_net_view_dict["params"])




    if args.loss == "byol":
        # Freeze target view to do ema update. Keeping the param group in the optimizer for consistency 
        target_param_group = [param_group for param_group in trainables if param_group["view_id"] == args.byol_target_view][0]
        for param_tensor in target_param_group["params"]:
            param_tensor.requires_grad = False


    return trainables, num_trainable_layers, num_trainable_params


def setup_optimizer(image_model, audio_models, aux_nets, args):
    # Gather trainable parameters
    if args.loss != "byol":
        # lump all paramters together for optimization
        (trainables, tot_trainable_layers, tot_trainable_params,
                trainable_layers_dict, trainable_params_dict) = get_all_model_params(audio_models, image_model, aux_nets, args)
        print(f'TRAINER: Total trainable layers/matrices: {tot_trainable_layers} Total trainable params: {tot_trainable_params:,}')
        for key in trainable_layers_dict.keys():
            print(f'\t\t{key:<12}-- total layers/matrices: {trainable_layers_dict[key]:6,} trainable parameters: {int(trainable_params_dict[key].item()):,}')

    else: 
        # Create param groups. trainables is a list of dicts. 
        trainables, len_trainable_layers, len_trainable_params = get_partitioned_model_params(audio_models, image_model, aux_nets,args)
        print(f'TRAINER: Total {len_trainable_layers} trainable layers/matrices. {len_trainable_params:,} trainable parameters')
        for d in trainables:
            len_trainable_layers = len(d["params"])
            len_trainable_params= sum([torch.prod(torch.tensor(p.shape)) for p in  d["params"]])
            print(f'TRAINER: {d["view_id"]} total {len_trainable_layers} trainable layers/matrices. {len_trainable_params:,} trainable parameters')

    # Instantiate optimizer type
    if args.optim == 'sgd':
        optim_class = torch.optim.SGD 
        # optimizer = torch.optim.SGD(trainables, args.lr,
        #                          momentum=args.momentum,
        #                          weight_decay=args.weight_decay)
        kwargs = {"momentum":args.momentum}
        print('TRAINER: Using %s optimizer w/ lr: %f, momentum: %d, weight_decay: %f' % (args.optim, args.lr, args.momentum, args.weight_decay))
    elif args.optim == 'adam':
        optim_class = torch.optim.Adam

        # optimizer = torch.optim.Adam(trainables, args.lr,
                                # weight_decay=args.weight_decay)
        kwargs = {}
        print('TRAINER: Using %s optimizer w/ lr: %f, weight_decay: %f' % (args.optim, args.lr, args.weight_decay))
    else:
        raise ValueError('Optimizer %s is not supported' % args.optim)

    optimizer = optim_class(trainables, args.lr, weight_decay=args.weight_decay, **kwargs)
    return optimizer, trainables



