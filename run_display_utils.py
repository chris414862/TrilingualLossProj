import numpy as np
import torch
import torch.nn as nn
import re
from torch.autograd import Variable
from collections import OrderedDict, defaultdict, Counter
from torch.profiler import profile, record_function, ProfilerActivity
from itertools import zip_longest
import pprint
import sys


GLOBAL_LAYER_IDX = 0

def get_longest_str(l):
    """
        Get the string length of the element with the longest string representation
    """
    max_len = 0
    obj_w_max = None
    for thing in l:
        string = len(str(thing))
        if string > max_len:
            max_len = string
            obj_w_max = thing

    return max_len

def raise_nesting(nested:list):
    """
        This is a super simple function, but a visualization can help prevent bugs. This function
        takes a nesting that is a list of lists. The each inner list could have differing lengths.
        This function removes the nesting by raising the last level up one.
        
        Example:
        input:
        [
            [
                item1
            ],
            [
                item2,
                item3
            ]
        ]
        output:
        [
            item1,
            item2,
            item3.
        ]
    """
    unnested = [item for inner_list in nested for item in inner_list]
    return unnested


def format_layer_names(layer, summary):
    """
       Returns a list of 1 or more names for one layer. If the layer has more than one input/output 
       it will be given multiple lines in the summary printout. This function controls how the label is changed 
       for the subsequent names of the layer in the summary.
    """
    # Format for layer name
    new_name_format = "{prepend}{name}"
    # Format for excess input/output marker
    prepend_format = "({var})"
    # Function to get marker from input/output iteration number
    get_marker_from_iter = lambda x: "" if x == 0 else prepend_format.format(var=str(x+1))

    inps, outs = summary[layer]["input_shapes"], summary[layer]["output_shapes"]
    names = []
    for i, (inp, out) in enumerate(zip_longest(inps, outs)):
        marker = get_marker_from_iter(i)
        names.append(new_name_format.format(prepend=marker, name=layer))
    return names


def get_col_lengths(summary, column_headers):
    """
        Gets the appropriate lengths for each column. Does this by calculating
        the max string representation length in each column
    """
    # column_headers = {"name":"Layer Type", "input":"Input Dims", "output":"Output Dims", "params":"#Params"}
    # max layer name length
    layer_names = [layer for layer in list(summary.keys()) ]
    names = [name for layer_name in layer_names for name in format_layer_names(layer_name, summary)]
    names = names+[column_headers["layer_id"]]
    name_cols = get_longest_str(names)

    # max input shape string length
    input_shapes = [summary[layer]["input_shapes"] for layer in summary.keys()]
    # nested_list structure: [num_layers, num_layer_inputs, shapes]
    # change to: -->[all_inputs, shapes ]
    input_shapes = raise_nesting(input_shapes)
    input_shapes = input_shapes+[column_headers["input_shapes"]]
    in_cols = get_longest_str(input_shapes)

    # max output shape string length
    output_shapes = [summary[layer]["output_shapes"] for layer in summary.keys()]
    output_shapes = raise_nesting(output_shapes)
    output_shapes = output_shapes+[column_headers["output_shapes"]]
    out_cols = get_longest_str(output_shapes)

    # max num_params string length
    nparams = [f"{summary[layer]['nb_params']}" for layer in summary.keys()]
    nparams = nparams+[column_headers["params"]]
    np_cols = get_longest_str(nparams)

    return name_cols, in_cols, out_cols, np_cols


def print_header_line(format_str, col_lengths, model_name=""):
    name_cols, in_cols, out_cols, np_cols = col_lengths
    header_line = format_str.format(name= "Layer Type",
                                 name_cols=name_cols,
                                 inp="Input Shape",
                                 in_cols=in_cols,
                                 out="Output Shape",
                                 out_cols=out_cols,
                                 params="Param #",
                                 np_cols = np_cols)
    print("-"*len(header_line))
    print(f"{model_name:^{len(header_line)}}")
    print("-"*len(header_line))
    print(header_line)
    print("="*len(header_line))
    return header_line


def get_output_shapes(summary, layer):
    """
    Some modules have multiple inputs and/or multiple outputs to their forward method.
    This function organizes them into first and extras.     
    """
    extra_shapes = None
    first_in, first_out, extra_in, extra_out = None, None, [], []
    for i, (input_shape, out_shape) in enumerate(zip_longest(summary[layer]["input_shapes"], summary[layer]["output_shapes"])):
        if i == 0:
            first_in, first_out = input_shape, out_shape
        else:
            extra_in.append(input_shape)
            extra_out.append(out_shape)

    return first_in, first_out, extra_in, extra_out


def print_info_line(summary, layer, format_str, col_lengths):
    """
    The module/layer info will only be printed with the first input/output pair.
    If more than one input/output tensor present, only the module/layer name
    will be displayed.
    """
    name_cols, in_cols, out_cols, np_cols = col_lengths

    # Organize input and output shapes
    first_in, first_out, extra_in, extra_out = get_output_shapes(summary, layer)

    # Get layer names (could be more than one if multiple inputs or outputs present
    names = format_layer_names(layer, summary)
    name = names.pop(0)
    extra_names = names
    line_new = format_str.format(name= name,
                                 name_cols=name_cols,
                                 inp=str(first_in),
                                 in_cols=in_cols,
                                 out=str(first_out),
                                 out_cols=out_cols,
                                 params=summary[layer]["nb_params"],
                                 np_cols = np_cols)

    print(line_new)
    # Handle the case where multiple lines are necessary
    print_excess_info(extra_in, extra_out, extra_names, format_str, col_lengths)


def print_excess_info(extra_ins, extra_outs, extra_names, format_str, col_lengths):
    """
    Prints the excess lines for module/layers with more than one input/output tensor
    """
    name_cols, in_cols, out_cols, np_cols = col_lengths
    for i, (inp_shape, out_shape, extra_name) in enumerate(zip_longest(extra_ins, extra_outs, extra_names)):
        line_new = format_str.format(name= extra_name,
                                     name_cols=name_cols,
                                     inp=str(inp_shape),
                                     in_cols=in_cols,
                                     out=str(out_shape),
                                     out_cols=out_cols,
                                     params="-",
                                     np_cols = np_cols)
        print(line_new)




def get_all_keys(summaries, seen=set(), ignore=[]):
    order = list()
    for summary in summaries:
        for key in summary.keys():
            if key in ignore:
                continue
            if key != "submods":
                if not key in seen:
                    seen.add(key)
                    order.append(key)
            else: #submods
                order.extend(get_all_keys(summary["submods"], seen, ignore))

    return order


def prepend_and_stringify(nested_l, prefix, cols, prefix_cols_fmt_strs:dict, depth_col_idx=3):
    new_nested_l = []
    for l in nested_l:
        new_l = []
        for i, item in enumerate(l):
            if cols[i] in prefix_cols_fmt_strs.keys():
                item = prefix*int(l[depth_col_idx])+f"{item:{prefix_cols_fmt_strs[cols[i]]}}"
            new_l.append(item)
        new_nested_l.append(new_l)

    return new_nested_l


def convert_nested_summary(submods_summaries, depth=0, col_names=[]):
    if depth==0:
        col_names:list = get_all_keys(submods_summaries, seen=set(), ignore=["trainable"])

    rows = []
    for i, submod_summary in enumerate(submods_summaries):
        row = []
        row_depth = submod_summary["depth"]
        for col in col_names:
            if not col in submod_summary.keys():
                if col == "parameters":
                    row.append(init_param_dict())#{"trainable": Counter(), "frozen": Counter(), "total":0})
                else:
                    row.append("N/A")
            else:
                row.append(submod_summary[col])

        rows.append(row)
        if "submods" in submod_summary.keys():
            nested_rows, _ = convert_nested_summary(submod_summary["submods"], depth=depth+1, col_names=col_names)
            rows.extend(nested_rows)

    return rows, col_names

def merge_cols(summary_rows, col_names):
    new_rows = []
    for row in summary_rows:
        # print(row)
        new_row = []
        mod_class = row.pop(1)
        layer_num = row.pop(1)
        new_row.append("["+str(layer_num)+"]"+str(row[0]) + "("+str(mod_class) +")")
        new_row.extend(row[1:])
        new_rows.append(new_row)
    col_names.pop(1)
    col_names.pop(1)
    col_names[0] = "layer_id"
    return new_rows, col_names


def aggregate(parameter_dict, accum):
    # accum["trainable"].update(parameter_dict["trainable"])
    # accum["frozen"].update(parameter_dict["frozen"])
    accum["total"] += parameter_dict.get("total", 0)
    accum["num_bytes"] += parameter_dict.get("num_bytes", 0)
    accum["trainable"] += parameter_dict.get("trainable", 0)
    return accum




def add_cumulative_params(summary_rows, col_names, depth_col_idx=3):

    prev_depth = 0
    # depth_col_idx = 1
    param_col_idx = -1
    accumulating_idxs = []
    # stop_idx = 20
    for i in range(len(summary_rows)):

        # if i == stop_idx:
        #     break

        # print(i, summary_rows[i])
        cur_depth = int(summary_rows[i][depth_col_idx])
        if cur_depth > prev_depth:
            accumulating_idxs.append(i-1)
        elif cur_depth < prev_depth:
            dif = prev_depth - cur_depth
            for j in range(dif):
                accumulating_idxs.pop(-1)

        # print(i, "Accumulating:",accumulating_idxs)
        for acc_idx in accumulating_idxs:
            # print(i, "Accumulating:", accumulating_idxs)
            summary_rows[acc_idx][-1] = aggregate(summary_rows[i][param_col_idx],
                                                  summary_rows[acc_idx][param_col_idx])

        prev_depth = cur_depth

    return summary_rows#[:stop_idx]


def convert_to_dict_of_dicts(summary_rows, col_names):
    new_summary = OrderedDict()
    first_row_len = len(summary_rows[0])
    for i, row in enumerate(summary_rows):
        assert len(row) == first_row_len
        od = OrderedDict()
        layer_id = None
        for item, col_name in zip(row, col_names):
            if col_name == "layer_id":
                layer_id = item
                continue
            od[col_name] = item

        # print(layer_id)
        # print(od)
        new_summary[layer_id] = od

    return new_summary


def print_final_summary(input_size, batch_size, total_output, total_params,
                        header_line, trainable_params, total_param_size,
                        total_buffer_size, device, mem_stats):
    """
        Prints aggregate statistics
    """

    # Estimate model's size in memory
    # assumes 4 bytes/number (float on cuda).
    total_input_size = np.prod(input_size) * batch_size * 4. / (2**20)
    total_activations_size = total_output * 4. / (2**20)
    total_param_size = total_param_size /(2**20)
    total_buffer_size = total_buffer_size/(2**20)
    total_size =  total_activations_size + total_input_size + total_param_size

    # Number of Parameter
    # longest label
    padding =len("Non-trainable params: ")
    label_fmt = "<"+str(padding)
    data_fmt = ","
    print("="*len(header_line))
    print(f'{"Total Number of Parameters":^{len(header_line)}}')
    print("-"*len(header_line))
    print(f"{'Total params: ':{label_fmt}}{total_params:{data_fmt}}")
    print(f"{'Trainable params: ':{label_fmt}}{trainable_params:{data_fmt}}")
    print(f"{'Non-trainable params: ':{label_fmt}}{total_params - trainable_params:{data_fmt}}")

    # Size and
    # longest label
    padding =len('Size of layer activations (MB): ')
    label_fmt = "<"+str(padding)
    data_fmt = "0,.2f"
    print("-"*len(header_line))
    print(f'{"Parameter and Activation Sizes":^{len(header_line)}}')
    print("-"*len(header_line))

    print(f"{'Input size (MB): ':{label_fmt}}{total_input_size:{data_fmt}}")
    print(f"{'Size of layer activations (MB): ':{label_fmt}}{total_activations_size:{data_fmt}}")
    print(f"{'Params size (MB): ':{label_fmt}}{total_param_size:{data_fmt}}")
    print(f"{'Buffer size (MB): ':{label_fmt}}{total_buffer_size:{data_fmt}}")
    print(f"{'Estimated total size (MB): ':{label_fmt}}{total_size:{data_fmt}}")

    padding =len("Max memory allocated w/ Adam (MB): ")
    label_fmt = "<"+str(padding)
    print("-"*len(header_line))
    print(f'{"Actual Memory Usage During Mock Training":^{len(header_line)}}')
    print("-"*len(header_line))
    print("----SGD")
    print(f"{'Max memory allocated w/ SGD (MB): ':{label_fmt}}{mem_stats['sgd']['max_mem_alloc']:{data_fmt}}")
    print(f"{'Max memory reserved w/ SGD (MB): ':{label_fmt}}{mem_stats['sgd']['max_mem_res']:{data_fmt}}")
    print("----Adam")
    print(f"{'Max memory allocated w/ Adam (MB): ':{label_fmt}}{mem_stats['adam']['max_mem_alloc']:{data_fmt}}")
    print(f"{'Max memory reserved w/ Adam (MB): ':{label_fmt}}{mem_stats['adam']['max_mem_res']:{data_fmt}}")
    print("-"*len(header_line), flush=True)

def prep_summary_info(summaries, depth_prefix):
    # Convert nested summary to a single level
    summary_rows, col_names = convert_nested_summary(summaries)

    # Accumulate submodule info for each "custom"/container module
    summary_rows = add_cumulative_params(summary_rows, col_names)

    # Add prefix to var_name indicate the depth of the module in the hierarchy
    summary_rows = prepend_and_stringify(summary_rows, depth_prefix, col_names, prefix_cols_fmt_strs={"var_name":""}, depth_col_idx=3)

    # Create the "layer_id" column from information in other columns
    summary_rows, col_names = merge_cols(summary_rows, col_names)

    # Add column for total parameters
    # Last element in each row should be the parameters dict
    for row in summary_rows:
        row.append(row[-1]["total"])
    col_names.append("nb_params")

    # Add prefix to var_name indicate the depth of the module in the hierarchy
    summary_rows = prepend_and_stringify(summary_rows, depth_prefix, col_names, prefix_cols_fmt_strs={"nb_params":","}, depth_col_idx=1)

    # Now that everything is flattened, convert to structure
    # expected by the rest of the code
    summary = convert_to_dict_of_dicts(summary_rows, col_names)
    return summary


def print_model_info(model,
                     submods_summaries,
                     input_size,
                     batch_size,
                     num_spaces=3,
                     column_headers=None,
                     depth_prefix="--",
                     device="cpu",
                     mem_stats=None,
                     model_name=None):
    """
        Top level method in the heirarchy for controlling printouts
    """
    if model_name is None:
        model_class_name = str(model.__class__).split(".")[-1].split("'")[0]
    else:
        model_class_name = str(model.__class__).split(".")[-1].split("'")[0]
        model_class_name = model_name + " ("+model_class_name+")"
    summary = prep_summary_info(submods_summaries["submods"], depth_prefix=depth_prefix)

    # Get lengths to format columns
    col_lengths = get_col_lengths(summary, column_headers)
    spacing = " "*num_spaces
    format_str = "{name:<{name_cols}}"+spacing+"{inp:<{in_cols}}"+spacing+"{out:<{out_cols}}"+spacing+"{params:<{np_cols}}"+spacing
    header_line = print_header_line(format_str, col_lengths, model_class_name)
    total_params = 0
    total_output = 0
    trainable_params = 0
    total_param_size = 0
    total_buffer_size = 0
    for i, (layer_name, layer_info_dict) in enumerate(summary.items()):
        # Print info for each layer
        print_info_line(summary, layer_name, format_str, col_lengths)

        # Aggregate info for total summary
        total_params += layer_info_dict["parameters"]["total"]

        if layer_info_dict["is_standard"] and layer_info_dict["parameters"]["total"] > 0: # Don't count the container layers, they are already included implicitly
            total_output += np.prod(sum([np.prod(shape) for shape in layer_info_dict["output_shapes"]]))


        trainable_params += layer_info_dict["parameters"]["trainable"]
        total_param_size += layer_info_dict["parameters"]["num_bytes"]
        total_buffer_size += layer_info_dict["parameters"]["num_buffer_bytes"]

    submods = [sm for sm in model.modules()]
    submods = submods[1:]
    print_final_summary(input_size, batch_size, total_output, total_params,
                        header_line, trainable_params, total_param_size,
                        total_buffer_size, device, mem_stats)


def remove_layers(summary):
    """
        This is called when print_major_layers_only is True.

        Removes layers with no weights, like activation layers. Retain layers that manipulate
        shapes, otherwise reading the model summary can be confusing.
    """
    new_summary = OrderedDict()
    prev_out_size = None
    for layer in summary.keys():
        if summary[layer]["nb_params"] <= 0.0:
            input_shapes = summary[layer]["input_shapes"]
            out_shapes = summary[layer]["output_shapes"]

            # Check for different shapes btw previous output and current input, then curr input w/ curr output
            should_keep = False
            for in_shape in input_shapes:
                for prev_out_shape in prev_out_shapes:
                    if in_shape != prev_out_shape:
                        should_keep = True
                for out_shape in out_shapes:
                    if in_shape == out_shape:
                        should_keep = True

            if not should_keep:
                continue

        prev_out_shapes = summary[layer]["output_shapes"]
        new_summary[layer] = summary[layer]

    return new_summary



def init_param_dict():
    return {"total":0, "num_bytes": 0, "num_buffer_bytes":0, "trainable":0, "DTs":set()}


def init_summary_info(module, parent_mod_summary, input_tuple, mod_var_name, depth):
    """
        Creates and stores information about a pytorch module. This
        function stores the information that does not need specialized 
        logic to handle. 
    """
    global GLOBAL_LAYER_IDX
    classname = str(module.__class__).split(".")[-1].split("'")[0]
    summary = OrderedDict()
    # Add identifier when part of a nn.Sequential set of layers
    if "module_class" in parent_mod_summary.keys() and \
            parent_mod_summary["module_class"] == "Sequential":
        var_name = "seq_"+parent_mod_summary["var_name"]+"-"+str(int(mod_var_name)+1)
    else:
        var_name = mod_var_name
    summary["var_name"] = var_name
    summary["module_class"] = classname
    summary["layer_number"] = GLOBAL_LAYER_IDX
    summary["depth"] = depth
    # Input is stored as tuples containing posibly more than one tensor
    # For each layer store as list of lists: [[dim1, dim2,...], [dim1,dim2,...],...]
    summary["input_shapes"] = [list(sub_input.size()) for sub_input in input_tuple if not sub_input is None]
    GLOBAL_LAYER_IDX += 1
    return summary

def hook_wrapper(hook_type, mod_var_name="", depth=-1, parent_mod_summary=None, handles=None):
    """
    This wrapper allows the hooks to have access to extra information outside of the given parameters.
    """

    ############### Embedded closure ###############
    def standard_module_hook(module:nn.Module, input_tuple, output_tuple):
        """
        This function should be sent to the nn.Module.register_forward_hook(...) function. It will then
        be called after the forward function of the registered nn.Module. We track sizing information
        and store it in the current parent module's summary (in the 'submods' field). 
        """
        classname = str(module.__class__).split(".")[-1].split("'")[0]
        summary = init_summary_info(module, parent_mod_summary, input_tuple, mod_var_name, depth)
        summary["is_standard"] = True

        # Output may or may not be a tuple
        if isinstance(output_tuple, (tuple, list)):
            summary["output_shapes"] = [list(o.size()) for o in output_tuple]
        elif isinstance(output_tuple, torch.Tensor):
            summary["output_shapes"] = [list(output_tuple.size())]
        else:
            raise ValueError("Expected forward output to be either torch.Tensor, tuple, or list")

        # Gather parameter info
        summary["parameters"] = init_param_dict()

        for tens in module.parameters(recurse=False):
            dt = tens.dtype
            trainable = tens.requires_grad
            num_params = tens.nelement()
            summary["parameters"]["total"] += num_params
            summary["parameters"]["DTs"].add(tens.dtype)
            if trainable:
                summary["parameters"]["trainable"] += num_params

            summary["parameters"]["num_bytes"] += num_params*tens.element_size()

        for tens in module.buffers(recurse=False):
            summary["parameters"]["num_buffer_bytes"] += tens.nelement()*tens.element_size()

        parent_mod_summary["submods"].append(summary)

    ############### New embedded closure ###############
    def custom_module_pre_hook(module:nn.Module, input_tuple):
        """
            This is used for user defined layers. These will typically contain several 
            "standard" nn.Module layers such as nn.Linear. These are treated as a container
            of the standard nn.Module layers. As such, nn.Sequential layers are also 
            handled with this method. I refer to these containers as "custom" to 
            differentiate them from the "standard" modules.
            
            We register a new set of hooks for each "child" module and record their 
            information hierarchically to preserve structural information. This is
            done BEFORE the forward method is called, so this hook must be registered 
            with register_forward_pre_hook. 

            This function does not store parameter info. Those will be aggregated later 
            from the "submods". This is to avoid counting parameters that don't actually get
            used in the forward method (either through lazy coding or specialized logic).
        """
        classname = str(module.__class__).split(".")[-1].split("'")[0]
        new_parent_mod_summary = init_summary_info(module, parent_mod_summary, input_tuple, mod_var_name, depth)
        # if classname == "MultiheadAttention":
        #     print("Found multihead")
        #     print([p for p in module.parameters()])
        #     print([c for c in module.children()])
        new_parent_mod_summary["is_standard"] = False
        new_parent_mod_summary["submods"] = list()
        parent_mod_summary["submods"].append(new_parent_mod_summary)
        register_hooks(module, new_parent_mod_summary, depth+1, handles=handles)


    ############### New embedded closure ###############
    def custom_module_post_hook(module:nn.Module, input_tuple, output_tuple):
        """
           This hook merely stores the output size of the custom module that was 
           summarized by custom_module_pre_hook. The output_tuple is only available
           after the forward method is called.
        """
        # Aliasing to add clarity.
        # In this hook we are not adding to the parent module's summary list
        mod_summary = parent_mod_summary
        if isinstance(output_tuple, (tuple, list)):
            mod_summary["output_shapes"] = [list(o.size()) for o in output_tuple]
        elif isinstance(output_tuple, torch.Tensor):
            mod_summary["output_shapes"] = [list(output_tuple.size())]
        else:
            raise ValueError("Expected forward output to be either torch.Tensor, tuple, or list")
    ############### End of embedded closures ###############

    if hook_type == "standard":
        return standard_module_hook
    elif hook_type == "custom_pre":
        return custom_module_pre_hook
    elif hook_type == "custom_post":
        return custom_module_post_hook
    else:
        return None


def register_hooks(parent_module, parent_mod_summary, depth=0, handles=None):
    """
    Registers hooks with every module on a given level of the model's module hierarchy 
    using the nn.Module.children() iterator. 
    """
    if handles is None:
        handles = list()

    # Only functions calling this one after the initial call will be "custom" modules
    # Use the special hook
    if depth > 0:
        handle = parent_module.register_forward_hook(hook_wrapper("custom_post", parent_mod_summary=parent_mod_summary))
        handles.append(handle)

    # if parent_mod_summary.get("module_class", None) is not None and parent_mod_summary['module_class'] == 'MultiheadAttention':
    #     print("!!!!!!!!!!!1")
    for i, (sm_name,submod) in enumerate(parent_module.named_children()):
        # if classname == "MultiheadAttention":

        submod_qualified_classname = str(submod.__class__).split("'")[1].strip()
        # if parent_mod_summary.get("module_class", None) is not None and parent_mod_summary['module_class'] == 'MultiheadAttention':
        #     print("submod", submod_qualified_classname)
        #     print("sm_name", sm_name)
        #     print([p for p in submod.parameters()])
        #     print([c for c in submod.children()])

        if (re.match(r"^torch\.nn\.modules.*", submod_qualified_classname) is not None) \
                and (len([c for c in submod.children() if re.match(r"^.*NonDynamicallyQuantizableLinear", str(c.__class__)) is None]) == 0 \
                and len([p for p in submod.parameters()]) > 0):

            if parent_mod_summary.get("module_class", None) is not None and parent_mod_summary['module_class'] == 'MultiheadAttention':
                print("About to register standard hook")

                # and submod_qualified_classname != "torch.nn.modules.container.Sequential")\
            handle = submod.register_forward_hook(
                                                    hook_wrapper(hook_type="standard",
                                                                 mod_var_name=sm_name,
                                                                 depth=depth,
                                                                 parent_mod_summary=parent_mod_summary))
        else:  #Custom module
            handle = submod.register_forward_pre_hook(
                                                    hook_wrapper(hook_type="custom_pre",
                                                                 mod_var_name=sm_name,
                                                                 depth=depth,
                                                                 parent_mod_summary=parent_mod_summary, handles=handles))

        handles.append(handle)

    return handles

def get_stats_from_training_loop(model, batch_size, input_shape, dtype, device, optimizer=None, loop_iters=5):
    out, x = None, None
    for i in range(loop_iters):
        print(i+1, end="\r", flush=True)
        x = torch.rand((batch_size, *input_shape)).type(dtype)
        out = model(x)
        optimizer.zero_grad()
        l = (torch.sum(out[0])*0)
        l.backward()
        optimizer.step()
    del out, x
    mem_stats = dict()
    mem_stats["max_mem_alloc"] = torch.cuda.max_memory_allocated(device)/(2**20)
    mem_stats["max_mem_res"] = torch.cuda.max_memory_reserved(device)/(2**20)
    return mem_stats


def get_memory_stats(model, batch_size, input_shape, dtype, device):
    # Run mock training loop iterations to
    # gather memory info. A single forward pass is ok for SGD
    mem_stats = dict()
    trainables = [p for p in model.parameters() if p.requires_grad]
    torch.cuda.reset_peak_memory_stats()
    optimizer = torch.optim.SGD(trainables, lr=.000, # Don't actually update the weights
                            weight_decay=.000)
    mem_stats["sgd"] = get_stats_from_training_loop(model,
                                                    batch_size,
                                                    input_shape,
                                                    dtype,
                                                    device,
                                                    optimizer=optimizer,
                                                    loop_iters=1)
    del optimizer

    # Run another mock training loop.
    # Adam seems to give consistent memory readings after 5 iterations
    torch.cuda.reset_peak_memory_stats()
    optimizer = torch.optim.Adam(trainables, lr=.000, # Don't actually update the weights
                            weight_decay=.000)
    mem_stats["adam"] = get_stats_from_training_loop(model,
                                                    batch_size,
                                                    input_shape,
                                                    dtype,
                                                    device,
                                                    optimizer=optimizer,
                                                    loop_iters=5)
    del optimizer
    return mem_stats




def my_model_summary(model,
                     input_shape,
                     batch_size=1,
                     device="cuda",
                     print_parametarized_layers_only=False,
                     spacing=3,
                     column_headers = {  # Change values to adjust column headers
                        "layer_id"      : "Layer ID",
                        "input_shapes"  : "Input Dims",
                        "output_shapes" : "Output Dims",
                        "params"        : "#Params"
                        },
                     mock_train4_mem_stats=False,
                     model_name=None):

    global GLOBAL_LAYER_IDX
    GLOBAL_LAYER_IDX = 0

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"
    assert device == "cpu" or torch.cuda.is_available()

    model = model.to(device)
    if device == "cuda" and torch.cuda.is_available() and next(model.parameters()).is_cuda:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # Get memory stats for SGD and Adam optimization (Adam requires more memory)
    if mock_train4_mem_stats:
        mem_stats = get_memory_stats(model, batch_size, input_shape, dtype, device)
    else:
        mem_stats = {"sgd":{"max_mem_alloc":0.0, "max_mem_res":0.0},
                     "adam":{"max_mem_alloc":0.0, "max_mem_res":0.0}}

    # create dict to hold properties
    mod_summary = OrderedDict()

    # Top layer summary only holds submods list
    mod_summary["submods"] = list()

    # register hooks
    handles = register_hooks(model, mod_summary)

    # make a forward pass
    x = torch.rand((batch_size, *input_shape)).type(dtype)
    out = model(x)
    # print(torch.cuda.memory_summary('cuda'))
    # print()

    ### remove these hooks
    for h in handles:
        h.remove()

    if print_parametarized_layers_only:
        mod_summary = remove_layers(summary)

    # display info
    print_model_info(model,
                     mod_summary,
                     input_shape,
                     batch_size,
                     num_spaces=spacing,
                     column_headers=column_headers,
                     device=device,
                     mem_stats=mem_stats,
                     depth_prefix="--",
                     model_name=model_name)
