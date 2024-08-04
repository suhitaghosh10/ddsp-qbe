import yaml
import json

import torch
    
class DotDict(dict):
    def __getattr__(*args):
        val = dict.get(*args)
        return DotDict(val) if type(val) is dict else val

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_network_params(model_dict):
    info = dict()
    for model_name, model in model_dict.items():
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        info[model_name] = trainable_params
    return info


def load_config(path_config):
    with open(path_config, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)
    # print(args)
    return args





# def convert_tensor_to_numpy(tensor, is_squeeze=True):
#     if is_squeeze:
#         tensor = tensor.squeeze()
#     if tensor.requires_grad:
#         tensor = tensor.detach()
#     if tensor.is_cuda:
#         tensor = tensor.cpu()
#     return tensor.numpy()

           
def load_model_params(
        path_pt, 
        model,
        device='cpu'):

    # check
    print(' [*] restoring model from', path_pt)

    model.load_state_dict(
        torch.load(
            path_pt, 
            map_location=torch.device(device)))
    return model
