'''
author: wayn391@mastertones
'''

import os
import time
import yaml
import json
import torch


class Checkpoint(object):
    def __init__(
            self, 
            args,
            initial_global_step=-1):

        self.expdir = args.env.expdir
        exists_ok = True if self.expdir == 'test' else False

        # cold start
        self.global_step = initial_global_step
        self.init_time = time.time()
        self.last_time = time.time()

        # makedirs
        os.makedirs(self.expdir, exist_ok=exists_ok)       

        # ckpt
        self.path_ckptdir = os.path.join(self.expdir, 'ckpts')
        os.makedirs(self.path_ckptdir, exist_ok=exists_ok)       

        # save config
        path_config = os.path.join(self.expdir, 'config.yaml')
        with open(path_config, "w") as out_config:
            yaml.dump(dict(args), out_config)

    def save_models(
            self, 
            model_dict, 
            postfix='', 
            to_json=False):
        """save method"""
        for name, model in model_dict.items():
            self.save_model(
                model, 
                name,
                postfix=postfix,
                to_json=to_json)

    def save_model(
            self, 
            model, 
            name='model',
            postfix='',
            to_json=False):
        """save method"""
        # path
        if postfix:
            postfix = '_' + postfix
        path_pt = os.path.join(
            self.path_ckptdir , name+postfix+'.pt')
        path_params = os.path.join(
            self.path_ckptdir, name+postfix+'_params.pt')
       
        # check
        print(' [*] model saved: {}'.format(path_pt))
        print(' [*] model params saved: {}'.format(path_params))

        # save
        torch.save(model, path_pt)
        torch.save(model.state_dict(), path_params)

        # to json
        if to_json:
            path_json = os.path.join(
                self.path_ckptdir , name+'.json')
            to_json(path_params, path_json)

    def global_step_increment(self):
        self.global_step += 1


def to_json(path_params, path_json):
    params = torch.load(path_params, map_location=torch.device('cpu'))
    raw_state_dict = {}
    for k, v in params.items():
        val = v.flatten().numpy().tolist()
        raw_state_dict[k] = val

    with open(path_json, 'w') as outfile:
        json.dump(raw_state_dict, outfile,indent= "\t")
