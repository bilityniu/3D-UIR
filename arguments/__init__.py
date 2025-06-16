#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._depths = ""
        self._resolution = -1
        self._white_background = False
        self.train_test_exp = False
        self.data_device = "cuda"
        self.eval = False
        self.render_items = ['RGB', 'RGB_J', 'Depth', 'Edge', 'Normal', 'Curvature', 'Alpha']
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.025
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001

        # self.medium_mlp_lr = 0.001
        # self.medium_mlp_lr_init = 0.001
        # self.medium_mlp_lr_final = 0.00015
        # self.medium_mlp_lr_delay_steps = 0.0
        # self.medium_mlp_lr_delay_mult = 0.01
        # self.medium_mlp_lr_max_steps = 40_000
        #
        # self.direction_encoding_lr = 0.001
        # self.direction_encoding_lr_init = 0.001
        # self.direction_encoding_lr_final = 0.00015
        # self.direction_encoding_lr_max_steps = 30_000

        self.medium_model = True
        self.bs_lr = 0.01
        self.bs_lr_init = 0.01
        self.bs_lr_final = 0.00015
        self.bs_lr_max_steps = 30_000

        self.da_lr = 0.001
        self.da_lr_init = 0.001
        self.da_lr_final = 0.00015
        self.da_lr_max_steps = 30_000


        self.position_encoding = False
        self.appearance_model = True
        self.appearance_mlp_lr = 0.001
        self.appearance_model_lr_init = 0.001
        self.appearance_model_lr_final = 0.00015
        self.appearance_model_lr_max_steps = 30_000

        self.use_avg_appearance = True

        self.embedding_dims = 16
        self.embeddings_lr = 0.001

        self.exposure_lr_init = 0.01
        self.exposure_lr_final = 0.001
        self.exposure_lr_delay_steps = 0
        self.exposure_lr_delay_mult = 0.0

        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 0.01

        self.densify_grad_abs_threshold = 0.0004

        self.use_reduce = True
        self.opacity_reduce_interval = 500
        self.use_prune_weight = True
        self.prune_until_iter = 25000
        self.min_weight = 0.7

        self.scale_loss_weight = 100.0

        self.warm_up_until = 7000
        self.prune_threshold = 0.005
        self.clone_threshold = 0.0002
        self.split_threshold = 0.0002
        self.atom_proliferation_until = 7000

        # Pearson Depth Loss
        self.lambda_local_pearson = 0.0
        self.lambda_pearson = 0.1
        self.box_p = 128
        self.p_corr = 0.5

        self.lambda_normal = 0.1
        self.lambda_tv = 0.1
        self.tv_from_iter = 500
        self.tv_until_iter = 30_000

        self.random_background = False
        self.optimizer_type = "default"
        
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
