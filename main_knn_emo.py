'''
author: Suhita Ghosh (suhitaghosh10)
email:  suhita.ghosh.10@gmail.com
'''

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import argparse
import torch

from fusion_synthesis.logger import utils
from data_cnpop_knn import get_data_loaders
from solver_knn import train

from fusion_synthesis.ddsp.vocoder import SubtractiveSynthesiser
from fusion_synthesis.ddsp.loss import PerceptualLoss
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument('-g', '--gpu', type=int, required=True, help='GPU number to use')
    parser.add_argument('-c', '--config', type=str, required=True, help='path to config file')
    return parser.parse_args(args=args, namespace=namespace)


if __name__ == '__main__':
    # parse commands
    cmd = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cmd.gpu)
    # load config
    args = utils.load_config(cmd.config)
    print(' > config:', cmd.config)
    print(' >    output folder:', args.env.expdir)

    # load model

    model = SubtractiveSynthesiser(
            sampling_rate=args.data.sampling_rate,
            block_size=args.data.block_size,
            n_mag_harmonic=args.model.n_mag_harmonic,
            n_mag_noise=args.model.n_mag_noise,
            n_harmonics=args.model.n_harmonics,
            window_type=args.model.window,
            convolve_power=args.model.convolve_power,
            device=args.device)

    # loss
    loss_func = PerceptualLoss(args.loss.n_ffts,
                               args.loss.jitter,
                               args.loss.shimmer,
                               use_kurtosis=args.kurtosis)

    # device
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(args.device)
    loss_func.to(args.device)

    # datas
    loader_train, loader_valid = get_data_loaders(args, whole_audio=False, extension=args.data.extension)

    # stage
    train(args, model=model, loss_func=loss_func, loader_train=loader_train, loader_test=loader_valid, is_part=args.is_part, generate_files=args.generate_files)