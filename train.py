'''
author: Suhita Ghosh (suhitaghosh10)
email:  suhita.ghosh.10@gmail.com
'''

import os

import argparse
import torch

from fusion_synthesis.utility.utils import load_config
from dataloader import get_data_loaders
from trainer import train

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
    args = load_config(cmd.config)
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
                               use_kurtosis=args.loss.use_kurtosis,
                               use_emo_loss=args.loss.use_emo_loss)

    # device
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(args.device)
    loss_func.to(args.device)

    # datas
    loader_train, loader_valid = get_data_loaders(args, whole_audio=False, extension=args.data.extension, use_emo_loss=args.loss.use_emo_loss)

    # stage
    train(args, model=model, loss_func=loss_func, loader_train=loader_train, loader_test=loader_valid, generate_files=args.inference.generate_files)