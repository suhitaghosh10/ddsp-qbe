import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from .conformer import Conformer


class ControlGen(nn.Module):
    def __init__(
            self,
            input_channel,
            output_splits,
            num_channels=64
    ):
        super().__init__()
        self.output_splits = output_splits

        # conv in stack
        self.stack_f0fuse = nn.Sequential(
            nn.Conv1d(input_channel + 1, num_channels, 3, 1, 1),
            nn.GroupNorm(4, num_channels),
            nn.LeakyReLU(),
            nn.Conv1d(num_channels, num_channels, 3, 1, 1))

        self.n_out = sum([v for k, v in
                          output_splits.items()])  # number of parameters to predict for the synthesis model [#(f0) + #(harmonic) + #(noise) ]
        # transformer
        self.decoder = Conformer(
            num_layers=3,
            num_heads=8,
            dim_model=num_channels,
            dim_keys=num_channels,
            dim_values=num_channels,
            residual_dropout=0.1,
            attention_dropout=0.1)

        self.norm = nn.LayerNorm(num_channels)
        self.dense_out = weight_norm(
            nn.Linear(num_channels, self.n_out))

    def forward(self, x6, x12=None, f0_norm=None, x6_emo1=None, x6_emo2=None):
        '''
        input:
            B x n_frames x n_mels
        return:
            dict of B x n_frames x feat
        '''

        e, x_phon_emo1, x_phon_emo2 = self._fusion(x6=x6, x12=x12, f0_norm=f0_norm, x6_emo1=x6_emo1, x6_emo2=x6_emo2)
        controls = self._split_to_dict(e, self.output_splits)

        return controls, x_phon_emo1, x_phon_emo2

    def _fusion(self, x6, x12, f0_norm, x6_emo1, x6_emo2):

        # non-prosodic branch
        x6 = x6.transpose(1, 2)
        f0_norm = f0_norm.transpose(1, 2)
        x6 = torch.concat([x6, f0_norm], dim=-2)
        x_phon = self.stack_f0fuse(x6).transpose(1, 2)  # phonetic embedding of source

        # non-prosodic representations for same content but different emotional content
        if x6_emo1 is not None and x6_emo2 is not None:
            x6_emo1 = x6_emo1.transpose(1, 2)
            x6_emo1 = torch.concat([x6_emo1, f0_norm], dim=-2)
            x_phon_emo1 = self.stack_f0fuse(x6_emo1).transpose(1, 2)  # phonetic embedding of emotion sample1

            x6_emo2 = x6_emo2.transpose(1, 2)
            x6_emo2 = torch.concat([x6_emo2, f0_norm], dim=-2)
            x_phon_emo2 = self.stack_f0fuse(x6_emo2).transpose(1, 2)  # phonetic embedding of emotion sample2
        else:
            x_phon_emo1 = x_phon_emo2 = None

        # prosodic branch
        x12 = x12.transpose(1, 2)
        x12 = torch.concat([x12, f0_norm], dim=-2)
        x_pro = self.stack_f0fuse(x12).transpose(1, 2)  # prosodic embedding of emotion sample2

        x = x_phon + x_pro  # combine prosodic and non-prosodic components

        x = self.decoder(x)
        x = self.norm(x)
        e = self.dense_out(x)
        return e, x_phon_emo1, x_phon_emo2

    def _split_to_dict(self, tensor, tensor_splits):
        """Split a tensor into a dictionary of multiple tensors."""
        labels = []
        sizes = []

        for k, v in tensor_splits.items():
            labels.append(k)
            sizes.append(v)

        tensors = torch.split(tensor, sizes, dim=-1)
        return dict(zip(labels, tensors))
