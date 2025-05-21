import torch
import torch.nn as nn
from sr_unet import SRUNet
from diffusion import Diffusion

class EDSRM(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(EDSRM, self).__init__()
        self.diffusion = Diffusion(timesteps=1000)
        self.sr_net = SRUNet(in_channels, out_channels)

    def forward(self, low_res, t, noise=None):
        if noise is None:
            noise = torch.randn_like(low_res)
        noised_input = self.diffusion.add_noise(low_res, t, noise)
        refined_output = self.sr_net(noised_input)
        return refined_output