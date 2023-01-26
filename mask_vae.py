from dataclasses import dataclass

import torch
from torch import nn

from models.shared.encoder_decoder import EncoderParams, DecoderParams
from models.shared.sequential_cnn import make_sequential_from_config


@dataclass(eq=False)
class MaskVAE(nn.Module):

    num_slots: int
    latent_size: int
    hidden_state_lstm: int
    encoded_image_size: int
    encoder_params: EncoderParams
    decoder_params: DecoderParams

    def __post_init__(self):
        super().__init__()
        self.encoder = make_sequential_from_config(**self.encoder_params)
        self.decoder = make_sequential_from_config(**self.decoder_params)
        self.output_decoder_channels = self.decoder_params.channels[-1]
        self.encoded_to_z_dist = nn.Linear(self.hidden_state_lstm, self.latent_size * 2)
        self.lstm = nn.LSTM(self.latent_size + self.hidden_state_lstm, 2 * self.latent_size)
        self.linear_z_k = nn.Linear(self.latent_size * 2, self.latent_size * 2)
        self.softplus = nn.Softplus()

    def forward(self, x):
        bs, channels, width, height = x.shape
        encoded = self.encoder(x).squeeze(3).squeeze(2)
        mu_0, sigma_0 = self.encoded_to_z_dist(encoded).chunk(2, dim=1)
        sigma_0 = self.softplus(sigma_0 + 0.5)  # from genesis paper
        post_0 = torch.distributions.Normal(mu_0, sigma_0)
        z_0 = post_0.rsample()
        z = [z_0]
        post_dists_masks = [post_0]
        state = None
        for k in range(1, self.num_slots):
            h_z = torch.cat([encoded, z[-1]], dim=1).unsqueeze(0)
            output, state = self.lstm(h_z, state)
            mu_k, sigma_k = self.linear_z_k(output.squeeze()).chunk(2, dim=1)
            sigma_k = self.softplus(sigma_k + 0.5)
            post_k = torch.distributions.Normal(mu_k, sigma_k)
            z_k = post_k.rsample()
            post_dists_masks.append(post_k)
            z.append(z_k)
        z = torch.stack(z, dim=1)  # bs, num_slots, latent_size
        z_flatten = z.flatten(0, 1).unsqueeze(-1).unsqueeze(-1)
        mask_logit = self.decoder(z_flatten)
        mask_logit = mask_logit.view(bs, self.num_slots, self.output_decoder_channels, width, height)
        log_masks_hat = self.stick_breaking_process(mask_logit, width, height)
        return dict(log_masks_hat=log_masks_hat, post_dists_masks=post_dists_masks, z_post_masks=z)

    def stick_breaking_process(self, mask_logit, width, height):
        bs = len(mask_logit)
        log_masks_hat = torch.zeros(bs, self.num_slots, 1, width, height, device=mask_logit.device)
        scope_mask = torch.zeros(bs, 1, width, height, device=mask_logit.device)
        for i in range(self.num_slots - 1):
            log_mask_i = mask_logit[:, i].log_softmax(dim=1)
            log_masks_hat[:, i] = scope_mask + log_mask_i[:, 0:1]
            scope_mask = scope_mask + log_mask_i[:, 1:2]
        log_masks_hat[:, -1] = scope_mask
        return log_masks_hat


class Flatten(nn.Module):

    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x):
        return x.flatten(self.start_dim, self.end_dim)


class Conv2dGLU(nn.Module):

    def __init__(self, in_channels, out_channels, ksize, stride, padding, use_norm=True):
        super().__init__()
        self.use_norm = use_norm
        self.conv_layer = nn.Conv2d(in_channels, out_channels * 2, ksize, stride, padding)
        if self.use_norm:
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out1, out2 = self.conv_layer(x).chunk(2, dim=1)
        if self.use_norm:
            out1 = self.bn1(out1)
            out2 = self.bn2(out2)
        return out1 * out2.sigmoid()


class ConvTranspose2dGLU(nn.Module):

    def __init__(self, in_channels, out_channels, ksize, stride, padding, output_padding, use_norm=True):
        super().__init__()
        self.use_norm = use_norm
        self.conv_layer = nn.ConvTranspose2d(in_channels, out_channels * 2, ksize, stride, padding, output_padding)
        if self.use_norm:
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out1, out2 = self.conv_layer(x).chunk(2, dim=1)
        if self.use_norm:
            out1 = self.bn1(out1)
            out2 = self.bn2(out2)
        return out1 * out2.sigmoid()
