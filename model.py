from dataclasses import dataclass
from typing import Dict

import torch
from torch import distributions as dists
from torch import nn

from models.genesis.component_vae import ComponentVAE
from models.genesis.geco import GECO
from models.genesis.mask_vae import MaskVAE
from models.shared.encoder_decoder import EncoderParams, DecoderParams


class MaskVAEParams(Dict):
    hidden_state_lstm: int
    encoded_image_size: int
    latent_size: int
    encoder_params: EncoderParams
    decoder_params: DecoderParams

class ComponentVAEParams(Dict):
    latent_size: int
    encoder_params: EncoderParams
    decoder_params: DecoderParams

@dataclass(eq=False)
class Genesis(nn.Module):

    width: int
    height: int
    num_slots: int
    sigma_recon: float
    geco_goal_constant: float
    geco_step_size: float
    geco_alpha: float
    geco_init: float
    geco_min: float
    geco_speedup: float
    mask_vae_params: MaskVAEParams
    component_vae_params: ComponentVAEParams
    latent_size: int = -1
    input_channels: int = 3
    name: str = 'genesis'

    def __post_init__(self):
        super().__init__()
        self.mask_latent_size = self.mask_vae_params.latent_size
        self.hidden_state_lstm = self.mask_vae_params.hidden_state_lstm
        self.component_latent_size = self.component_vae_params.latent_size
        self.prior_component = nn.Sequential(
            nn.Linear(self.mask_latent_size, self.hidden_state_lstm),
            nn.ELU(),
            nn.Linear(self.hidden_state_lstm, self.hidden_state_lstm),
            nn.ELU(),
            nn.Linear(self.hidden_state_lstm, self.component_latent_size * 2)
        )
        self.prior_autoregressive_mask = nn.LSTM(self.mask_latent_size,
                                                 self.hidden_state_lstm)
        self.prior_linear_mask = nn.Linear(self.hidden_state_lstm,
                                           self.mask_latent_size * 2)
        self.mask_vae_params.update(num_slots=self.num_slots)
        self.mask_vae = MaskVAE(**self.mask_vae_params)
        self.component_vae = ComponentVAE(**self.component_vae_params)

        self.assert_shapes = True

        self.geco_goal_constant *= 3 * self.width * self.height
        self.geco_step_size *= (64 ** 2 / (self.width * self.height))
        self.geco_speedup = self.geco_speedup

        self.geco = GECO(self.geco_goal_constant, self.geco_step_size, self.geco_alpha,
                         self.geco_init, self.geco_min, self.geco_speedup)
        self.latent_size = self.mask_vae.latent_size + self.component_vae.latent_size


    def to(self, device):
        super().to(device)
        self.prior_component.to(device)
        self.prior_autoregressive_mask.to(device)
        self.prior_linear_mask.to(device)
        self.geco.to(device)
        return self
    def latent_loss(self, mask_dist_posteriors, comp_dist_posteriors, z_post_mask, z_post_comp):
        masks_kl = self.mask_latent_loss(mask_dist_posteriors, z_post_mask)
        comp_kl = self.comp_latent_loss(comp_dist_posteriors, z_post_mask, z_post_comp)
        return masks_kl + comp_kl

    def prior_sigma(self, s):
        return (s + 4.0).sigmoid() + 1e-4

    def mask_latent_loss(self, dist_mask_post, z_post_mask):
        dist_mask_prior = [dists.Normal(0, 1)]
        seq_z_post_mask = z_post_mask.permute(1, 0, 2)[:-1]
        prior_rnn_masks, _ = self.prior_autoregressive_mask(seq_z_post_mask)
        mu_mask_prior, sigma_mask_prior = self.prior_linear_mask(prior_rnn_masks).chunk(2, dim=-1)
        mu_mask_prior = mu_mask_prior.tanh()
        sigma_mask_prior = self.prior_sigma(sigma_mask_prior)
        mask_kl = 0.0
        num_slots_minus_1 = len(mu_mask_prior)
        for i in range(num_slots_minus_1):
            dist_mask_prior.append(dists.Normal(mu_mask_prior[i], sigma_mask_prior[i]))
        if self.assert_shapes:
            assert len(dist_mask_prior) == len(dist_mask_post) == z_post_mask.size(1) == self.num_slots
        for i in range(len(dist_mask_prior)):
            mask_kl += (dist_mask_post[i].log_prob(z_post_mask[:, i]) -
                         dist_mask_prior[i].log_prob(z_post_mask[:, i])).sum(dim=1)
        # print(masks_kl.shape)
        return mask_kl.mean(dim=0)

    def comp_latent_loss(self, dist_post_comp: list, z_post_mask: torch.Tensor, z_post_comp: torch.Tensor):
        mu_comp_prior, sigma_comp_prior = self.prior_component(z_post_mask).chunk(2, dim=-1)  # bs, num_slots, self.mask_latent_size both mean and std
        mu_comp_prior = mu_comp_prior.tanh()  # one for each slot
        sigma_comp_prior = self.prior_sigma(sigma_comp_prior)
        if self.assert_shapes:
            assert mu_comp_prior.shape == (z_post_comp.size(0), self.num_slots, self.component_latent_size), \
                f"{mu_comp_prior.shape} - {(z_post_comp.size(0), self.num_slots, self.mask_latent_size)}"
        dist_prior_comp = [dists.Normal(mu_comp_prior[:, i], sigma_comp_prior[:, i]) for i in
                           range(mu_comp_prior.size(1))]
        comp_kl = 0.0
        for i in range(len(dist_prior_comp)):
            comp_kl += dist_post_comp[i].log_prob(z_post_comp[:, i]).sum(dim=1) - \
                       dist_prior_comp[i].log_prob(z_post_comp[:, i]).sum(dim=1)
        return comp_kl.mean(dim=0)

    def recon_loss(self, x, x_recon_comp, log_masks):
        if self.assert_shapes:
            assert x_recon_comp.shape == (len(x), self.num_slots, 3, self.width, self.height) == x.shape
            assert log_masks.shape == (len(x), self.num_slots, 1, self.width, self.height)
        recon_dist = dists.Normal(x_recon_comp, self.sigma_recon)
        log_p = recon_dist.log_prob(x)
        log_mx: torch.Tensor = log_p + log_masks
        log_mx = - log_mx.logsumexp(dim=1)  # over slots
        return log_mx.mean(dim=0).sum()

    def forward(self, x):
        bs = len(x) 
        output_mask_vae = self.mask_vae(x)
        log_masks_hat = output_mask_vae['log_masks_hat']
        if self.assert_shapes:
            assert log_masks_hat.shape == (bs, self.num_slots, 1, self.width, self.height)

        x_input = x.unsqueeze(1).repeat(1, self.num_slots, 1, 1, 1)
        in_vae = torch.cat([x_input, log_masks_hat], dim=2)  # bs, num_slots, 4, w, h
        in_vae = in_vae.flatten(0, 1)
        comp_output = self.component_vae(in_vae)

        mu_post_comp = comp_output['mu_p'].view(bs, self.num_slots, self.component_latent_size)
        sigma_post_comp = comp_output['sigma_p'].view(bs, self.num_slots, self.component_latent_size)
        dist_post_comp = [dists.Normal(mu_post_comp[:, i], sigma_post_comp[:, i]) for i in range(self.num_slots)]
        z_post_comp = comp_output['z_post_comp'].view(bs, self.num_slots, self.component_latent_size)
        z_post_mask = output_mask_vae['z_post_masks']
        dist_post_mask = output_mask_vae['post_dists_masks']

        kl_loss = self.latent_loss(dist_post_mask, dist_post_comp,
                                   z_post_mask, z_post_comp)
        slots_recon = comp_output['img_recon'].view(bs, self.num_slots,
                                                    3, x.size(2), x.size(3))
        recon_loss_value = self.recon_loss(x_input, slots_recon, log_masks_hat)
        loss_value = self.geco.loss(recon_loss_value, kl_loss)
        z = torch.cat([z_post_mask, z_post_comp], dim=2)
        self.assert_shapes = False
        return dict(kl_loss=kl_loss, recon_loss=recon_loss_value, slot=slots_recon,
                    mask=log_masks_hat.exp(), loss=loss_value, z=z)
