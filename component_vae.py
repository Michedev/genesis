from dataclasses import dataclass, field
from torch.nn.functional import softplus
from torch import nn, distributions as dists
from models.shared.encoder_decoder import EncoderParams, DecoderParams
from models.shared.encoder_decoder import BroadcastDecoderNet,EncoderNet

@dataclass(eq=False)
class ComponentVAE(nn.Module):

    latent_size: int
    encoder_params: EncoderParams
    decoder_params: DecoderParams

    encoder: EncoderNet = field(init=False)
    decoder: BroadcastDecoderNet = field(init=False)

    def __post_init__(self):
        super().__init__()
        self.encoder = EncoderNet(**self.encoder_params)
        self.decoder = BroadcastDecoderNet(**self.decoder_params)

    def forward(self, x):
        mu_p, sigma_p = self.encoder(x).chunk(2, dim=-1)
        sigma_p = softplus(sigma_p + 0.5)  # from genesis paper
        post_distr = dists.Normal(mu_p, sigma_p)
        z_post = post_distr.rsample()
        recon = self.decoder(z_post).sigmoid()  # from genesis paper
        return dict(img_recon=recon, z_post_comp=z_post,
                    post_dist=post_distr, mu_p=mu_p, sigma_p=sigma_p)