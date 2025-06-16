import torch
from torch import Tensor, nn
import torch.nn.functional as F
import tinycudann as tcnn


class EmbeddingModel(nn.Module):
    def __init__(
            self,
            appearance_n_fourier_freqs: int = 4,
            sh_degree: int = 3,
            appearance_embedding_dim: int = 16,
            appearance_model_sh: bool = True,
            use_tcnn: bool = True,  # Used to select whether to use tiny-cudann
    ):
        super().__init__()

        self.appearance_model_sh = appearance_model_sh
        self.use_tcnn = use_tcnn

        # Define input feature dimensions
        feat_in = 3
        if self.appearance_model_sh:
            feat_in = ((sh_degree + 1) ** 2) * 3
        mlp_input_dim = appearance_embedding_dim + feat_in + 6 * appearance_n_fourier_freqs

        # Initialize different types of MLP based on use_tcnn parameter
        if self.use_tcnn:
            # Use TCNN MLP
            network_config = {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 128,
                "n_hidden_layers": 2,  
            }
            self.mlp = tcnn.Network(
                n_input_dims=mlp_input_dim,
                n_output_dims=feat_in * 2,
                network_config=network_config,
            )
        else:
            # Use PyTorch MLP
            self.mlp = nn.Sequential(
                nn.Linear(mlp_input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, feat_in * 2),
            )

    def forward(self, color, image_embed, feature_embed):
        C0 = 0.28209479177387814
        input_color = color
        if not self.appearance_model_sh:
            color = color[..., :3]
        inp = torch.cat((color, image_embed, feature_embed), dim=-1)

        # Use MLP (whether TCNN or PyTorch implementation)
        output = self.mlp(inp) * 0.01
        offset, mul = torch.split(output, [color.shape[-1], color.shape[-1]], dim=-1)

        # Process offset and mul
        offset = torch.cat((offset / C0, torch.zeros_like(input_color[..., offset.shape[-1]:])), dim=-1)
        mul = mul.repeat(1, input_color.shape[-1] // mul.shape[-1])
        return input_color * mul + offset


class BackscatterNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backscatter_conv = nn.Conv2d(1, 3, 1, bias=False)
        self.residual_conv = nn.Conv2d(1, 3, 1, bias=False)

        nn.init.uniform_(self.backscatter_conv.weight, 0, 5.0)
        nn.init.uniform_(self.backscatter_conv.weight, 0, 5.0)

        self.B_inf = nn.Parameter(torch.rand(3, 1, 1))
        self.J_prime = nn.Parameter(torch.rand(3, 1, 1))
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, depth, appearance_embed=None):

        beta_b_conv = self.softplus(self.backscatter_conv(depth))
        beta_d_conv = self.softplus(self.residual_conv(depth))

        Bc = self.B_inf * (1 - torch.exp(-beta_b_conv)) + self.J_prime * torch.exp(-beta_d_conv)
        backscatter = Bc

        return backscatter


class DeattenuateNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.attenuation_conv = nn.Conv2d(1 + 16, 6, 1, bias=False)
        nn.init.uniform_(self.attenuation_conv.weight, 0, 5.0)
        self.attenuation_coef = nn.Parameter(torch.rand(6, 1, 1))
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()


    def forward(self, depth, appearance_embed=None):
        depth = depth.unsqueeze(0)
        appearance_embed = appearance_embed.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # Expand to (1, 8, 1, 1)
        appearance_embed = appearance_embed.repeat(depth.shape[0], 1, depth.shape[2], depth.shape[3])
        depth_emd = torch.cat((depth, appearance_embed), dim=1)

        attn_conv = torch.exp(-self.sigmoid(self.attenuation_conv(depth_emd)))
        beta_d = torch.stack(tuple(
            torch.sum(attn_conv[:, i:i + 2, :, :] * self.sigmoid(self.attenuation_coef[i:i + 2]), dim=1) for i in
            range(0, 6, 2)), dim=1)
 
        attenuation_map = torch.exp(-1.0 * torch.clamp(beta_d, 0.0) * depth).squeeze(0)


        attenuation_map = self.sigmoid(attenuation_map)

        return attenuation_map


