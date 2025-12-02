import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Embedder(nn.Module):
    def __init__(self, **kwargs):
        super(Embedder, self).__init__()

        self.kwargs = kwargs

        self.create_embedding_fn()


    def create_embedding_fn(self):
        embed_fns = []

        d = self.kwargs["input_dims"]

        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]

        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)

        for freq in freq_bands:
            
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim


    def forward(self, inputs):

        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class Update_SH_Coeffs(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Update_SH_Coeffs, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, output_size)
        )


    def forward(self, tvec_tx, shs_coeffs):
        N = shs_coeffs.size(0)
        
        tvec_tx = tvec_tx.repeat(N, 1)  
        
        sig_amp, sig_pha = shs_coeffs.split(1, dim=-1)  

        sig_amp = torch.cat((tvec_tx, sig_amp.squeeze(dim=-1)), dim=-1)
        sig_pha = torch.cat((tvec_tx, sig_pha.squeeze(dim=-1)), dim=-1)
        # 幅度和相位使用的是同一个 MLP
        sig_amp = self.model(sig_amp)
        sig_pha = self.model(sig_pha)

        shs_coeffs_updated = torch.stack((sig_amp, sig_pha), dim=-1)

        return shs_coeffs_updated

class WideFreqNetwork(nn.Module):
    def __init__(self, input_dim_pos, input_dim_freq, input_dim_pts, hidden_dim=256, D=8, skips=[1,3,5,7]):
        super(WideFreqNetwork, self).__init__()
        # Tx Pos + Frequency + Gaussian Pos
        self.total_input_dims = input_dim_pos + input_dim_freq + input_dim_pts
        self.D = D
        self.skips = skips

        self.model = nn.ModuleList()

        self.model.append(nn.Linear(self.total_input_dims, hidden_dim))

        for i in range(D - 1):
            if i in skips:
                # Skip connection: concat input to hidden feature
                self.model.append(nn.Linear(hidden_dim + self.total_input_dims, hidden_dim))
            else:
                self.model.append(nn.Linear(hidden_dim, hidden_dim))

        self.output = nn.Linear(hidden_dim, 4)

    def forward(self, tx, pts, freq):
        device = next(self.parameters()).device
        if pts.device != device:
            pts = pts.to(device)
        N = pts.size(0)
        tx = tx.repeat(N, 1).to(device)
        freq = freq.repeat(N, 1).to(device)
        # Concatenate to form input vector x
        x = torch.cat([tx, pts, freq], dim=-1)

        h = x
        for i, layer in enumerate(self.model):
            h = layer(h)
            h = F.relu(h)
            # Apply skip connection logic for the NEXT layer
            if i in self.skips:
                h = torch.cat([x, h], dim=-1)

        out = self.output(h)

        # Decompose output into 4 components
        s_a, s_p, att_a, att_p = out[:, 0:1], out[:, 1:2], out[:, 2:3], out[:, 3:4]
        # Radiance Amplitude Scale, Radiance Phase Shift, Transmittance Amplitude Scale, Transmittance Phase Shift

        # Physical constraints (Your customized logic)
        # 1. Phase: Map to [0, 2pi] using sigmoid
        s_p = torch.sigmoid(s_p) * np.pi * 2
        att_p = torch.sigmoid(att_p) * np.pi * 2
        # 2. Amplitude: Ensure non-negative
        s_a = torch.abs(F.leaky_relu(s_a)) # Using torch.abs is safer for gradients than builtin abs()
        att_a = torch.abs(F.leaky_relu(att_a)) # Usually we want transmittance scale close to 1.0 initially, leaky_relu might start at 0. Consider adding +1 bias if needed.

        # 3. Convert Phase to Complex Phasor: e^(j*phi)
        s_p_complex = torch.exp(1j * s_p)
        att_p_complex = torch.exp(1j * att_p)

        # Return 4 values: 2 real amplitudes, 2 complex phasors
        return s_a, s_p_complex, att_a, att_p_complex

