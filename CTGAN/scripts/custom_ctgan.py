import torch
import torch.nn as nn
from torch.nn import Linear, LeakyReLU, BatchNorm1d, Dropout
from sdv.single_table import CTGANSynthesizer
from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer
from ctgan.errors import InvalidDataError
from ctgan.synthesizers.base import random_state

# residual blocks from CTGAN source
class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super().__init__()
        self.fc = Linear(inp_dim, out_dim)
        self.bn = BatchNorm1d(out_dim)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.fc(x)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([x, out], dim=1)

# attention 
class Attention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x): 
        attn_out, _ = self.mha(x.unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1))
        return self.norm(x + attn_out.squeeze(1))

# customized generator (added optional attention, depth)
class CustomGenerator(nn.Module):
    def __init__(self, embedding_dim, generator_dim, data_dim, cross_att_heads=None):
        super().__init__()
        self.use_attn = cross_att_heads is not None and cross_att_heads > 0
        if self.use_attn:
            self.attn = Attention(embedding_dim, cross_att_heads)
        dim = embedding_dim
        layers = []
        for h in generator_dim:
            layers.append(Residual(dim, h))
            dim += h
        layers.append(Linear(dim, data_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, z, bucket=None):
        if self.use_attn:
            z = self.attn(z)
        return self.net(z)

# customized discriminator (added optional attention, depth)
class CustomDiscriminator(nn.Module):
    def __init__(self, input_dim, discriminator_dim, pac=10):
        super().__init__()
        dim = input_dim * pac
        self.pac = pac
        self.pacdim = dim
        layers = []
        for h in discriminator_dim:
            layers += [Linear(dim, h), LeakyReLU(0.2), Dropout(0.5)]
            dim = h
        layers.append(Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def calc_gradient_penalty(self, real_data, fake_data, device, pac=10, lambda_=10):
        alpha = torch.rand(real_data.size(0)//pac,1,1,device=device)
        alpha = alpha.repeat(1,pac,real_data.size(1)).view(-1,real_data.size(1))
        interp = alpha*real_data + (1-alpha)*fake_data
        disc_interp = self(interp)
        grads = torch.autograd.grad(
            outputs=disc_interp, inputs=interp,
            grad_outputs=torch.ones_like(disc_interp,device=device),
            create_graph=True, retain_graph=True)[0]
        grad_norm = grads.view(-1,pac*real_data.size(1)).norm(2,dim=1)-1
        return (grad_norm**2).mean()*lambda_
    
    def forward(self, x):
        assert x.size(0)%self.pac==0
        return self.net(x.view(-1,self.pacdim))

# combine
class CustomCTGAN(CTGANSynthesizer):

    def __init__(self,generator_dim=(256,256),discriminator_dim=(256,256),cross_att_heads=None, **kwargs):
        super().__init__(generator_dim=generator_dim, discriminator_dim=discriminator_dim, **kwargs)
        self._cross_att_heads = cross_att_heads

    def _create_generator_and_discriminator(self):
        cond_size = self._data_sampler.dim_cond_vec()
        emb_dim = self._embedding_dim + cond_size
        data_dim = self._transformer.output_dimensions
        self._generator = CustomGenerator(emb_dim, self._generator_dim, data_dim,cross_att_heads=self._cross_att_heads).to(self._device)
        self._discriminator = CustomDiscriminator(data_dim + cond_size, self._discriminator_dim,pac=self.pac).to(self._device)
