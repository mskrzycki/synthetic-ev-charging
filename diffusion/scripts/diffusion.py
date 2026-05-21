import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from types import SimpleNamespace
import matplotlib.pyplot as plt
from pathlib import Path
from network import Attention, CNN               


class DDPM:
    def __init__(self, opt, data_loader):
        super().__init__()
        if opt.network == "attention":
            self.eps_model = Attention(opt).to(opt.device)
        else:
            self.eps_model = CNN(opt).to(opt.device)
        self.opt = opt
        self.data_loader = data_loader #### customised 
        self.n_steps = opt.n_steps

        if opt.schedule == "linear":
            self.beta = torch.linspace(opt.beta_start, opt.beta_end, self.n_steps, device=opt.device)
        else:
            self.beta = torch.linspace(opt.beta_start**0.5, opt.beta_end**0.5, self.n_steps, device=opt.device)**2

        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.sigma2 = torch.cat((self.beta[:1], self.beta[1:] * (1 - self.alpha_bar[:-1])/(1 - self.alpha_bar[1:])))

        self.optimizer = torch.optim.Adam(self.eps_model.parameters(), lr=opt.init_lr)
        self.loss_func = nn.MSELoss()
        p1, p2 = int(0.75 * opt.n_epochs), int(0.9 * opt.n_epochs)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[p1, p2], gamma=0.1)

    def gather(self, const, t):
        return const.gather(-1, t).view(-1, 1, 1)

    
    def q_xt_x0(self, x0, t):
        alpha_bar = self.gather(self.alpha_bar, t)
        mean = (alpha_bar**0.5)*x0
        var = 1 - alpha_bar
        return mean, var

    
    def q_sample(self, x0, t, eps):
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var**0.5)*eps

    
    def p_sample(self, xt, c, t):
        eps_theta = self.eps_model(xt, c, t)
        alpha_bar = self.gather(self.alpha_bar, t)
        alpha = self.gather(self.alpha, t)
        eps_coef = (1 - alpha)/(1 - alpha_bar)**0.5
        mean = (xt - eps_coef*eps_theta)/(alpha**0.5)
        var = self.gather(self.sigma2, t)
        if (t == 0).all():
            z = torch.zeros(xt.shape, device=xt.device)
        else:
            z = torch.randn(xt.shape, device=xt.device)
        return mean + (var**0.5)*z

    def cal_loss(self, x0, c, target_mask=None): 
        batch_size = x0.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, eps = noise)
        eps_theta = self.eps_model(xt, c, t)

        if target_mask is not None: ## customised: only calculate the loss on the target variable
            noise     = noise     * target_mask
            eps_theta = eps_theta * target_mask

        return self.loss_func(eps_theta, noise)
    

    @torch.no_grad()
    def sample(self, weight_path, n_samples, condition): ## simplified; outputs samples
        c = torch.from_numpy(condition).float().to(self.opt.device)
        c = c.view(1, -1).repeat(n_samples, 1)

        weight = torch.load(weight_path, map_location=self.opt.device) 
        self.eps_model.load_state_dict(weight)
        self.eps_model.eval() 

        samples = [] # output ready samples
        for i in range(n_samples):
            x = torch.randn([1, self.opt.seq_len, self.opt.input_dim], device=self.opt.device)
            for j in range(self.n_steps-1, -1, -1):
                t = torch.full((1,), j, dtype=torch.long, device=self.opt.device)
                x = self.p_sample(x, c, t)
            samples.append(x.squeeze(0).cpu().numpy())
        return np.stack(samples, axis=0)

    def train(self): # customised: simplified
        epoch_losses = []
        for epoch in range(self.opt.n_epochs):
            batch_losses = []
            for data in self.data_loader:
                x0 = data['input'].to(self.opt.device)
                c  = data['condition'].to(self.opt.device)
                mask = data.get('mask', None)

                self.optimizer.zero_grad()
                loss = self.cal_loss(x0, c, target_mask=mask)
                loss.backward()
                self.optimizer.step()
                batch_losses.append(loss.item())

            avg_loss = sum(batch_losses) / len(batch_losses)
            epoch_losses.append(avg_loss)
            print(f"Epoch {epoch+1}/{self.opt.n_epochs} | Loss: {avg_loss:.4f}")
            self.lr_scheduler.step()

        return epoch_losses


### additional version of sampling, in case the model is not loaded but is in the environment (useful for experiments and trials)
    @torch.no_grad()
    def sample_no_load(self, n_samples, condition):
        self.eps_model.eval()

        c = torch.from_numpy(condition).float().to(self.opt.device)
        c = c.view(1, -1).repeat(n_samples, 1)

        samples = []
        for _ in range(n_samples):
            x = torch.randn([1, self.opt.seq_len, self.opt.input_dim], device=self.opt.device)
            for step in range(self.n_steps - 1, -1, -1):
                t = torch.full((1,), step, dtype=torch.long, device=self.opt.device)
                x = self.p_sample(x, c, t)
            samples.append(x.squeeze(0).cpu().numpy())
        return np.stack(samples, axis=0)
