#!/usr/bin/env python3

import torch.nn.functional as F
from torch import nn, optim
from collections import defaultdict
from utils import *


class VAE_x2(torch.nn.Module):
    def __init__(self, n_genes, n_latent, n_en_hidden, n_de_hidden, BCE_pos_weight, **kwargs):
        """ Constructor for class x1 """
        super(VAE_x2, self).__init__()
        self.n_genes = n_genes
        self.n_latent = n_latent
        self.n_en_hidden = n_en_hidden
        self.n_de_hidden = n_de_hidden

        self.init_w = kwargs.get('init_w', False)
        self.beta = kwargs.get('beta', 0.05)
        self.path_model = kwargs.get('path_model', "trained_vae")
        self.dev = kwargs.get('device', torch.device('cpu'))
        self.dropout = kwargs.get('dropout', 0.3)
        self.BCE_pos_weight = torch.Tensor([BCE_pos_weight]).to(self.dev) if BCE_pos_weight != None else None
        self.random_seed = kwargs.get('random_seed', 1234)


        encoder = [
            nn.Linear(self.n_genes, self.n_en_hidden[0]),
            nn.BatchNorm1d(self.n_en_hidden[0]),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        ]
        if len(n_en_hidden) > 1:
            for i in range(len(n_en_hidden)-1):
                en_hidden = [
                    nn.Linear(self.n_en_hidden[i], self.n_en_hidden[i+1]),
                    nn.BatchNorm1d(self.n_en_hidden[i+1]),
                    nn.ReLU(),
                    nn.Dropout(self.dropout)
                ]
                encoder = encoder + en_hidden

        self.encoder_x2 = nn.Sequential(*encoder)


        self.mu_z2 = nn.Sequential(nn.Linear(self.n_en_hidden[-1], self.n_latent),)
        self.logvar_z2 = nn.Sequential(nn.Linear(self.n_en_hidden[-1], self.n_latent),)

        if len(n_de_hidden) == 0:
            decoder = [nn.Linear(self.n_latent, self.n_genes)]
        else:
            decoder = [
                    nn.Linear(self.n_latent, self.n_de_hidden[0]),
                    nn.BatchNorm1d(self.n_de_hidden[0]),
                    nn.ReLU(),
                    nn.Dropout(self.dropout)
                    ]

            if len(n_de_hidden) > 1:
                for i in range(len(self.n_de_hidden)-1):
                    de_hidden = [
                        nn.Linear(self.n_de_hidden[i], self.n_de_hidden[i+1]),
                        nn.BatchNorm1d(self.n_de_hidden[i+1]),
                        nn.ReLU(),
                        nn.Dropout(self.dropout)
                    ]
                    decoder = decoder + de_hidden


            decoder.append(nn.Linear(self.n_de_hidden[-1], self.n_genes))
            decoder.append(nn.ReLU())

        self.decoder_x2 = nn.Sequential(*decoder)

        if self.init_w:
            self.encoder_x2.apply(self._init_weights)
            self.decoder_x2.apply(self._init_weights)
            
        
    def _init_weights(self, layer):
        """ Initialize weights of layer with Xavier uniform"""
        if type(layer)==nn.Linear:
            torch.nn.init.xavier_uniform_(layer.weight)
        return


    def encode(self, X):
        """ Encode data """
        y = self.encoder_x2(X)
        mu, logvar = self.mu_z2(y), self.logvar_z2(y)
        z = self.sample_latent(mu, logvar)
        return z, mu, logvar
    
    def decode(self, z):
        """ Decode data """
        X_rec = self.decoder_x2(z)
        return X_rec
    
    def sample_latent(self, mu, logvar):
        """ Sample latent space with reparametrization trick. First convert to std, sample normal(0,1) and get Z."""
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(self.dev)
        eps = eps.mul_(std).add_(mu)
        return eps

    def to_latent(self, X):
        """ Same as encode, but only returns z (no mu and logvar) """
        y = self.encoder(X)
        mu, logvar = self.mean(y), self.logvar(y)
        z = self.sample_latent(mu, logvar)
        return z

    def to_mu_logvar(self, X):
        y = self.encoder(X)
        mu, logvar = self.mean(y), self.logvar(y)
        return mu, logvar

    def _average_latent(self, X):
        """ """
        z = self.to_latent(X)
        mean_z = z.mean(0)
        return mean_z

    
    def forward(self, x1):
        """ Forward pass through full network"""
        z1, z1_mu, z1_logvar = self.encode(x1)
        rec_x1 = self.decode(z1)
        return rec_x1, z1_mu, z1_logvar

    def vae_loss(self, y_pred, y_true, mu, logvar):
        """ Custom loss for VAE """
        kld = -0.5 * torch.sum(1. + logvar - mu.pow(2) - logvar.exp(), )
        mse = F.mse_loss(y_pred, y_true, reduction="sum")
        return mse + self.beta * kld, mse, kld
