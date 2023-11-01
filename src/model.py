#!/usr/bin/env python3

import torch
import torch.nn.functional as F
from torch import nn, optim
import copy
from collections import defaultdict
from utils import *


class TranSiGen(torch.nn.Module):
    def __init__(self, n_genes, n_latent, n_en_hidden, n_de_hidden, features_dim, features_embed_dim, **kwargs):
        """ Constructor for class TranSiGen """
        super(TranSiGen, self).__init__()
        self.n_genes = n_genes
        self.n_latent = n_latent
        self.n_en_hidden = n_en_hidden
        self.n_de_hidden = n_de_hidden
        self.features_dim = features_dim
        self.feat_embed_dim = features_embed_dim
        self.init_w = kwargs.get('init_w', False)
        self.beta = kwargs.get('beta', 0.05)
        self.path_model = kwargs.get('path_model', 'trained_model')
        self.dev = kwargs.get('device', torch.device('cpu'))
        self.dropout = kwargs.get('dropout', 0.3)
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

        encoder_x1 = copy.deepcopy(encoder)
        self.encoder_x1 = nn.Sequential(*encoder_x1)

        self.mu_z2 = nn.Sequential(nn.Linear(self.n_en_hidden[-1], self.n_latent),)
        self.logvar_z2 = nn.Sequential(nn.Linear(self.n_en_hidden[-1], self.n_latent),)

        self.mu_z1 = nn.Sequential(nn.Linear(self.n_en_hidden[-1], self.n_latent),)
        self.logvar_z1 = nn.Sequential(nn.Linear(self.n_en_hidden[-1], self.n_latent),)

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

        decoder_x1 = copy.deepcopy(decoder)
        self.decoder_x1 = nn.Sequential(*decoder_x1)

        if self.feat_embed_dim == None:
            self.mu_z2Fz1 = nn.Sequential(nn.Linear(self.n_latent + self.features_dim, self.n_latent),)
            self.logvar_z2Fz1 = nn.Sequential(nn.Linear(self.n_latent + self.features_dim, self.n_latent),)
        else:
            feat_embeddings = [
                                nn.Linear(self.features_dim, self.feat_embed_dim[0]),
                                nn.BatchNorm1d(self.feat_embed_dim[0]),
                                nn.ReLU(),
                                nn.Dropout(self.dropout)
                                ]
            if len(self.feat_embed_dim) > 1:
                for i in range(len(self.feat_embed_dim)-1):
                    feat_hidden = [
                        nn.Linear(self.feat_embed_dim[i], self.feat_embed_dim[i+1]),
                        nn.BatchNorm1d(self.feat_embed_dim[i+1]),
                        nn.ReLU(),
                        nn.Dropout(self.dropout)
                    ]
                    feat_embeddings = feat_embeddings + feat_hidden
            self.feat_embeddings = nn.Sequential(*feat_embeddings)

            self.mu_z2Fz1 = nn.Sequential(nn.Linear(self.n_latent + self.feat_embed_dim[-1], self.n_latent),)
            self.logvar_z2Fz1 = nn.Sequential(nn.Linear(self.n_latent + self.feat_embed_dim[-1], self.n_latent),)


        if self.init_w:
            self.encoder_x1.apply(self._init_weights)
            self.decoder_x1.apply(self._init_weights)

            self.encoder_x2.apply(self._init_weights)
            self.decoder_x2.apply(self._init_weights)

            self.mu_z1.apply(self._init_weights)
            self.logvar_z1.apply(self._init_weights)

            self.mu_z2.apply(self._init_weights)
            self.logvar_z2.apply(self._init_weights)

            
        
    def _init_weights(self, layer):
        """ Initialize weights of layer with Xavier uniform"""
        if type(layer)==nn.Linear:
            torch.nn.init.xavier_uniform_(layer.weight)
        return

    def encode_x1(self, X):
        """ Encode data """
        y = self.encoder_x1(X)
        mu, logvar = self.mu_z1(y), self.logvar_z1(y)
        z = self.sample_latent(mu, logvar)
        return z, mu, logvar

    def encode_x2(self, X):
        """ Encode data """
        y = self.encoder_x2(X)
        mu, logvar = self.mu_z2(y), self.logvar_z2(y)
        z = self.sample_latent(mu, logvar)
        return z, mu, logvar
    
    def decode_x1(self, z):
        """ Decode data """
        X_rec = self.decoder_x1(z)
        return X_rec

    def decode_x2(self, z):
        """ Decode data """
        X_rec = self.decoder_x2(z)
        return X_rec
    
    def sample_latent(self, mu, logvar):
        """ Sample latent space with reparametrization trick. First convert to std, sample normal(0,1) and get Z."""
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(self.dev)
        eps = eps.mul_(std).add_(mu)
        return eps

    
    def forward(self, x1, features):
        """ Forward pass through full network"""
        z1, mu1, logvar1 = self.encode_x1(x1)
        x1_rec = self.decode_x1(z1)

        if self.feat_embed_dim != None:
            feat_embed = self.feat_embeddings(features)
        else:
            feat_embed = features
        z1_feat = torch.cat([z1, feat_embed], 1)
        mu_pred, logvar_pred = self.mu_z2Fz1(z1_feat), self.logvar_z2Fz1(z1_feat)
        z2_pred = self.sample_latent(mu_pred, logvar_pred)
        x2_pred = self.decode_x2(z2_pred)

        return x1_rec, mu1, logvar1, x2_pred, mu_pred, logvar_pred, z2_pred


    def loss(self, x1_train, x1_rec, mu1, logvar1, x2_train, x2_rec, mu2, logvar2, x2_pred, mu_pred, logvar_pred):

        mse_x1 = F.mse_loss(x1_rec, x1_train, reduction="sum")
        mse_x2 = F.mse_loss(x2_rec, x2_train, reduction="sum")
        mse_pert = F.mse_loss(x2_pred - x1_train, x2_train - x1_train, reduction="sum")

        kld_x1 = -0.5 * torch.sum(1. + logvar1 - mu1.pow(2) - logvar1.exp(), )
        kld_x2 = -0.5 * torch.sum(1. + logvar2 - mu2.pow(2) - logvar2.exp(), )
        kld_pert = -0.5 * torch.sum(1 + (logvar_pred - logvar2) - ((mu_pred - mu2).pow(2) + logvar_pred.exp()) / logvar2.exp(), )


        return mse_x1 + mse_x2 + mse_pert + self.beta * kld_x1 + self.beta * kld_x2 + self.beta * kld_pert, \
               mse_x1, mse_x2, mse_pert, kld_x1, kld_x2, kld_pert


    def get_feat_embdding(self, features):
        feat_embed = self.feat_embeddings(features)
        return feat_embed

    def train_model(self, learning_rate, weight_decay, n_epochs, train_loader, test_loader, save_model=True, metrics_func=None):
        """ Train TranSiGen """
        epoch_hist = defaultdict(list)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        loss_item = ['loss', 'mse_x1', 'mse_x2', 'mse_pert', 'kld_x1', 'kld_x2', 'kld_pert']

        # Train
        best_value = np.inf
        best_epoch = 0
        for epoch in range(n_epochs):
            train_size = 0
            loss_value = 0
            self.train()
            for x1_train, x2_train, features, mol_id, cid, sig in train_loader:
                x1_train = x1_train.to(self.dev)
                x2_train = x2_train.to(self.dev)
                features = features.to(self.dev)
                if x1_train.shape[0] == 1:
                    continue
                train_size += x1_train.shape[0]
                optimizer.zero_grad()

                x1_rec, mu1, logvar1, x2_pert, mu_pred, logvar_pred, z2_pred = self.forward(x1_train, features)
                z2, mu2, logvar2 = self.encode_x2(x2_train)
                x2_rec = self.decode_x2(z2)
                loss, _, _, _, _, _, _ = self.loss(x1_train, x1_rec, mu1, logvar1, x2_train, x2_rec, mu2, logvar2, x2_pert, mu_pred, logvar_pred)

                loss_value += loss.item()
                loss.backward()
                optimizer.step()

            # Eval
            train_dict, train_metrics_dict, train_metrics_dict_ls= self.test_model(loader=train_loader, loss_item=loss_item, metrics_func=metrics_func)
            train_loss = train_dict['loss']
            train_mse_x1 = train_dict['mse_x1']
            train_mse_x2 = train_dict['mse_x2']
            train_mse_pert = train_dict['mse_pert']
            train_kld_x1 = train_dict['kld_x1']
            train_kld_x2 = train_dict['kld_x2']
            train_kld_pert = train_dict['kld_pert']


            for k, v in train_dict.items():
                epoch_hist['train_' + k].append(v)

            for k, v in train_metrics_dict.items():
                epoch_hist['train_' + k].append(v)

            test_dict, test_metrics_dict, test_metricsdict_ls= self.test_model(loader=test_loader, loss_item=loss_item, metrics_func=metrics_func)
            test_loss = test_dict['loss']
            test_mse_x1 = test_dict['mse_x1']
            test_mse_x2 = test_dict['mse_x2']
            test_mse_pert = test_dict['mse_pert']
            test_kld_x1 = test_dict['kld_x1']
            test_kld_x2 = test_dict['kld_x2']
            test_kld_pert = test_dict['kld_pert']

            for k, v in test_dict.items():
                epoch_hist['valid_'+k].append(v)
            for k, v in test_metrics_dict.items():
                epoch_hist['valid_' + k].append(v)

            print('[Epoch %d] | loss: %.3f, mse_x1_rec: %.3f, mse_x2_rec: %.3f, mse_pert: %.3f, kld_x1: %.3f, kld_x2: %.3f, kld_pert: %.3f| '
                'valid_loss: %.3f, valid_mse_x1_rec: %.3f, valid_mse_x2_rec: %.3f, valid_mse_pert: %.3f, valid_kld_x1: %.3f, valid_kld_x2: %.3f, valid_kld_pert: %.3f|'
                % (epoch, train_loss, train_mse_x1, train_mse_x2, train_mse_pert, train_kld_x1, train_kld_x2, train_kld_pert,
                   test_loss, test_mse_x1, test_mse_x2, test_mse_pert, test_kld_x1, test_kld_x2, test_kld_pert), flush=True)

            if test_loss < best_value:
                best_value = test_loss
                best_epoch = epoch
                if save_model:
                    torch.save(self, self.path_model + 'best_model.pt')

        return epoch_hist, best_epoch

    def test_model(self, loader, loss_item=None, metrics_func=None):
        """Test model on input loader."""

        test_dict = defaultdict(float)
        metrics_dict_all = defaultdict(float)
        metrics_dict_all_ls = defaultdict(list)
        test_size = 0

        self.eval()
        with torch.no_grad():
            for x1_data, x2_data, mol_features, mol_id, cid, sig in loader:
                x1_data = x1_data.to(self.dev)
                x2_data = x2_data.to(self.dev)
                mol_features = mol_features.to(self.dev)
                cid = np.array(list(cid))
                sig = np.array(list(sig))
                test_size += x1_data.shape[0]

                x1_rec, mu1, logvar1, x2_pred, mu_pred, logvar_pred, z2_pred = self.forward(x1_data, mol_features)
                z2, mu2, logvar2 = self.encode_x2(x2_data)
                x2_rec = self.decode_x2(z2)
                loss_ls = self.loss(x1_data, x1_rec, mu1, logvar1, x2_data, x2_rec, mu2, logvar2, x2_pred, mu_pred, logvar_pred)
                if loss_item != None:
                    for idx, k in enumerate(loss_item):
                        test_dict[k] += loss_ls[idx].item()

                if metrics_func != None:
                    metrics_dict, metrics_dict_ls = self.eval_x_reconstruction(x1_data, x1_rec, x2_data, x2_rec, x2_pred, metrics_func=metrics_func)
                    for k in metrics_dict.keys():
                        metrics_dict_all[k] += metrics_dict[k]
                    for k in metrics_dict_ls.keys():
                        metrics_dict_all_ls[k] += metrics_dict_ls[k]

                    metrics_dict_all_ls['cp_id'] += list(mol_id.numpy())
                    metrics_dict_all_ls['cid'] += list(cid)
                    metrics_dict_all_ls['sig'] += list(sig)

                try:
                    x1_array = torch.cat([x1_array, x1_data], dim=0)
                    x1_rec_array = torch.cat([x1_rec_array, x1_rec], dim=0)
                    x2_array = torch.cat([x2_array, x2_data], dim=0)
                    x2_rec_array = torch.cat([x2_rec_array, x2_rec], dim=0)
                    x2_pred_array = torch.cat([x2_pred_array, x2_pred], dim=0)
                    mol_id_array = torch.cat([mol_id_array, mol_id], dim=0)
                    cid_array = np.concatenate((cid_array, cid), axis=0)
                except:
                    x1_array = x1_data.clone()
                    x1_rec_array = x1_rec.clone()
                    x2_array = x2_data.clone()
                    x2_rec_array = x2_rec.clone()
                    x2_pred_array = x2_pred.clone()
                    mol_id_array = mol_id.clone()
                    cid_array = cid.copy()


        for k in test_dict.keys():
            test_dict[k] = test_dict[k] / test_size

        for k in metrics_dict_all.keys():
            metrics_dict_all[k] = metrics_dict_all[k] / test_size

        return test_dict, metrics_dict_all, metrics_dict_all_ls


    def predict_profile(self, loader):
        """predict profiles."""

        test_size = 0
        # setup_seed(self.random_seed)
        self.eval()
        with torch.no_grad():
            for x1_data, x2_data, mol_features, mol_id, cid, sig in loader:
                cid = np.array(list(cid))
                sig = np.array(list(sig))
                x1_data = x1_data.to(self.dev)
                x2_data = x2_data.to(self.dev)
                mol_features = mol_features.to(self.dev)
                test_size += x1_data.shape[0]

                x1_rec, mu1, logvar1, x2_pred, mu_pred, logvar_pred, z2_pred = self.forward(x1_data, mol_features)
                z2, mu2, logvar2 = self.encode_x2(x2_data)
                x2_rec = self.decode_x2(z2)

                try:
                    x1_array = torch.cat([x1_array, x1_data], dim=0)
                    x1_rec_array = torch.cat([x1_rec_array, x2_rec], dim=0)
                    x2_array = torch.cat([x2_array, x2_data], dim=0)
                    x2_rec_array = torch.cat([x2_rec_array, x2_rec], dim=0)
                    x2_pred_array = torch.cat([x2_pred_array, x2_pred], dim=0)
                    z2_pred_array = torch.cat([z2_pred_array, z2_pred], dim=0)
                    mol_id_array = torch.cat([mol_id_array, mol_id], dim=0)
                    cid_array = np.concatenate((cid_array, cid), axis=0)
                    sig_array = np.concatenate((sig_array, sig), axis=0)

                except:
                    x1_array = x1_data.clone()
                    x1_rec_array = x1_rec.clone()
                    x2_array = x2_data.clone()
                    x2_rec_array = x2_rec.clone()
                    x2_pred_array = x2_pred.clone()
                    z2_pred_array = z2_pred.clone()
                    mol_id_array = mol_id.clone()
                    cid_array = cid.copy()
                    sig_array = sig.copy()


        x1_array = x1_array.cpu().numpy().astype(float)
        x1_rec_array = x1_rec_array.cpu().numpy().astype(float)
        x2_array = x2_array.cpu().numpy().astype(float)
        x2_rec_array = x2_rec_array.cpu().numpy().astype(float)
        x2_pred_array = x2_pred_array.cpu().numpy().astype(float)
        z2_pred_array = z2_pred_array.cpu().numpy().astype(float)
        mol_id_array = mol_id_array.cpu().numpy().astype(float)
        return x1_array, x2_array, x1_rec_array, x2_rec_array, x2_pred_array, z2_pred_array, mol_id_array, cid_array, sig_array

    def predict_profile_for_x1(self, loader):

        self.eval()
        with torch.no_grad():
            for x1_data, mol_features, mol_id, cid in loader:
                cid = np.array(list(cid))
                x1_data = x1_data.to(self.dev)
                mol_features = mol_features.to(self.dev)
                x1_rec, mu1, logvar1, x2_pred, mu_pred, logvar_pred, z2_pred = self.forward(x1_data, mol_features)
                try:
                    x2_pred_array = torch.cat([x2_pred_array, x2_pred], dim=0)
                    mol_id_array = torch.cat([mol_id_array, mol_id], dim=0)
                    cid_array = np.concatenate((cid_array, cid), axis=0)
                except:
                    x2_pred_array = x2_pred.clone()
                    mol_id_array = mol_id.clone()
                    cid_array = cid.copy()

        x2_pred_array = x2_pred_array.cpu().numpy().astype(float)
        mol_id_array = mol_id_array.cpu().numpy().astype(float)

        return x2_pred_array, mol_id_array, cid_array


    def eval_x_reconstruction(self, x1, x1_rec, x2, x2_rec, x2_pred, metrics_func=['pearson']):
        """
        Compute reconstruction evaluation metrics for x1_rec, x2_rec, x2_pred
        """
        x1_np = x1.data.cpu().numpy().astype(float)
        x2_np = x2.data.cpu().numpy().astype(float)
        x1_rec_np = x1_rec.data.cpu().numpy().astype(float)
        x2_rec_np = x2_rec.data.cpu().numpy().astype(float)
        x2_pred_np = x2_pred.data.cpu().numpy().astype(float)

        DEG_np = x2_np - x1_np
        DEG_rec_np = x2_rec_np - x1_np
        DEG_pert_np = x2_pred_np - x1_np

        metrics_dict = defaultdict(float)
        metrics_dict_ls = defaultdict(list)
        for m in metrics_func:
            if m in ['precision10', 'precision20', 'precision50', 'precision100', 'precision200']:
                for i in range(x1_np.shape[0]):
                    precision_neg, precision_pos = get_metric_func(m)(x1_np[i, :], x1_rec_np[i, :])
                    metrics_dict['x1_rec_neg_' + m] += precision_neg
                    metrics_dict['x1_rec_pos_' + m] += precision_pos
                    metrics_dict_ls['x1_rec_neg_' + m] += [precision_neg]
                    metrics_dict_ls['x1_rec_pos_' + m] += [precision_pos]
            else:
                for i in range(x1_np.shape[0]):
                    metrics_dict['x1_rec_' + m] += get_metric_func(m)(x1_np[i, :], x1_rec_np[i, :])
                    metrics_dict_ls['x1_rec_' + m] += [get_metric_func(m)(x1_np[i, :], x1_rec_np[i, :])]

        for m in metrics_func:
            if m in ['precision10', 'precision20', 'precision50', 'precision100', 'precision200']:
                for i in range(x1_np.shape[0]):
                    precision_neg, precision_pos = get_metric_func(m)(x2_np[i, :], x2_rec_np[i, :])
                    metrics_dict['x2_rec_neg_' + m] += precision_neg
                    metrics_dict['x2_rec_pos_' + m] += precision_pos
                    metrics_dict_ls['x2_rec_neg_' + m] += [precision_neg]
                    metrics_dict_ls['x2_rec_pos_' + m] += [precision_pos]
            else:
                for i in range(x1_np.shape[0]):
                    metrics_dict['x2_rec_' + m] += get_metric_func(m)(x2_np[i, :], x2_rec_np[i, :])
                    metrics_dict_ls['x2_rec_' + m] += [get_metric_func(m)(x2_np[i, :], x2_rec_np[i, :])]

        for m in metrics_func:
            if m in ['precision10', 'precision20', 'precision50', 'precision100', 'precision200']:
                for i in range(x1_np.shape[0]):
                    precision_neg, precision_pos = get_metric_func(m)(x2_np[i, :], x2_pred_np[i, :])
                    metrics_dict['x2_pred_neg_' + m] += precision_neg
                    metrics_dict['x2_pred_pos_' + m] += precision_pos
                    metrics_dict_ls['x2_pred_neg_' + m] += [precision_neg]
                    metrics_dict_ls['x2_pred_pos_' + m] += [precision_pos]
            else:
                for i in range(x1_np.shape[0]):
                    metrics_dict['x2_pred_' + m] += get_metric_func(m)(x2_np[i, :], x2_pred_np[i, :])
                    metrics_dict_ls['x2_pred_' + m] += [get_metric_func(m)(x2_np[i, :], x2_pred_np[i, :])]

        for m in metrics_func:
            if m in ['precision10', 'precision20', 'precision50', 'precision100', 'precision200']:
                for i in range(x1_np.shape[0]):
                    precision_neg, precision_pos = get_metric_func(m)(DEG_np[i, :], DEG_rec_np[i, :])
                    metrics_dict['DEG_rec_neg_' + m] += precision_neg
                    metrics_dict['DEG_rec_pos_' + m] += precision_pos
                    metrics_dict_ls['DEG_rec_neg_' + m] += [precision_neg]
                    metrics_dict_ls['DEG_rec_pos_' + m] += [precision_pos]
            else:
                for i in range(x1_np.shape[0]):
                    metrics_dict['DEG_rec_' + m] += get_metric_func(m)(DEG_np[i, :], DEG_rec_np[i, :])
                    metrics_dict_ls['DEG_rec_' + m] += [get_metric_func(m)(DEG_np[i, :], DEG_rec_np[i, :])]

        for m in metrics_func:
            if m in ['precision10', 'precision20', 'precision50', 'precision100', 'precision200']:
                for i in range(x1_np.shape[0]):
                    precision_neg, precision_pos = get_metric_func(m)(DEG_np[i, :], DEG_pert_np[i, :])
                    metrics_dict['DEG_pred_neg_' + m] += precision_neg
                    metrics_dict['DEG_pred_pos_' + m] += precision_pos
                    metrics_dict_ls['DEG_pred_neg_' + m] += [precision_neg]
                    metrics_dict_ls['DEG_pred_pos_' + m] += [precision_pos]
            else:
                for i in range(x1_np.shape[0]):
                    metrics_dict['DEG_pred_' + m] += get_metric_func(m)(DEG_np[i, :], DEG_pert_np[i, :])
                    metrics_dict_ls['DEG_pred_' + m] += [get_metric_func(m)(DEG_np[i, :], DEG_pert_np[i, :])]
        return metrics_dict, metrics_dict_ls

