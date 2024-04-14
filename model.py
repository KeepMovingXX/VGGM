import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn.models import InnerProductDecoder, VGAE
from torch_geometric.nn.conv import GCNConv


class GCNEncoder(nn.Module):
    def __init__(self, linder_dim, in_channels, hidden_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.gcn_shared = GCNConv(in_channels, hidden_channels)
        self.gcn_mu = GCNConv(hidden_channels, out_channels)
        self.gcn_logvar = GCNConv(hidden_channels, out_channels)
        self.liner = nn.Linear(linder_dim, in_channels)
    def forward(self, x, edge_index):
        x = self.liner(x)
        x = F.relu(self.gcn_shared(x, edge_index))
        mu = self.gcn_mu(x, edge_index)
        # logvar = self.gcn_logvar(x, edge_index)
        logvar = F.relu(self.gcn_logvar(x, edge_index))
        return mu, logvar

class DeepVGAE(VGAE):
    def __init__(self, args):
        super(DeepVGAE, self).__init__(encoder=GCNEncoder(args.liner_dim,
                                                          args.enc_in_channels,
                                                          args.enc_hidden_channels,
                                                          args.enc_out_channels,
                                                          ),
                                       decoder=InnerProductDecoder())
        self.pi = nn.Parameter(torch.FloatTensor(args.components))
        self.mu = nn.Parameter(torch.FloatTensor(args.components, args.enc_out_channels))
        self.logvar = nn.Parameter(torch.FloatTensor(args.components, args.enc_out_channels))
        self.decoder_emb = nn.Linear(args.enc_out_channels, args.enc_out_channels*2)

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        return z

    def loss(self, x, edge_index):
        z = self.encode(x, edge_index)
        rec_loss = self.recon_loss(z, edge_index)
        kl_loss = 1 / x.size(0) * self.kl_loss()
        return rec_loss + kl_loss

    def loss_function(self, num_nodes, x, edge_index):
        mu, logvar = self.encoder(x, edge_index)
        z = self.encode(x, edge_index).unsqueeze(1)
        recon_loss = self.recon_loss(z, edge_index)
        h = z - self.mu
        h = torch.exp(-0.5 * torch.sum((h * h) / torch.exp(self.logvar), dim=2))
        weights = torch.softmax(self.pi, dim=0)

        h = h / torch.exp(torch.sum(0.5 * self.logvar, dim=1))
        p_z_given_c = h / (2 * math.pi)
        p_z_c = p_z_given_c * weights + 1e-9
        gamma = p_z_c / torch.sum(p_z_c, dim=1, keepdim=True)

        h = logvar.exp().pow(2).unsqueeze(1) + (mu.unsqueeze(1) - self.mu).pow(2)
        h = torch.sum(self.logvar + h / torch.exp(self.logvar), dim=2)
        com_loss = 0.5 * torch.sum(gamma * h) \
                   - torch.sum(gamma * torch.log(weights + 1e-9)) \
                   + torch.sum(gamma * torch.log(gamma + 1e-9)) \
                   - 0.5 * torch.sum(1 + logvar)
        com_loss = com_loss / (num_nodes * num_nodes)
        return recon_loss + com_loss