import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import to_dense_batch

from .neural_network import *
from .rl_base.policy_base import ActorCriticBase


class ActorCritic(ActorCriticBase):
    
    def __init__(self, p_net_num_nodes, p_net_feature_dim, p_net_edge_dim, v_net_feature_dim, v_net_edge_dim, embedding_dim=128, dropout_prob=0., batch_norm=False):
        super(ActorCritic, self).__init__()
        self.actor = Actor(p_net_num_nodes, p_net_feature_dim, p_net_edge_dim, v_net_feature_dim, v_net_edge_dim, embedding_dim, dropout_prob=dropout_prob, batch_norm=batch_norm)
        self.critic = Critic(p_net_num_nodes, p_net_feature_dim, p_net_edge_dim, v_net_feature_dim, v_net_edge_dim, embedding_dim, dropout_prob=dropout_prob, batch_norm=batch_norm)


class BaseModel(nn.Module):

    def __init__(self, p_net_num_nodes, p_net_feature_dim, p_net_edge_dim, v_net_feature_dim, v_net_edge_dim, embedding_dim=128, dropout_prob=0., batch_norm=False):
        super(BaseModel, self).__init__()
        self.v_net_encoder = NetEncoder(v_net_feature_dim, v_net_edge_dim, embedding_dim=embedding_dim, dropout_prob=dropout_prob, batch_norm=batch_norm)
        self.p_net_encoder = NetEncoder(p_net_feature_dim, p_net_edge_dim, embedding_dim=embedding_dim, dropout_prob=dropout_prob, batch_norm=batch_norm)
        self.num_heads = 4
        # self.att = nn.MultiheadAttention(embedding_dim, num_heads=self.num_heads, batch_first=True)
        self.att = MultiHeadSelfAttention(embedding_dim, embedding_dim, num_heads=self.num_heads)
        self._init_parameters()

    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                # nn.init.orthogonal_(param)
                stdv = 1. / math.sqrt(param.size(-1))
                param.data.uniform_(-stdv, stdv)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, obs):
        """Return logits of actions"""
        v_net_batch, p_net_batch = obs['v_net'], obs['p_net']
        v_node_embeddings, v_graph_embedding, v_node_dense_embeddings_with_graph_embedding, v_net_mask = self.v_net_encoder(v_net_batch)
        p_node_embeddings, p_graph_embedding, p_node_dense_embeddings_with_graph_embedding, p_net_mask = self.p_net_encoder(p_net_batch)
        # att_mask = ~v_net_mask.unsqueeze(1).repeat(1, p_node_dense_embeddings_with_graph_embedding.shape[1], 1).repeat_interleave(self.num_heads, dim=0)
        att_mask = v_net_mask.unsqueeze(1).repeat(1, p_node_dense_embeddings_with_graph_embedding.shape[1], 1)
        # fusion_embeddings, attn_weights = self.att(p_node_dense_embeddings_with_graph_embedding, v_node_dense_embeddings_with_graph_embedding, v_node_dense_embeddings_with_graph_embedding, attn_mask=att_mask)
        compatibility = self.att(p_node_dense_embeddings_with_graph_embedding, v_node_dense_embeddings_with_graph_embedding)
        fusion_embeddings = (p_graph_embedding + v_graph_embedding) / 2
        return fusion_embeddings, compatibility, att_mask


class Actor(nn.Module):

    def __init__(self, p_net_num_nodes, p_net_feature_dim, p_net_edge_dim, v_net_feature_dim, v_net_edge_dim, embedding_dim=128, dropout_prob=0., batch_norm=False):
        super(Actor, self).__init__()
        self.embeder = BaseModel(p_net_num_nodes, p_net_feature_dim, p_net_edge_dim, v_net_feature_dim, v_net_edge_dim, embedding_dim, dropout_prob=dropout_prob, batch_norm=batch_norm)

    def forward(self, obs):
        fusion_embeddings, attn_weights, att_mask = self.embeder(obs)  # [batch_size, num_p_nodes, num_v_nodes]
        action_logits = attn_weights.view(attn_weights.shape[0], -1)
        return action_logits


class Critic(nn.Module):
    
    def __init__(self, p_net_num_nodes, p_net_feature_dim, p_net_edge_dim, v_net_feature_dim, v_net_edge_dim, embedding_dim=128, dropout_prob=0., batch_norm=False):
        super(Critic, self).__init__()
        self.embeder = BaseModel(p_net_num_nodes, p_net_feature_dim, p_net_edge_dim, v_net_feature_dim, v_net_edge_dim, embedding_dim, dropout_prob=dropout_prob, batch_norm=batch_norm)
        # self.value_head = nn.Sequential(
        #     nn.Linear(embedding_dim, embedding_dim * 2),
        #     nn.ReLU(),
        #     nn.Linear(embedding_dim * 2, embedding_dim),
        #     nn.ReLU(),
        #     nn.Linear(embedding_dim, 1)
        # )
        self.value_head = nn.Sequential(
            nn.Linear(p_net_num_nodes, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )
        
    def forward(self, obs):
        # fusion_embeddings, attn_weights, att_mask = self.embeder(obs)
        # value = self.value_head(fusion_embeddings)
        # return value
        fusion_embeddings, attn_weights, att_mask = self.embeder(obs)
        # fusion_embeddings = fusion_embeddings.transpose(1, 2)
        value = self.value_head(attn_weights.transpose(1, 2))
        # print(value.shape, att_mask.shape, obs['v_net_size'], obs['v_net_size'])
        mask = att_mask.transpose(1, 2).all(dim=-1, keepdim=True)
        num_elements = torch.sum(mask, 1)
        value = torch.sum(value * mask, dim=1) / num_elements
        return value


class NetEncoder(nn.Module):

    GNNConvNet = GCNConvNet

    def __init__(self, net_feature_dim, net_edge_dim,embedding_dim=128, dropout_prob=0., batch_norm=False):
        super(NetEncoder, self).__init__()
        self.init_lin = nn.Linear(net_feature_dim, embedding_dim)
        self.net_gnn = self.GNNConvNet(embedding_dim, embedding_dim, num_layers=2, embedding_dim=embedding_dim, edge_dim=net_edge_dim, dropout_prob=dropout_prob, batch_norm=batch_norm)
        self.net_mean_pooling = GraphPooling('mean')

    def forward(self, net_batch):
        node_init_embeddings = self.init_lin(net_batch.x)
        net_batch = net_batch.clone()
        net_batch.x = node_init_embeddings
        node_embeddings = self.net_gnn(net_batch)
        graph_embedding = self.net_mean_pooling(node_embeddings, net_batch.batch)
        node_dense_embeddings, net_mask = to_dense_batch(node_embeddings, net_batch.batch)
        node_init_dense_embeddings, net_mask = to_dense_batch(node_init_embeddings, net_batch.batch)


        node_dense_embeddings_with_graph_embedding = node_dense_embeddings + graph_embedding.unsqueeze(1) + node_init_dense_embeddings
        return node_embeddings, graph_embedding, node_init_dense_embeddings, net_mask

