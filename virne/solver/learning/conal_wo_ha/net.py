import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import to_dense_batch

from ..neural_network import *
from ..rl_base.policy_base import ActorCriticBase


class ActorCritic(ActorCriticBase):
    
    def __init__(self, p_net_num_nodes, p_net_feature_dim, p_net_edge_dim, v_net_feature_dim, v_net_edge_dim, embedding_dim=128, dropout_prob=0., batch_norm=False):
        super(ActorCritic, self).__init__()
        self.embedding_dim = embedding_dim
        self.encoder = BaseModel(p_net_num_nodes, p_net_feature_dim, p_net_edge_dim, v_net_feature_dim, v_net_edge_dim, embedding_dim, dropout_prob=dropout_prob, batch_norm=batch_norm)
        self.actor = Actor(p_net_num_nodes, p_net_feature_dim, p_net_edge_dim, v_net_feature_dim, v_net_edge_dim, embedding_dim, dropout_prob=dropout_prob, batch_norm=batch_norm)
        self.critic = Critic(p_net_num_nodes, p_net_feature_dim, p_net_edge_dim, v_net_feature_dim, v_net_edge_dim, embedding_dim, dropout_prob=dropout_prob, batch_norm=batch_norm)
        self.cost_critic = Critic(p_net_num_nodes, p_net_feature_dim, p_net_edge_dim, v_net_feature_dim, v_net_edge_dim, embedding_dim, dropout_prob=dropout_prob, batch_norm=batch_norm)
        self.lambda_net = Critic(p_net_num_nodes, p_net_feature_dim, p_net_edge_dim, v_net_feature_dim, v_net_edge_dim, embedding_dim, dropout_prob=dropout_prob, batch_norm=batch_norm)
        
    def act(self, x):
        x = self.encoder(x)
        return self.actor(x)
    
    def evaluate(self, x):
        x = self.encoder(x)
        return self.critic(x)

    def evaluate_cost(self, x):
        x = self.encoder(x)
        return self.cost_critic(x)

    def evaluate_lambda(self, x):
        x = self.encoder(x)
        return self.lambda_net(x)

    def contrastive_learning(self, x):
        obs_a = {'v_net': x['aug_v_net'], 'p_net': x['p_net'], 'curr_v_node_id': x['curr_v_node_id']}
        obs_b = {'v_net': x['v_net'], 'p_net': x['aug_p_net'], 'curr_v_node_id': x['curr_v_node_id']}
        p_node_dense_embeddings_a = self.encoder(obs_a).reshape(-1, self.embedding_dim)
        p_node_dense_embeddings_b = self.encoder(obs_b).reshape(-1, self.embedding_dim)
        return p_node_dense_embeddings_a, p_node_dense_embeddings_b


class Actor(nn.Module):

    def __init__(self, p_net_num_nodes, p_net_feature_dim, p_net_edge_dim, v_net_feature_dim, v_net_edge_dim, embedding_dim=128, dropout_prob=0., batch_norm=False):
        super(Actor, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, state_embeddings):
        logits = self.head(state_embeddings).squeeze(-1)
        return logits


class Critic(nn.Module):
    
    def __init__(self, p_net_num_nodes, p_net_feature_dim, p_net_edge_dim, v_net_feature_dim, v_net_edge_dim, embedding_dim=128, dropout_prob=0., batch_norm=False):
        super(Critic, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )
        
    def forward(self, state_embeddings):
        fusion = self.head(state_embeddings).squeeze(-1)
        value = torch.mean(fusion, dim=-1, keepdim=True)
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
        return node_embeddings, graph_embedding, node_dense_embeddings_with_graph_embedding, net_mask


class BaseModel(nn.Module):

    def __init__(self, p_net_num_nodes, p_net_feature_dim, p_net_edge_dim, v_net_feature_dim, v_net_edge_dim, embedding_dim=128, dropout_prob=0., batch_norm=False):
        super(BaseModel, self).__init__()
        self.v_net_encoder = NetEncoder(v_net_feature_dim, v_net_edge_dim, embedding_dim=embedding_dim, dropout_prob=dropout_prob, batch_norm=batch_norm)
        self.p_net_encoder = NetEncoder(p_net_feature_dim, p_net_edge_dim, embedding_dim=embedding_dim, dropout_prob=dropout_prob, batch_norm=batch_norm)
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
        curr_v_node_id = obs['curr_v_node_id']
        v_node_embeddings, v_graph_embedding, v_node_dense_embeddings, v_net_mask = self.v_net_encoder(v_net_batch)
        p_node_embeddings, p_graph_embedding, p_node_dense_embeddings, p_net_mask = self.p_net_encoder(p_net_batch)
        curr_v_node_id = curr_v_node_id.unsqueeze(1).unsqueeze(1).long()
        curr_v_node_embeding = v_node_dense_embeddings.gather(1, curr_v_node_id.expand(v_node_dense_embeddings.size()[0], -1, v_node_dense_embeddings.size()[-1])).squeeze(1)
        p_node_dense_embeddings = p_node_dense_embeddings + v_graph_embedding.unsqueeze(1) + curr_v_node_embeding.unsqueeze(1)
        return p_node_dense_embeddings