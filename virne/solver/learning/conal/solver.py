import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.data import Data, Batch, HeteroData

from virne.solver import registry
from virne.solver.learning.neural_network.loss import BarlowTwinsContrastiveLoss
from .instance_env import InstanceRLEnv, CostAwareInstanceRLEnv, SimpleRewardInstanceRLEnv
from .net import ActorCritic
from virne.solver.learning.rl_base import RLSolver, PPOSolver, A2CSolver, InstanceAgent, A3CSolver
from ..utils import get_pyg_data
from virne.solver.learning.rl_base.safe_instance_agent import SafeInstanceAgent
from virne.solver.learning.rl_base.safe_rl_solver import RobustNeuralLagrangianPPOSolver


@registry.register(
    solver_name='conal',
    solver_type='r_learning')
class ConalSolver(SafeInstanceAgent, RobustNeuralLagrangianPPOSolver):
    def __init__(self, controller, recorder, counter, **kwargs):
        self.use_baseline_solver = True
        SafeInstanceAgent.__init__(self, CostAwareInstanceRLEnv)
        RobustNeuralLagrangianPPOSolver.__init__(self, controller, recorder, counter, make_policy, obs_as_tensor, **kwargs)
        self.contrastive_loss_fn = BarlowTwinsContrastiveLoss()
        self.coef_contrastive_loss = 0.001
        self.contrastive_optimizer = torch.optim.Adam(self.policy.encoder.parameters(), lr=1e-3)
        self.augment_ratio = kwargs.get('augment_ratio', 1.)
        self.compute_cost_method = 'reachability'
        self.buffer.extend('feasibility_flags')
        self.buffer.extend('feasibility_budgets')
        print(f'Compute cost method: {self.compute_cost_method}')
        self.if_allow_baseline_unsafe_solve = False

    def merge_instance_experience(self, instance, solution, instance_buffer, last_value):
        baseline_solution_info = self.get_baseline_solution_info(instance, self.use_baseline_solver)
        instance_buffer.extend('feasibility_flags')
        instance_buffer.feasibility_flags = [float(baseline_solution_info['result'])] * instance_buffer.size()
        instance_buffer.extend('feasibility_budgets')
        instance_buffer.feasibility_budgets = [float(baseline_solution_info['v_net_max_single_step_positive_violation']) / 1000] * instance_buffer.size()
        instance_buffer.compute_returns_and_advantages(last_value, gamma=self.gamma, gae_lambda=self.gae_lambda, method=self.compute_advantage_method)
        instance_buffer.compute_cost_returns(gamma=self.cost_gamma, method=self.compute_cost_method)
        self.buffer.merge(instance_buffer)
        self.time_step += 1
        return self.buffer

    def load_model(self, checkpoint_path):
        """Due to the use of hetero data, we need to load the model in a special way."""
        self.eval()
        with open('test_instance.npy', 'rb') as f:
            test_instance = np.load(f, allow_pickle=True).item()
        self.solve(test_instance)
        return super().load_model(checkpoint_path) 

    def update(self, avg_cost):
        assert self.buffer.size() >= self.batch_size
        device = torch.device('cpu')
        # batch_observations = self.preprocess_obs(self.buffer.observations, device)
        batch_actions = torch.LongTensor(self.buffer.actions)
        batch_old_action_logprobs = torch.FloatTensor(np.concatenate(self.buffer.logprobs, axis=0))
        batch_rewards = torch.FloatTensor(self.buffer.rewards)
        batch_costs = torch.FloatTensor(self.buffer.costs)
        batch_cost_returns = torch.FloatTensor(self.buffer.cost_returns)
        batch_returns = torch.FloatTensor(self.buffer.returns)
        batch_feasibiliy_flags = torch.FloatTensor(self.buffer.feasibility_flags)
        batch_feasibility_budgets = torch.FloatTensor(self.buffer.feasibility_budgets)
        batch_cost_violations = batch_cost_returns - self.cost_budget
        # update the policy params repeatly
        # sample_times = 1 + int(self.buffer.size() * self.repeat_times / self.batch_size)
        sample_times = self.repeat_times
        for i in range(sample_times):
            sample_indices = torch.randint(0, self.buffer.size(), size=(self.batch_size,)).long()
            sample_obersevations = [self.buffer.observations[i] for i in sample_indices]
            observations = self.preprocess_obs(sample_obersevations, self.device)
            actions, returns = batch_actions[sample_indices].to(self.device), batch_returns[sample_indices].to(self.device)
            cost_returns = batch_cost_returns[sample_indices].to(self.device)
            feasibility_flags = batch_feasibiliy_flags[sample_indices].to(self.device)
            feasibility_budgets = batch_feasibility_budgets[sample_indices].to(self.device)
            old_action_logprobs = batch_old_action_logprobs[sample_indices].to(self.device)
            # evaluate actions and observations
            values, action_logprobs, dist_entropy, other = self.evaluate_actions(observations, actions, return_others=True)
            cost_values = self.estimate_cost_with_grad(observations)
            # calculate advantage
            advantages = returns - values.detach()
            cost_advantages = (cost_returns - self.cost_budget) - (cost_values.detach() - self.cost_budget)
            if self.norm_advantage and values.numel() != 0:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)
                cost_advantages = (cost_advantages - cost_advantages.mean()) / (cost_advantages.std() + 1e-9)
            ratio = torch.exp(action_logprobs - old_action_logprobs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1. - self.eps_clip, 1. + self.eps_clip) * advantages
            # calculate loss
            cur_penalty = self.calc_penalty_params(observations).detach()
            # cur_penalty = torch.ones(size=(1,), device=self.device) * 1000
            reward_loss = (- torch.min(surr1, surr2)).mean()
            #   * feasibility_flags
            cost_loss = (cur_penalty * (ratio * cost_advantages)).mean()
            actor_loss = reward_loss + cost_loss
            critic_loss = self.criterion_critic(returns, values)
            cost_critic_loss = self.criterion_cost_critic(cost_returns, cost_values)
            entropy_loss = dist_entropy.mean()
            mask_loss = other.get('mask_actions_probs', 0)

            if (self.update_time+1) % (self.repeat_times) == 0:
                # Compute lam_net loss
                # current_penalty = torch.ones(size=(1,), device=self.device) * 1000
                print(f'feasibility_flags: {feasibility_flags.mean().item():.4f}')
                print(f'batch_feasibility_budgets: {batch_feasibility_budgets.mean().item():.4f}')
                cur_penalty = self.calc_penalty_params(observations)
                cost_values = self.estimate_cost(observations).detach()
                cur_penalty = cur_penalty * feasibility_flags
                penalty_loss = -(cur_penalty * (cost_values - feasibility_budgets)).mean()
                
                # penalty_loss = torch.zeros(size=(1,), device=self.device).mean()

                print(f'curr_penalty: {cur_penalty.mean().item():.4f},  cost_budget {batch_feasibility_budgets.mean().item():.4f}, penalty_loss: {penalty_loss.mean().item():.4f},  cost_returns: {cost_returns.mean().item():.4f},  cost_values: {cost_values.mean().item():.4f}, cost_critic_loss: {cost_critic_loss.mean().item():.4f}')
            else:
                penalty_loss = torch.zeros(size=(1,), device=self.device).mean()


            loss = actor_loss \
                    + self.coef_critic_loss * critic_loss \
                    + self.coef_cost_critic_loss * cost_critic_loss \
                    - self.coef_entropy_loss * entropy_loss \
                    + self.coef_mask_loss * mask_loss \
                    + penalty_loss
            # update parameters
            self.optimizer.zero_grad()
            loss.backward()
            if self.clip_grad:
                grad_clipped = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
    
            # contrastive learning
            p_node_embeddings_a, p_node_embeddings_b = self.policy.contrastive_learning(observations)
            contrastive_loss = self.contrastive_loss_fn(p_node_embeddings_a, p_node_embeddings_b)
            self.contrastive_optimizer.zero_grad()
            (self.coef_contrastive_loss * contrastive_loss).backward()
            self.contrastive_optimizer.step()


            if self.open_tb and self.update_time % self.log_interval == 0:
                info = {
                    'lr': self.optimizer.defaults['lr'],
                    'loss/loss': loss.detach().cpu().numpy(),
                    'loss/actor_loss': actor_loss.detach().cpu().numpy(),
                    'loss/critic_loss': critic_loss.detach().cpu().numpy(),
                    'loss/cost_critic_loss': cost_critic_loss.detach().cpu().numpy(),
                    'loss/entropy_loss': entropy_loss.detach().cpu().numpy(),
                    'loss/constrastive_loss': contrastive_loss.detach().cpu().numpy(),
                    'loss/penalty_loss': penalty_loss.detach().cpu().numpy(),
                    'value/logprob': action_logprobs.detach().mean().cpu().numpy(),
                    'value/old_action_logprob': old_action_logprobs.mean().cpu().numpy(),
                    'value/value': values.detach().mean().cpu().numpy(),
                    'value/advantage': advantages.detach().mean().cpu().numpy(),
                    'value/return': returns.mean().cpu().numpy(),
                    'cost_value/cost_value': cost_values.detach().mean().cpu().numpy(),
                    'cost_value/cost_advantage': cost_advantages.detach().mean().cpu().numpy(),
                    'cost_value/cost_budget': self.cost_budget,
                    'cost_value/cost_return': batch_cost_returns.mean().cpu().numpy(),
                    'cost_value/cost_violation': batch_cost_violations.mean().cpu().numpy(),
                    'cost_value/cost': batch_costs.mean().cpu().numpy(),
                    # 'penalty_params': cur_penalty.mean().cpu().numpy(),
                    'grad/grad_clipped': grad_clipped.detach().cpu().numpy()
                }
                only_tb = not (i == sample_times-1)
                self.log(info, self.update_time, only_tb=only_tb)

            self.update_time += 1

            if self.use_baseline_solver and self.update_time % 100 == 0:
                self.baseline_solver.policy.load_state_dict(self.policy.state_dict())
                print(f'Update time == {self.update_time},  baseline_solver updated')


        # print(f'loss: {loss.detach():+2.4f} = {actor_loss.detach():+2.4f} & {critic_loss:+2.4f} & {entropy_loss:+2.4f} & {mask_loss:+2.4f}, ' +
        #         f'action log_prob: {action_logprobs.mean():+2.4f} (old: {batch_old_action_logprobs.detach().mean():+2.4f}), ' +
        #         f'mean reward: {returns.detach().mean():2.4f}', file=self.fwriter) if self.verbose >= 0 else None
        self.lr_scheduler.step() if self.lr_scheduler is not None else None
        
        self.buffer.clear()
        if self.distributed_training: self.sync_parameters()
        return loss.detach()

@registry.register(
    solver_name='conal_wo_pc',
    solver_type='r_learning')
class CONALWoPcSolver(ConalSolver):
    def __init__(self, controller, recorder, counter, **kwargs):
        super().__init__(controller, recorder, counter, **kwargs)
        self.coef_contrastive_loss = 0.0
        self.use_baseline_solver = False
        self.if_allow_baseline_unsafe_solve = True

@registry.register(
    solver_name='conal_wo_afb',
    solver_type='r_learning')
class CONALWoAfbSolver(ConalSolver):
    def __init__(self, controller, recorder, counter, **kwargs):
        super().__init__(controller, recorder, counter, **kwargs)
        self.use_baseline_solver = False
        self.if_allow_baseline_unsafe_solve = False


@registry.register(
    solver_name='conal_wo_reach',
    solver_type='r_learning')
class CONALWoReachSolver(ConalSolver):
    def __init__(self, controller, recorder, counter, **kwargs):
        super().__init__(controller, recorder, counter, **kwargs)
        self.use_baseline_solver = False
        self.compute_cost_method = 'cumulative'
        self.if_allow_baseline_unsafe_solve = True


def make_policy(agent, **kwargs):
    num_vn_attrs = agent.v_sim_setting_num_node_resource_attrs
    num_vl_attrs = agent.v_sim_setting_num_link_resource_attrs
    policy = ActorCritic(p_net_num_nodes=agent.p_net_setting_num_nodes, 
                        # p_net_feature_dim=num_vn_attrs + num_vl_attrs*4*2 + 2 + 1 + 2 + 1, 
                        p_net_feature_dim=num_vn_attrs + 2 + 1 + 2 + 1, 
                        p_net_edge_dim=num_vl_attrs,
                        v_net_feature_dim=num_vn_attrs + num_vl_attrs*4 + 1 + 2 + 2,
                        v_net_edge_dim=num_vl_attrs,
                        embedding_dim=agent.embedding_dim, 
                        dropout_prob=agent.dropout_prob, 
                        batch_norm=agent.batch_norm).to(agent.device)
    optimizer = torch.optim.Adam([
            {'params': policy.actor.parameters(), 'lr': agent.lr_actor},
            {'params': policy.critic.parameters(), 'lr': agent.lr_critic},
            {'params': policy.cost_critic.parameters(), 'lr': agent.lr_critic},
            {'params': policy.lambda_net.parameters(), 'lr': agent.lr_critic / 10},
        ], weight_decay=agent.weight_decay
    )
    return policy, optimizer

def get_pyg_hetero_data(x_dict, edge_index_dict, edge_attr_dict, reverse_edge=True):
    """Preprocess the observation to adapt to batch mode."""
    hetero_data = HeteroData()
    for key, value in x_dict.items():
        hetero_data[key].x = torch.tensor(value, dtype=torch.float32)
    for key, value in edge_index_dict.items():
        hetero_data[key[0], key[1], key[2]].edge_index = torch.tensor(value).long()
    for key, value in edge_attr_dict.items():
        hetero_data[key[0], key[1], key[2]].edge_attr = torch.tensor(value, dtype=torch.float32)
    if reverse_edge:
        for key, value in edge_index_dict.items():
            if key[0] == key[2]: continue
            hetero_data[key[2], key[1], key[0]].edge_index = hetero_data[key[0], key[1], key[2]].edge_index.flip(0)
        for key, value in edge_attr_dict.items():
            if key[0] == key[2]: continue
            hetero_data[key[2], key[1], key[0]].edge_attr = hetero_data[key[0], key[1], key[2]].edge_attr.flip(0)
    return hetero_data


def obs_as_tensor(obs, device):
    # one
    if isinstance(obs, dict):
        """Preprocess the observation to adapt to batch mode."""
        observation = obs
        x_dict = {'p': observation['p_net_x'], 'v': observation['v_net_x']}
        edge_index_dict = {('p', 'connect', 'p'): observation['p_net_edge_index'], ('v', 'connect', 'v'): observation['v_net_edge_index'], ('v', 'mapping', 'p'): observation['vp_mapping_edge_index'], ('v', 'imaginary', 'p'): observation['vp_imaginary_edge_index']}
        edge_attr_dict = {('p', 'connect', 'p'): observation['p_net_edge_attr'], ('v', 'connect', 'v'): observation['v_net_edge_attr'], ('v', 'mapping', 'p'): observation['vp_mapping_edge_attr'], ('v', 'imaginary', 'p'): observation['vp_imaginary_edge_attr']}
        hetero_data = get_pyg_hetero_data(x_dict, edge_index_dict, edge_attr_dict)
        aug_edge_index_dict_a = {('p', 'connect', 'p'): observation['p_net_edge_index'], ('v', 'connect', 'v'): observation['v_net_aug_edge_index'], ('v', 'mapping', 'p'): observation['vp_mapping_edge_index'], ('v', 'imaginary', 'p'): observation['vp_imaginary_edge_index']}
        aug_edge_attr_dict_a = {('p', 'connect', 'p'): observation['p_net_edge_attr'], ('v', 'connect', 'v'): observation['v_net_aug_edge_attr'], ('v', 'mapping', 'p'): observation['vp_mapping_edge_attr'], ('v', 'imaginary', 'p'): observation['vp_imaginary_edge_attr']}
        aug_edge_index_dict_b = {('p', 'connect', 'p'): observation['p_net_aug_edge_index'], ('v', 'connect', 'v'): observation['v_net_edge_index'], ('v', 'mapping', 'p'): observation['vp_mapping_edge_index'], ('v', 'imaginary', 'p'): observation['vp_imaginary_edge_index']}
        aug_edge_attr_dict_b = {('p', 'connect', 'p'): observation['p_net_aug_edge_attr'], ('v', 'connect', 'v'): observation['v_net_edge_attr'], ('v', 'mapping', 'p'): observation['vp_mapping_edge_attr'], ('v', 'imaginary', 'p'): observation['vp_imaginary_edge_attr']}
        aug_hetero_data_a = get_pyg_hetero_data(x_dict, aug_edge_index_dict_a, aug_edge_attr_dict_a)
        aug_hetero_data_b = get_pyg_hetero_data(x_dict, aug_edge_index_dict_b, aug_edge_attr_dict_b)
        obs_aug_hetero_data_a = Batch.from_data_list([aug_hetero_data_a]).to(device)
        obs_aug_hetero_data_b = Batch.from_data_list([aug_hetero_data_b]).to(device)
        obs_hetero_data = Batch.from_data_list([hetero_data]).to(device)
        obs_curr_v_node_id = torch.LongTensor(np.array([observation['curr_v_node_id']])).to(device)
        obs_action_mask = torch.FloatTensor(np.array([observation['action_mask']])).to(device)
        obs_v_net_size = torch.LongTensor(np.array([observation['v_net_size']])).to(device)
        return {'hetero_data': obs_hetero_data, 'curr_v_node_id': obs_curr_v_node_id, 'action_mask': obs_action_mask, 'v_net_size': obs_v_net_size, 'aug_hetero_data_a': obs_aug_hetero_data_a, 'aug_hetero_data_b': obs_aug_hetero_data_b}
    # batch
    elif isinstance(obs, list):
        p_net_data_list, v_net_data_list, hetero_data_list, curr_v_node_id_list, action_mask_list, v_net_size_list = [], [], [], [], [], []
        aug_hetero_data_list_a, aug_hetero_data_list_b = [], []
        for observation in obs:
            x_dict = {'p': observation['p_net_x'], 'v': observation['v_net_x']}
            edge_index_dict = {('p', 'connect', 'p'): observation['p_net_edge_index'], ('v', 'connect', 'v'): observation['v_net_edge_index'], ('v', 'mapping', 'p'): observation['vp_mapping_edge_index'], ('v', 'imaginary', 'p'): observation['vp_imaginary_edge_index']}
            edge_attr_dict = {('p', 'connect', 'p'): observation['p_net_edge_attr'], ('v', 'connect', 'v'): observation['v_net_edge_attr'], ('v', 'mapping', 'p'): observation['vp_mapping_edge_attr'], ('v', 'imaginary', 'p'): observation['vp_imaginary_edge_attr']}
            hetero_data = get_pyg_hetero_data(x_dict, edge_index_dict, edge_attr_dict)
            hetero_data_list.append(hetero_data)
            aug_edge_index_dict_a = {('p', 'connect', 'p'): observation['p_net_edge_index'], ('v', 'connect', 'v'): observation['v_net_aug_edge_index'], ('v', 'mapping', 'p'): observation['vp_mapping_edge_index'], ('v', 'imaginary', 'p'): observation['vp_imaginary_edge_index']}
            aug_edge_attr_dict_a = {('p', 'connect', 'p'): observation['p_net_edge_attr'], ('v', 'connect', 'v'): observation['v_net_aug_edge_attr'], ('v', 'mapping', 'p'): observation['vp_mapping_edge_attr'], ('v', 'imaginary', 'p'): observation['vp_imaginary_edge_attr']}
            aug_edge_index_dict_b = {('p', 'connect', 'p'): observation['p_net_aug_edge_index'], ('v', 'connect', 'v'): observation['v_net_edge_index'], ('v', 'mapping', 'p'): observation['vp_mapping_edge_index'], ('v', 'imaginary', 'p'): observation['vp_imaginary_edge_index']}
            aug_edge_attr_dict_b = {('p', 'connect', 'p'): observation['p_net_aug_edge_attr'], ('v', 'connect', 'v'): observation['v_net_edge_attr'], ('v', 'mapping', 'p'): observation['vp_mapping_edge_attr'], ('v', 'imaginary', 'p'): observation['vp_imaginary_edge_attr']}
            aug_hetero_data_a = get_pyg_hetero_data(x_dict, aug_edge_index_dict_a, aug_edge_attr_dict_a)
            aug_hetero_data_b = get_pyg_hetero_data(x_dict, aug_edge_index_dict_b, aug_edge_attr_dict_b)
            aug_hetero_data_list_a.append(aug_hetero_data_a)
            aug_hetero_data_list_b.append(aug_hetero_data_b)
            curr_v_node_id_list.append(observation['curr_v_node_id'])
            action_mask_list.append(observation['action_mask'])
            v_net_size_list.append(observation['v_net_size'])
        obs_hetero_data = Batch.from_data_list(hetero_data_list).to(device)
        obs_aug_hetero_data_a = Batch.from_data_list(aug_hetero_data_list_a).to(device)
        obs_aug_hetero_data_b = Batch.from_data_list(aug_hetero_data_list_b).to(device)
        obs_curr_v_node_id = torch.LongTensor(np.array(curr_v_node_id_list)).to(device)
        obs_v_net_size = torch.FloatTensor(np.array(v_net_size_list)).to(device)
        # Get the length of the longest sequence
        max_len_action_mask = max(len(seq) for seq in action_mask_list)
        # Pad all sequences with zeros up to the max length
        padded_action_mask = np.zeros((len(action_mask_list), max_len_action_mask))
        for i, seq in enumerate(action_mask_list):
            padded_action_mask[i, :len(seq)] = seq
        obs_action_mask = torch.FloatTensor(np.array(padded_action_mask)).to(device)
        # return {'p_net': obs_p_net, 'v_net': obs_v_net, 'hetero_data': obs_hetero_data, 'curr_v_node_id': obs_curr_v_node_id, 'action_mask': obs_action_mask, 'v_net_size': obs_v_net_size}
        return {'hetero_data': obs_hetero_data, 'curr_v_node_id': obs_curr_v_node_id, 'action_mask': obs_action_mask, 'v_net_size': obs_v_net_size, 'aug_hetero_data_a': obs_aug_hetero_data_a, 'aug_hetero_data_b': obs_aug_hetero_data_b}
    else:
        raise Exception(f"Unrecognized type of observation {type(obs)}")
    