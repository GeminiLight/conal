import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.data import Data, Batch

from virne.solver import registry
from virne.solver.learning.neural_network.loss import BarlowTwinsContrastiveLoss
from .instance_env import InstanceRLEnv
from .net import ActorCritic
from virne.solver.learning.rl_base import RLSolver, PPOSolver, A2CSolver, InstanceAgent, SafeInstanceAgent, NeuralLagrangianPPOSolver
from virne.solver.learning.rl_base.safe_rl_solver import RobustNeuralLagrangianPPOSolver
from ..utils import get_pyg_data
from ..obs_handler import POSITIONAL_EMBEDDING_DIM


@registry.register(
    solver_name='conal_wo_ha',
    solver_type='r_learning')
class ConalWoHaPcSolver(SafeInstanceAgent, RobustNeuralLagrangianPPOSolver):
    def __init__(self, controller, recorder, counter, **kwargs):
        SafeInstanceAgent.__init__(self, InstanceRLEnv)
        RobustNeuralLagrangianPPOSolver.__init__(self, controller, recorder, counter, make_policy, obs_as_tensor, **kwargs)
        self.compute_cost_method = kwargs.get('compute_cost_method', 'reachability')
        # self.use_baseline_solver = False
        # self.if_allow_baseline_unsafe_solve = False
        self.buffer.extend('feasibility_flags')
        self.buffer.extend('feasibility_budgets')
        print(f'Compute cost method: {self.compute_cost_method}')
        self.contrastive_loss_fn = BarlowTwinsContrastiveLoss()
        self.coef_contrastive_loss = kwargs.get('coef_contrastive_loss', 0.001)
        self.contrastive_optimizer = torch.optim.Adam(self.policy.encoder.parameters(), lr=1e-3)
        self.augment_ratio = kwargs.get('augment_ratio', 1.)

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

def make_policy(agent, **kwargs):
    num_vn_attrs = agent.v_sim_setting_num_node_resource_attrs
    num_vl_attrs = agent.v_sim_setting_num_link_resource_attrs
    policy = ActorCritic(p_net_num_nodes=agent.p_net_setting_num_nodes, 
                        p_net_feature_dim=num_vn_attrs + num_vl_attrs*3 + 2 + 1 + 2, 
                        # p_net_feature_dim=num_vn_attrs + num_vl_attrs*3 + 2 + 1 + 2 + 6, 
                        p_net_edge_dim=num_vl_attrs,
                        v_net_feature_dim=num_vn_attrs + num_vl_attrs*3 + 2 + 1,
                        # v_net_feature_dim=num_vn_attrs + num_vl_attrs*3 + 2 + 1 + 3,
                        v_net_edge_dim=num_vl_attrs,
                        embedding_dim=agent.embedding_dim, 
                        dropout_prob=agent.dropout_prob, 
                        batch_norm=agent.batch_norm).to(agent.device)
    agent.lr_penalty_params = agent.lr_cost_critic / 10
    optimizer = torch.optim.Adam([
            {'params': policy.actor.parameters(), 'lr': agent.lr_actor},
            {'params': policy.critic.parameters(), 'lr': agent.lr_critic},
            {'params': policy.cost_critic.parameters(), 'lr': agent.lr_cost_critic},
            {'params': policy.lambda_net.parameters(), 'lr': agent.lr_penalty_params},
        ], weight_decay=agent.weight_decay
    )
    return policy, optimizer

@registry.register(
    solver_name='conal_wo_ha_pc',
    solver_type='r_learning')
class ConalWoHaPcReachSolver(ConalWoHaPcSolver):
    def __init__(self, controller, recorder, counter, **kwargs):
        super().__init__(controller, recorder, counter, **kwargs)
        self.compute_cost_method = 'reachability'
        self.use_baseline_solver = True


def obs_as_tensor(obs, device):
    # one
    if isinstance(obs, dict):
        """Preprocess the observation to adapt to batch mode."""
        observation = obs
        p_net_data = get_pyg_data(observation['p_net_x'], observation['p_net_edge_index'], observation['p_net_edge_attr'])
        v_net_data = get_pyg_data(observation['v_net_x'], observation['v_net_edge_index'], observation['v_net_edge_attr'])
        aug_p_net_data = get_pyg_data(observation['p_net_x'], observation['p_net_aug_edge_index'], observation['p_net_aug_edge_attr'])
        aug_v_net_data = get_pyg_data(observation['v_net_x'], observation['v_net_aug_edge_index'], observation['v_net_aug_edge_attr'])
        obs_p_net = Batch.from_data_list([p_net_data]).to(device)
        obs_v_net = Batch.from_data_list([v_net_data]).to(device)
        obs_curr_v_node_id = torch.LongTensor(np.array([observation['curr_v_node_id']])).to(device)
        obs_action_mask = torch.FloatTensor(np.array([observation['action_mask']])).to(device)
        obs_v_net_size = torch.LongTensor(np.array([observation['v_net_size']])).to(device)
        return {'p_net': obs_p_net, 'v_net': obs_v_net, 'curr_v_node_id': obs_curr_v_node_id, 'action_mask': obs_action_mask, 'v_net_size': obs_v_net_size, 'aug_p_net': aug_p_net_data, 'aug_v_net': aug_v_net_data}
    # batch
    elif isinstance(obs, list):
        p_net_data_list, v_net_data_list, curr_v_node_id_list, action_mask_list, v_net_size_list = [], [], [], [], []
        aug_p_net_data_list, aug_v_net_data_list = [], []
        for observation in obs:
            p_net_data = get_pyg_data(observation['p_net_x'], observation['p_net_edge_index'], observation['p_net_edge_attr'])
            p_net_data_list.append(p_net_data)
            v_net_data = get_pyg_data(observation['v_net_x'], observation['v_net_edge_index'], observation['v_net_edge_attr'])
            v_net_data_list.append(v_net_data)
            aug_p_net_data = get_pyg_data(observation['p_net_x'], observation['p_net_aug_edge_index'], observation['p_net_aug_edge_attr'])
            aug_p_net_data_list.append(aug_p_net_data)
            aug_v_net_data = get_pyg_data(observation['v_net_x'], observation['v_net_aug_edge_index'], observation['v_net_aug_edge_attr'])
            aug_v_net_data_list.append(aug_v_net_data)        
            curr_v_node_id_list.append(observation['curr_v_node_id'])
            action_mask_list.append(observation['action_mask'])
            v_net_size_list.append(observation['v_net_size'])
        obs_p_net = Batch.from_data_list(p_net_data_list).to(device)
        obs_v_net = Batch.from_data_list(v_net_data_list).to(device)
        obs_aug_p_net = Batch.from_data_list(aug_p_net_data_list).to(device)
        obs_aug_v_net = Batch.from_data_list(aug_v_net_data_list).to(device)
        obs_curr_v_node_id = torch.LongTensor(np.array(curr_v_node_id_list)).to(device)
        obs_v_net_size = torch.FloatTensor(np.array(v_net_size_list)).to(device)
        # Get the length of the longest sequence
        max_len_action_mask = max(len(seq) for seq in action_mask_list)
        # Pad all sequences with zeros up to the max length
        padded_action_mask = np.zeros((len(action_mask_list), max_len_action_mask))
        for i, seq in enumerate(action_mask_list):
            padded_action_mask[i, :len(seq)] = seq
        obs_action_mask = torch.FloatTensor(np.array(padded_action_mask)).to(device)
        return {'p_net': obs_p_net, 'v_net': obs_v_net, 'curr_v_node_id': obs_curr_v_node_id, 'action_mask': obs_action_mask, 'v_net_size': obs_v_net_size, 'aug_p_net': obs_aug_p_net, 'aug_v_net': obs_aug_v_net}
    else:
        raise Exception(f"Unrecognized type of observation {type(obs)}")
    