import copy
import time
import tqdm
import torch
import pprint
import numpy as np

from .instance_agent import InstanceAgent
from .buffer import RolloutBufferWithCost


class SafeInstanceAgent(InstanceAgent):

    def __init__(self, InstanceEnv):
        super(SafeInstanceAgent, self).__init__(InstanceEnv)
        self.compute_advantage_method = 'mc'

    def unsafe_solve(self, instance):
        v_net, p_net = instance['v_net'], instance['p_net']
        instance_env = self.InstanceEnv(p_net, v_net, self.controller, self.recorder, self.counter, **self.basic_config)
        instance_env.check_feasibility = False
        solution = self.searcher.find_solution(instance_env)
        return solution

    def solve(self, instance):
        v_net, p_net = instance['v_net'], instance['p_net']
        instance_env = self.InstanceEnv(p_net, v_net, self.controller, self.recorder, self.counter, **self.basic_config)
        instance_env.check_feasibility = True
        solution = self.searcher.find_solution(instance_env)
        return solution

    def learn_singly(self, env, num_epochs=1, **kwargs):
        # main env
        for epoch_id in range(num_epochs):
            print(f'Training Epoch: {epoch_id}') if self.verbose > 0 else None
            instance = env.reset()
            success_count = 0
            epoch_logprobs = []
            epoch_cost_list = []
            revenue2cost_list = []
            cost_list = []
            for i in range(env.num_v_nets):
                ### --- sub env --- ###
                sub_buffer = RolloutBufferWithCost()
                v_net, p_net = instance['v_net'], instance['p_net']
                instance_env = self.InstanceEnv(p_net, v_net, self.controller, self.recorder, self.counter, **self.basic_config)
                instance_obs = instance_env.get_observation()
                instance_done = False
                while not instance_done:
                    tensor_instance_obs = self.preprocess_obs(instance_obs, self.device)
                    action, action_logprob = self.select_action(tensor_instance_obs, sample=True)
                    value = self.estimate_value(tensor_instance_obs) if hasattr(self.policy, 'evaluate') else None
                    cost_value = self.estimate_cost(tensor_instance_obs) if hasattr(self.policy, 'evaluate_cost') else None
                    next_instance_obs, instance_reward, instance_done, instance_info = instance_env.step(action)
                    sub_buffer.add(instance_obs, action, instance_reward, instance_done, action_logprob, value=value)
                    single_step_violation = instance_info['v_net_single_step_violation'] / 1000 # / v_net.total_resource_demand
                    sub_buffer.costs.append(single_step_violation)
                    sub_buffer.cost_values.append(cost_value)
                    cost_list.append(single_step_violation)
                    if instance_done:
                        break
                    instance_obs = next_instance_obs
                last_value = self.estimate_value(self.preprocess_obs(next_instance_obs, self.device)) if hasattr(self.policy, 'evaluate') else None
                solution = instance_env.solution
                # print(f'{v_net.num_nodes:2d}', f'{sum(sub_buffer.costs):2.2f}', f'{sum(sub_buffer.costs)/ v_net.num_nodes:2.2f}', sub_buffer.costs)
                epoch_logprobs += sub_buffer.logprobs
                self.merge_instance_experience(instance, solution, sub_buffer, last_value)
                # instance_env.solution['result'] or self.use_negative_sample:  #  or True
                if solution.is_feasible():
                    success_count = success_count + 1
                    revenue2cost_list.append(solution['v_net_r2c_ratio'])
                # update parameters
                if self.buffer.size() >= self.target_steps:
                    avg_cost = sum(cost_list) / len(cost_list)
                    loss = self.update(avg_cost)
                    epoch_cost_list += cost_list
                    print(f'avg_cost: {avg_cost:+2.4f}, cost budget: {self.cost_budget:+2.4f}, loss: {loss.item():+2.4f}, mean r2c: {np.mean(revenue2cost_list):+2.4f}') if self.verbose > 0 else None
                ### --- sub env --- ###
                instance, reward, done, info = env.step(solution)
                # instance = env.reset()
                # epoch finished
                if done:
                    break
            epoch_logprobs_tensor = np.concatenate(epoch_logprobs, axis=0)
            avg_epoch_cost = sum(epoch_cost_list) / len(epoch_cost_list)
            print(f'\nepoch {epoch_id:4d}, success_count {success_count:5d}, r2c {info["long_term_r2c_ratio"]:1.4f}, avg_cost: {avg_epoch_cost:1.4f}, mean logprob {epoch_logprobs_tensor.mean():2.4f}') if self.verbose > 0 else None
            
            if self.rank == 0:
                if (epoch_id + 1) != num_epochs and (epoch_id + 1) % self.save_interval == 0:
                    self.save_model(f'model-{epoch_id}.pkl')
                if (epoch_id + 1) != num_epochs and (epoch_id + 1) % self.eval_interval == 0:
                    self.validate(env)


    def merge_instance_experience(self, instance, solution, instance_buffer, last_value):
        merge_flag = False
        if self.use_negative_sample:
            baseline_solution_info = self.get_baseline_solution_info(instance, self.use_baseline_solver)
            if baseline_solution_info['result'] or solution['result']:
                merge_flag = True
        elif solution['result']:
            merge_flag = True
        else:
            pass
        if merge_flag:
            instance_buffer.compute_returns_and_advantages(last_value, gamma=self.gamma, gae_lambda=self.gae_lambda, method=self.compute_advantage_method)
            instance_buffer.compute_cost_returns(gamma=self.gamma, method=self.compute_cost_method)
            self.buffer.merge(instance_buffer)
            self.time_step += 1
        return self.buffer