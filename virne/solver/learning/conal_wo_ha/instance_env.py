import numpy as np
import networkx as nx
from gym import spaces
from virne.solver.learning.rl_base import JointPRStepInstanceRLEnv

from ..utils import *


class InstanceRLEnv(JointPRStepInstanceRLEnv):

    def __init__(self, p_net, v_net, controller, recorder, counter, **kwargs):
        kwargs['node_ranking_method'] = 'nrm'
        super(InstanceRLEnv, self).__init__(p_net, v_net, controller, recorder, counter, **kwargs)
        self.check_feasibility = False
        self.calcuate_graph_metrics(degree=True, closeness=False, eigenvector=False, betweenness=False)
        self.init_candidates_dict = self.controller.construct_candidates_dict(self.v_net, self.p_net)
    
    def get_observation(self):
        p_net_obs = self._get_p_net_obs()
        v_net_obs = self._get_v_net_obs()
        vp_obs = self._get_vp_obs()
        v_augumented_obs, p_augumented_obs = self.get_augumented_observation(v_net_obs, p_net_obs)
        return {
            'p_net_x': p_net_obs['x'],
            'p_net_edge_index': p_net_obs['edge_index'],
            'p_net_edge_attr': p_net_obs['edge_attr'],
            'v_net_x': v_net_obs['x'],
            'v_net_edge_index': v_net_obs['edge_index'],
            'v_net_edge_attr': v_net_obs['edge_attr'],
            'curr_v_node_id': self.curr_v_node_id,
            'v_net_size': self.v_net.num_nodes,
            'action_mask': self.generate_action_mask(),

            'vp_mapping_edge_index': vp_obs['mapping_edge_index'],
            'vp_mapping_edge_attr': vp_obs['mapping_edge_attr'],
            'vp_imaginary_edge_index': vp_obs['imaginary_edge_index'],
            'vp_imaginary_edge_attr': vp_obs['imaginary_edge_attr'],

            'v_net_aug_edge_index': v_augumented_obs['v_net_aug_edge_index'],
            'v_net_aug_edge_attr': v_augumented_obs['v_net_aug_edge_attr'],
            'p_net_aug_edge_index': p_augumented_obs['p_net_aug_edge_index'],
            'p_net_aug_edge_attr': p_augumented_obs['p_net_aug_edge_attr'],
        }

    def _get_vp_obs(self):
        # mapping edge
        vp_mapping_edge_index = []
        vp_mapping_edge_attr = []
        for v_node_id, p_node_id in self.solution['node_slots'].items():
            vp_mapping_edge_index.append([v_node_id, p_node_id])
            vp_mapping_edge_attr.append([1.])
        vp_mapping_edge_index = np.array(vp_mapping_edge_index).astype(np.int64).reshape(-1, 2).T
        vp_mapping_edge_attr = np.array(vp_mapping_edge_attr) if len(vp_mapping_edge_attr) != 0 else np.array([[]])
        if vp_mapping_edge_attr.shape == (1, 0):
            vp_mapping_edge_attr = vp_mapping_edge_attr.T
        
        # if len(vp_mapping_edge_index) != 0:
            # vp_mapping_edge_index = np.concatenate([vp_mapping_edge_index, vp_mapping_edge_index[:, [1,0]]], axis=0).T
            # vp_mapping_edge_attr = np.concatenate([vp_mapping_edge_attr, vp_mapping_edge_attr], axis=0)
        # imaginary edge
        vp_imaginary_edge_index = []
        vp_imaginary_edge_attr = []
        for p_node_id in self.init_candidates_dict[self.curr_v_node_id]:
            if p_node_id not in self.selected_p_net_nodes:
                vp_imaginary_edge_index.append([self.curr_v_node_id, p_node_id])
                vp_imaginary_edge_attr.append([1.])
        vp_imaginary_edge_index = np.array(vp_imaginary_edge_index).astype(np.int64).reshape(-1, 2).T
        vp_imaginary_edge_attr = np.array(vp_imaginary_edge_attr) if len(vp_imaginary_edge_attr) != 0 else np.array([[]])
        if vp_imaginary_edge_attr.shape == (1, 0):
            vp_imaginary_edge_attr = vp_imaginary_edge_attr.T
        # if len(vp_imaginary_edge_index) != 0:
            # vp_imaginary_edge_index = np.concatenate([vp_imaginary_edge_index, vp_imaginary_edge_index[:, [1,0]]], axis=0).T
            # vp_imaginary_edge_attr = np.concatenate([vp_imaginary_edge_attr, vp_imaginary_edge_attr], axis=0)
        vp_obs = {
            'mapping_edge_index': vp_mapping_edge_index,
            'mapping_edge_attr': vp_mapping_edge_attr,
            'imaginary_edge_index': vp_imaginary_edge_index,
            'imaginary_edge_attr': vp_imaginary_edge_attr,
        }
        return vp_obs

    def _get_p_net_obs(self, ):
        attr_type_list = ['resource']
        # node data
        node_data = self.obs_handler.get_node_attrs_obs(self.p_net, node_attr_types=attr_type_list, node_attr_benchmarks=self.node_attr_benchmarks)
        p_node_link_max_resource = self.obs_handler.get_link_aggr_attrs_obs(self.p_net, link_attr_types=attr_type_list, aggr='max', link_attr_benchmarks=self.link_attr_benchmarks)
        p_node_link_sum_resource = self.obs_handler.get_link_aggr_attrs_obs(self.p_net, link_attr_types=attr_type_list, aggr='sum', link_sum_attr_benchmarks=self.link_sum_attr_benchmarks)
        p_node_link_mean_resource = self.obs_handler.get_link_aggr_attrs_obs(self.p_net, link_attr_types=attr_type_list, aggr='mean', link_sum_attr_benchmarks=self.link_attr_benchmarks)
        p_nodes_status = self.obs_handler.get_p_nodes_status(self.p_net, self.v_net, self.solution['node_slots'], self.curr_v_node_id)
        v_node_sizes = np.ones((self.p_net.num_nodes, 1), dtype=np.float32) * self.v_net.num_nodes / 10
        avg_distance = self.obs_handler.get_average_distance(self.p_net, self.solution['node_slots'], normalization=True)
        node_data = np.concatenate((node_data, v_node_sizes, p_nodes_status, p_node_link_sum_resource, p_node_link_max_resource, p_node_link_mean_resource, self.p_net_node_degrees, avg_distance), axis=-1)
        edge_index = self.obs_handler.get_link_index_obs(self.p_net)
        link_data = self.obs_handler.get_link_attrs_obs(self.p_net, link_attr_types=attr_type_list, link_attr_benchmarks=self.link_attr_benchmarks)
        # data
        p_net_obs = {
            'x': node_data,
            'edge_index': edge_index,
            'edge_attr': link_data,
        }
        return p_net_obs

    def _get_v_net_obs(self):
        # node data
        attr_type_list = ['resource']
        v_node_sizes = np.ones((self.v_net.num_nodes, 1), dtype=np.float32) * self.v_net.num_nodes / 10
        node_data = self.obs_handler.get_node_attrs_obs(self.v_net, node_attr_types=['resource'], node_attr_benchmarks=self.node_attr_benchmarks)
        v_node_status = self.obs_handler.get_v_nodes_status(self.v_net, self.solution['node_slots'], self.curr_v_node_id, consist_decision=True)
        v_node_link_max_resource = self.obs_handler.get_link_aggr_attrs_obs(self.v_net, link_attr_types=['resource'], aggr='max', link_attr_benchmarks=self.link_attr_benchmarks)
        v_node_link_sum_resource = self.obs_handler.get_link_aggr_attrs_obs(self.v_net, link_attr_types=['resource'], aggr='sum', link_sum_attr_benchmarks=self.link_sum_attr_benchmarks)
        v_node_link_mean_resource = self.obs_handler.get_link_aggr_attrs_obs(self.v_net, link_attr_types=['resource'], aggr='mean', link_sum_attr_benchmarks=self.link_attr_benchmarks)
        node_data = np.concatenate((node_data, v_node_sizes, v_node_status, v_node_link_max_resource, v_node_link_sum_resource, v_node_link_mean_resource), axis=-1)
        link_data = self.obs_handler.get_link_attrs_obs(self.v_net, link_attr_types=['resource'], link_attr_benchmarks=self.link_attr_benchmarks)
        # edge_index
        edge_index = self.obs_handler.get_link_index_obs(self.v_net)
        # edge_attr
        v_net_obs = {
            'x': node_data,
            'edge_index': edge_index,
            'edge_attr': link_data,
        }
        return v_net_obs

    def get_augumented_observation(self, v_net_obs, p_net_obs):
        # v_num_added_links = int(self.v_net.num_edges / 2)
        v_num_added_links = min(self.v_net.num_nodes, self.v_net.num_edges)
        v_non_existence_link_pairs = get_unexistent_link_pairs(self.v_net)
        v_added_link_pairs = get_random_unexistent_links(v_non_existence_link_pairs, v_num_added_links)
        v_net_aug_edge_index = np.concatenate((v_net_obs['edge_index'], v_added_link_pairs.T), axis=-1)
        v_net_aug_edge_attr = np.concatenate((v_net_obs['edge_attr'], np.zeros((v_added_link_pairs.shape[0], 1))), axis=0)
        v_net_aug_edge_index, v_net_aug_edge_attr = sort_edge_index(v_net_aug_edge_index, v_net_aug_edge_attr, num_nodes=self.v_net.num_nodes, sort_by_row=True)
        v_augumented_obs = {
            'v_net_aug_edge_index': v_net_aug_edge_index,
            'v_net_aug_edge_attr': v_net_aug_edge_attr,
        }
        v_bw_resource = nx.get_edge_attributes(self.v_net, 'bw')
        unrouted_link_list = list(set(self.v_net.edges()) - set(self.solution.link_paths.keys()))
        unrouted_link_resource_list = [v_bw_resource[link] for link in unrouted_link_list]
        min_unrouted_link_resource = min(unrouted_link_resource_list) if len(unrouted_link_resource_list) !=0 else 0
        # p_num_added_links = int(self.p_net.num_edges / 2)
        p_num_added_links = self.p_net.num_nodes
        # p_num_added_links = self.p_net.num_edges * 2
        p_non_existence_link_pairs = get_unexistent_link_pairs(self.p_net)
        p_added_link_pairs = get_random_unexistent_links(p_non_existence_link_pairs, p_num_added_links)
        p_net_aug_edge_index = np.concatenate((p_net_obs['edge_index'], p_added_link_pairs.T), axis=-1)
        p_net_added_edge_attr = np.ones((p_added_link_pairs.shape[0], 1)) * (min_unrouted_link_resource - 1) / self.link_attr_benchmarks['bw']
        p_net_aug_edge_attr = np.concatenate((p_net_obs['edge_attr'], p_net_added_edge_attr), axis=0)
        p_net_aug_edge_index, p_net_aug_edge_attr = sort_edge_index(p_net_aug_edge_index, p_net_aug_edge_attr, num_nodes=self.p_net.num_nodes, sort_by_row=True)
        p_augumented_obs = {
            'p_net_aug_edge_index': p_net_aug_edge_index,
            'p_net_aug_edge_attr': p_net_aug_edge_attr,
        }
        return v_augumented_obs, p_augumented_obs

    def compute_reward(self, solution):
        """Calculate deserved reward according to the result of taking action."""
        weight = (1 / self.v_net.num_nodes)
        weight = 0
        if solution['result']:
            reward = solution['v_net_r2c_ratio']
        elif solution['place_result'] and solution['route_result']:
            reward = weight
        else:
            reward = - weight
        self.solution['v_net_reward'] += reward
        return reward
