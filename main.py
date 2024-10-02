import os
from args import get_args
from virne.base import BasicScenario
from virne import Config, REGISTRY, Generator, update_simulation_setting


def run(config):
    print(f"\n{'-' * 20}    Start     {'-' * 20}\n")
    # Load solver info: environment and solver class
    solver_info = REGISTRY.get(config.solver_name)
    Env, Solver = solver_info['env'], solver_info['solver']
    print(f'Use {config.solver_name} Solver (Type = {solver_info["type"]})...\n')

    scenario = BasicScenario.from_config(Env, Solver, config)
    scenario.run()

    print(f"\n{'-' * 20}   Complete   {'-' * 20}\n")


if __name__ == '__main__':
    # 1. Get config / Load config
    args = get_args()
    config = Config(p_net_setting_path=args.p_net_setting_path, v_sim_setting_path=args.v_sim_setting_path)
    config.update(args)
    ### --- Simulation --- ###
    # print(args.p_net_topology)
    pn_generation_flag = False
    if args.p_net_topology.lower() == 'wx100':
        config.p_net_setting['topology']['file_path'] = 'dataset/topology/Waxman100.gml'
        config.p_net_setting['num_nodes'] = 100
        config.p_net_setting_num_nodes = 100
        v_sim_setting_aver_arrival_rate_list = [0.06, 0.08, 0.10, 0.12, 0.14, 0.16]
    elif args.p_net_topology.lower() == 'brain':
        config.p_net_setting['topology']['file_path'] = 'dataset/topology/Brain.gml'
        config.p_net_setting['num_nodes'] = 161
        config.p_net_setting_num_nodes = 161
        v_sim_setting_aver_arrival_rate_list = [0.06, 0.08, 0.10, 0.12, 0.14, 0.16]
    elif args.p_net_topology.lower() == 'geant':
        config.p_net_setting['topology']['file_path'] = 'dataset/topology/Geant.gml'
        config.p_net_setting['num_nodes'] = 40
        config.p_net_setting_num_nodes = 40
        # v_sim_setting_aver_arrival_rate_list = [0.0001, 0.0003, 0.0005, 0.0007, 0.0009, 0.0011]
        v_sim_setting_aver_arrival_rate_list = [i * 0.0001 for i in range(17, 24)]
    elif args.p_net_topology.lower() == 'wx500':
        config.p_net_setting['topology']['file_path'] = 'dataset/topology/Waxman500.gml'
        config.p_net_setting['num_nodes'] = 500
        config.p_net_setting_num_nodes = 500
    elif args.p_net_topology.lower() == 'wxn':
        config.p_net_setting['topology'].pop('file_path') if 'file_path' in config.p_net_setting['topology'] else None
        assert args.p_net_num_nodes > 0
        config.p_net_setting['num_nodes'] = args.p_net_num_nodes
        config.p_net_setting_num_nodes = args.p_net_num_nodes
        pn_generation_flag = True
    else:
        raise NotImplementedError
    update_simulation_setting(
        config, 
        v_sim_setting_num_v_nets=args.v_sim_setting_num_v_nets,
        v_sim_setting_v_net_size_low=args.v_sim_setting_v_net_size_low,
        v_sim_setting_v_net_size_high=args.v_sim_setting_v_net_size_high,
        v_sim_setting_node_resource_attrs_low=args.v_sim_setting_node_resource_attrs_low,
        v_sim_setting_node_resource_attrs_high=args.v_sim_setting_node_resource_attrs_high,
        v_sim_setting_link_resource_attrs_low=args.v_sim_setting_link_resource_attrs_low,
        v_sim_setting_link_resource_attrs_high=args.v_sim_setting_link_resource_attrs_high,
        v_sim_setting_aver_arrival_rate=args.v_sim_setting_aver_arrival_rate,
        v_sim_setting_aver_lifetime=args.v_sim_setting_aver_lifetime,
    )
    ### ---    End     --- ###

    # 2. Generate Dataset
    # Already generated
    config.renew_v_net_simulator = False
    # p_net, v_net_simulator = Generator.generate_dataset(config, p_net=True, v_nets=False, save=True)
    p_net, v_net_simulator = Generator.generate_dataset(config, p_net=pn_generation_flag, v_nets=False, save=False)

    if config.if_dynamic_v_nets:
        v_net_simulator = Generator.generate_dynamic_v_nets_dataset_from_config(config, save=False)
    run(config)
