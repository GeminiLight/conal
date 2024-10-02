# ================== All solver list ================== #
# CONAL and its variants:
#       [conal, conal_wo_ha, conal_wo_pc, conal_wo_ha_pc, conal_wo_arb, conal_wo_reach]
# Heuristic Baselines: 
#       [nrm_rank, grc_rank, nea_rank, pso_vne, ga_vne]
# RL-based Baselines: 
#       [mcts, a3c_gcn, pg_cnn, ddpg_attention]
# ================== 1. Key Settings ================== #
solver_name="conal"                 # Solver name. Options: ALL_SOLVER_LIST
topology="wx100"                    # Topology name. Options: [geant, wx100, brain]
num_train_epochs=0                 # Number of training epochs. Options: [0, >0]. If 0, then inference only.
pretrained_model_path="null"        # Path to the pretrained model. Options: ['null', $path]. If inference, then the path must be valid.
k_shortest=5
# train_aver_arrival_rate=0.14 
v_sim_setting_v_net_size_low=2
v_sim_setting_v_net_size_high=10
v_sim_setting_node_resource_attrs_low=0
v_sim_setting_node_resource_attrs_high=20
num_workers=1                      # Number of parallel workers for A3C
cuda_device=0                      # Cuda device id
batch_size=128                     # Batch size
# ===================================================== #
if_dynamic_v_nets=1
# ===================================================== #
identifier=""

declare -A pretrained_model_path_dict_for_geant
declare -A pretrained_model_path_dict_for_wx100
use_pretrained_model=1
pretrained_model_path_dict_for_wx100["a3c_gcn"]="PRETRAINED_MODEL_PATH"
pretrained_model_path_dict_for_wx100["ddpg_attention"]="PRETRAINED_MODEL_PATH"
pretrained_model_path_dict_for_wx100["pg_cnn"]="PRETRAINED_MODEL_PATH"
pretrained_model_path_dict_for_wx100["conal"]="PRETRAINED_MODEL_PATH"

if [ $topology == "wx100" -a $use_pretrained_model == 1 ]; then
    pretrained_model_path=${pretrained_model_path_dict_for_wx100[$solver_name]}
else
    pretrained_model_path="null"
fi

echo "pretrained_model_path: $pretrained_model_path"

if [ $topology == "wx100" ]; then
    # wx100 topology
    if [ $num_train_epochs == "0" ]; then
        # inference setting
        aver_arrival_rate_list=(0.14)
        identifier="-test-$pretrained_model_name"
        identifier=""
        echo $pretrained_model_path
    else
        # pretrain setting
        aver_arrival_rate_list=($train_aver_arrival_rate)
        pretrained_model_path="null"
    fi
else
    echo "Error: topology $topology is not supported!"
    exit 1
fi

# Judge if the pretrained model exists. If inference, then the path must be valid.
if [ $pretrained_model_path == "null" ]; then
    echo "pretrained model path is null, skip the check"
else
    if [ ! -f $pretrained_model_path ]; then
        echo "Error: pretrained model $pretrained_model_path does not exist!"
        exit 1
    fi
fi

seed_list=$(seq 0 1111 9999)
for aver_arrival_rate in $aver_arrival_rate_list
do
    for seed in $seed_list
    do 
        echo aver_arrival_rate $aver_arrival_rate seed $seed
        CUDA_VISIBLE_DEVICES=$cuda_device \
        python main.py \
            --p_net_topology=$topology \
            --decode_strategy="greedy" \
            --k_searching=1 \
            --shortest_method="k_shortest_length" \
            --k_shortest=$k_shortest \
            --solver_name=$solver_name \
            --num_train_epochs=$num_train_epochs \
            --eval_interval=10 \
            --save_interval=10 \
            --num_workers=$num_workers \
            --pretrained_model_path=$pretrained_model_path \
            --batch_size=$batch_size \
            --v_sim_setting_aver_arrival_rate=$aver_arrival_rate \
            --verbose=1 \
            --v_sim_setting_v_net_size_low=$v_sim_setting_v_net_size_low \
            --v_sim_setting_v_net_size_high=$v_sim_setting_v_net_size_high \
            --v_sim_setting_node_resource_attrs_low=$v_sim_setting_node_resource_attrs_low \
            --v_sim_setting_node_resource_attrs_high=$v_sim_setting_node_resource_attrs_high \
            --lr_actor=0.001 \
            --lr_critic=0.001 \
            --summary_dir="exp_data/conal/dynamic_wx100" \
            --seed=$seed \
            --if_dynamic_v_nets=$if_dynamic_v_nets \
            --summary_file_name="dynamic_$topology-$solver_name$identifier.csv"
    done
done