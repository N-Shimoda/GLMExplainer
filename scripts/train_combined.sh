torchrun --nproc_per_node 2 train_combined.py \
--base_model "Qwen/Qwen3-4B-Base" \
--num_graph_tokens 4 --node_feat_dim 8 --node_pos_dim 8 \
--gnn_hidden_dim 256 --gnn_out_dim 512 --num_gnn_layers 4 \
--epochs 3 --lr 0.01 \
--do_eval --wandb