torchrun --nproc_per_node 2 train_multitask.py \
--base-model "Qwen/Qwen3-4B-Base" \
--gnn-type "GCN" \
--num-graph-tokens 4 --node-feat-dim 8 --pos-emb-dim 8 \
--gnn-hidden-dim 256 --gnn-out-dim 512 --num-gnn-layers 4 \
--epochs 24 --lr 0.01 \
--save-intermediate-models --save-epoch-interval 4 \
--wandb

python eval_multitask.py --model-path outputs/multitask --num-trials 10
