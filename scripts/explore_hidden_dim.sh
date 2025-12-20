for subset in cycle_tree ba_two_motifs; do
	for dim in 16 32 64 128 256 512; do
		torchrun --nproc_per_node=2 train.py \
			--dataset "MotifQA" --subset "${subset}" \
			--num-graph-tokens 4 --node-feat-dim 8 --pos-emb-dim 8 \
			--gnn-hidden-dim "${dim}" --gnn-out-dim "${dim}" --num-gnn-layers 3 \
			--epochs 24 \
			--gnn-type "GCN" \
			--optim "adamw" --lr 0.0075 --weight-decay 0.01 \
			--lr-scheduler-type "cosine" --warmup-ratio 0.05 \
			--do-eval \
			--wandb
	done
done
