for thresh in 0.0 2.0; do
	for edge_size in 1e-4 1e-3 3e-3 1e-2 1e-1; do
		torchrun --nproc_per_node=2 explain.py \
			--dataset MotifQA --subset ba_shapes \
			--model-path masters/ba_shapes \
			--target-pos-samples --num-samples 20 \
			--num-trials 5 \
			--epochs 200 --lr 0.01 \
			--edge-size $edge_size --edge-ent 1.0 \
			--llr-threshold $thresh --baseline-graph "complete" \
			--wandb --tags thresh_search mini
	done
done
