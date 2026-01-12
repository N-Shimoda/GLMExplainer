for thresh in 0.0 1.5 3.0 4.5; do
	torchrun --nproc_per_node=2 explain.py \
		--dataset MotifQA --subset ba_shapes \
		--model-path masters/ba_shapes \
		--target-pos-samples \
		--num-trials 4 --num-samples 2 \
		--llr-threshold $thresh --baseline-graph "complete" \
		--num-gen-trials 10 \
		--epochs 200 --lr 0.01 \
		--edge-size 96 --edge-ent 1.0
done
