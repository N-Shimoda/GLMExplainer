for edge_size in 0.005 0.01 0.1 1 6 12 24 48; do
	python explain.py \
		--model-path masters/house_check \
		--explain-pos-samples \
		--edge-size $edge_size --edge-ent 1.0 \
		--epochs 200 --lr 0.01 \
		--wandb
done
