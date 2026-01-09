for baseline_type in empty complete; do
	python case_study.py \
		--subset ba_shapes \
		--model-path masters/ba_shapes/ \
		--num-samples 20 \
		--baseline-graph $baseline_type
done
