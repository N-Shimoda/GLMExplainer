from safetensors.torch import load_file

# safetensors ファイルのパス
path = "outputs/edge_count/0922-1827/checkpoint-189/model-00001-of-00004.safetensors"

# safetensors ファイルをロード（辞書型で返される）
weights = load_file(path)

# 中身を確認（キー: パラメータ名、値: Tensor）
for name, tensor in weights.items():
    print(name, tensor.shape)
