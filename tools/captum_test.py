import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.explain import CaptumExplainer, Explainer
from torch_geometric.nn import GCNConv

# 例: Coraでノード分類
dataset = Planetoid(root="/tmp/Cora", name="Cora")
data = dataset[0]


class GCN(torch.nn.Module):
    def __init__(self, in_dim, hid, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid)
        self.conv2 = GCNConv(hid, out_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=-1)  # log_probs を返す


model = GCN(dataset.num_features, 16, dataset.num_classes)
model.eval()

# Captum の手法を選ぶ（例: IntegratedGradients）
explainer = Explainer(
    model=model,
    algorithm=CaptumExplainer("IntegratedGradients"),
    explanation_type="model",  # モデルの個別予測を説明
    node_mask_type="attributes",  # 特徴量重要度（feature attribution）
    edge_mask_type="object",  # 構造（エッジ）重要度
    model_config=dict(
        mode="multiclass_classification",
        task_level="node",
        return_type="log_probs",  # モデルの出力の型に合わせる
    ),
    # 重要度をTop-Kだけに絞りたい場合は閾値設定も可:
    # threshold_config=dict(threshold_type="topk", value=20),
)

node_idx = 10
with torch.no_grad():
    target = data.y[node_idx].unsqueeze(0)  # 説明したいクラス（通常はそのノードの正解/予測クラス）

# 説明を生成
explanation = explainer(
    data.x,
    data.edge_index,
    index=node_idx,  # このノードの予測を説明
    target=target,  # 対象クラス（Tensorで渡す）
)

# 重要度ベクトル（featureごと・edgeごと）
print(explanation.node_mask)  # 形状: [num_nodes, num_features]（attributes指定時）
print(explanation.edge_mask)  # 形状: [num_edges]

# 可視化
# explanation.visualize_feature_importance("feat.pdf", top_k=10)
# explanation.visualize_graph("graph.pdf")
