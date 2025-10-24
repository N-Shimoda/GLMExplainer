import argparse
import os
from collections import Counter
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset
from scipy import stats


def load_data(subset: str, split: Literal["train", "validation", "test"]):
    ds = load_dataset("baharef/GraphQA", subset, split=f"zero_shot_{split}")
    digit_ans_li = [int(example.split(".")[0]) for example in ds["answer"]]
    return digit_ans_li


def summarize_distribution(data: list[int]):
    # 1) 前処理：Noneなどを排除
    x = pd.Series([v for v in data if v is not None])
    n = len(x)
    if n == 0:
        raise ValueError("有効なデータがありません。")

    # 2) 位置・散らばり・形
    mean = x.mean()
    median = x.median()
    # 最頻値（複数ある場合は最小のもの）
    mode_vals = x.mode()
    mode = mode_vals.iloc[0] if not mode_vals.empty else None

    var = x.var(ddof=1)
    std = x.std(ddof=1)
    q = x.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    iqr = q.loc[0.75] - q.loc[0.25]
    mad = stats.median_abs_deviation(x, scale=1.0)

    skew = stats.skew(x, bias=False)
    kurt = stats.kurtosis(x, fisher=True, bias=False)  # 0なら正規分布相当

    # 3) 外れ値候補（Tukey）
    lower_fence = q.loc[0.25] - 1.5 * iqr
    upper_fence = q.loc[0.75] + 1.5 * iqr
    tukey_outliers_idx = x.index[(x < lower_fence) | (x > upper_fence)]
    tukey_outliers = x.loc[tukey_outliers_idx].tolist()

    # 修正Zスコア（MADベース）
    if mad == 0:
        modz = pd.Series(np.zeros(n), index=x.index)
    else:
        modz = 0.6745 * (x - median) / mad
    modz_outliers_idx = x.index[modz.abs() > 3.5]
    modz_outliers = x.loc[modz_outliers_idx].tolist()

    # 4) 正規性検定（参考）
    # Shapiroはn<=5000程度で有効 / D’Agostinoはやや広範囲
    sw_stat, sw_p = stats.shapiro(x) if n <= 5000 else (np.nan, np.nan)
    dag_stat, dag_p = stats.normaltest(x) if n >= 8 else (np.nan, np.nan)

    # 5) 値ごとの頻度（離散性確認）
    counts = Counter(x.tolist())
    pmf = pd.DataFrame({"value": list(counts.keys()), "count": list(counts.values())}).sort_values("value")
    pmf["prob"] = pmf["count"] / n

    summary = {
        "n": n,
        "mean": mean,
        "median": median,
        "mode": mode,
        "var": var,
        "std": std,
        "iqr": iqr,
        "mad": mad,
        "skew": skew,
        "kurtosis_fisher": kurt,
        "p05": q.loc[0.05],
        "p25": q.loc[0.25],
        "p50": q.loc[0.5],
        "p75": q.loc[0.75],
        "p95": q.loc[0.95],
        "range": (x.min(), x.max()),
        "tukey_outliers_count": len(tukey_outliers),
        "tukey_outliers_sample": tukey_outliers[:10],
        "modz_outliers_count": len(modz_outliers),
        "modz_outliers_sample": modz_outliers[:10],
        "shapiro": {"stat": sw_stat, "p": sw_p},
        "dagostino": {"stat": dag_stat, "p": dag_p},
    }
    return x, pmf, summary


def freedman_diaconis_bins(x: pd.Series):
    # ヒストグラムのビン幅をFD法で決定
    q25, q75 = np.percentile(x, [25, 75])
    iqr = q75 - q25
    n = x.size
    if iqr == 0:
        return max(1, int(np.sqrt(n)))
    h = 2 * iqr / (n ** (1 / 3))
    bins = int(np.ceil((x.max() - x.min()) / h)) if h > 0 else max(1, int(np.sqrt(n)))
    return max(1, bins)


def plot_all(x: pd.Series, pmf: pd.DataFrame, prefix="dist"):
    # 1) ヒストグラム
    plt.figure()
    bins = freedman_diaconis_bins(x)
    plt.hist(x, bins=bins)
    plt.title("Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{prefix}_hist.png")
    plt.close()

    # 2) ECDF
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    plt.figure()
    plt.step(xs, ys, where="post")
    plt.title("ECDF")
    plt.xlabel("Value")
    plt.ylabel("Cumulative Probability")
    plt.tight_layout()
    plt.savefig(f"{prefix}_ecdf.png")
    plt.close()

    # 3) 箱ひげ図
    plt.figure()
    plt.boxplot(x, vert=True, whis=1.5, showfliers=True)
    plt.title("Boxplot")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig(f"{prefix}_box.png")
    plt.close()

    # 4) QQプロット（正規性の目視）
    plt.figure()
    stats.probplot(x, dist="norm", plot=plt)
    plt.title("QQ Plot vs Normal")
    plt.tight_layout()
    plt.savefig(f"{prefix}_qq.png")
    plt.close()

    # 5) PMF（ユニーク値が少ない場合のみ）
    if len(pmf) <= 30:
        plt.figure()
        plt.bar(pmf["value"], pmf["prob"])
        plt.title("PMF (Discrete)")
        plt.xlabel("Value")
        plt.ylabel("Probability")
        plt.tight_layout()
        plt.savefig(f"{prefix}_pmf.png")
        plt.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--subset",
        type=str,
        choices=["node_count", "edge_count", "triangle_counting"],
        default="node_count",
        help="Dataset subset to analyze",
    )
    p.add_argument(
        "--split", type=str, choices=["train", "validation", "test"], default="test", help="Dataset split to analyze"
    )
    args = p.parse_args()

    data = load_data(subset=args.subset, split=args.split)
    x, pmf, summary = summarize_distribution(data)
    print(pd.Series(summary))

    OUT_DIR = f"fig/{args.subset}"
    os.makedirs(OUT_DIR, exist_ok=True)
    plot_all(x, pmf, prefix=f"{OUT_DIR}/{args.split}")
