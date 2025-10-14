"""OpenAI GPT-4o-mini を使って簡単な数学/グラフ問題を解かせるサンプルスクリプト。

更新点 (2025-10-01):
    - 取得したモデル出力は標準出力には表示せず、ランダムなファイル名のテキストファイルに保存します。
    - 保存先パスのみ標準出力に表示されます (回答本文は表示されません)。

使い方:
    1. 環境変数 OPENAI_API_KEY に API キーを設定:
             export OPENAI_API_KEY=sk-xxxx
    2. 実行例:
             python tools/openai_api.py --question "2x + 3 = 11 の x を求めて"

    質問を省略するとデフォルトで (12 * (7 - 2)) の計算を行います。

依存インストール (environment.yml へ openai 追加済み):
        conda env update -f environment.yml

注意:
    - API キーは決してリポジトリにコミットしないでください。
    - ネットワークやレート制限エラーに対して指数バックオフ風の再試行を行います。
    - 回答出力は自動的に tools/outputs/answers/ 配下 (無ければ作成) に保存されます。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

from openai import OpenAI  # openai>=1.0.0 の新クライアント


def build_prompt(question: str) -> str:
    return (
        "あなたは丁寧でステップを示す数学チュータです。"
        "以下の問題を解き、途中式を列挙し、最後に '答え: <number or expression>' の形式で明示してください。\n\n"
        f"問題: {question}\n"
    )


def solve_math(question: str, max_retries: int = 3, retry_wait: float = 2.0) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY が環境変数に設定されていません。export OPENAI_API_KEY=... してください。")

    client = OpenAI(api_key=api_key)
    # prompt = build_prompt(question)
    prompt_data = {
        "triangle_counting": (
            "In an undirected graph, (i,j) means that node i and node j are connected with an undirected edge. "
            "G describes a graph among nodes 0, 1, 2, 3, 4, 5, 6, and 7. "
            "The edges in G are: (0, 1) (0, 2) (0, 5) (0, 7) (1, 2) (1, 4) (1, 5) (1, 7) (2, 5) (3, 5) (5, 7). "
            "Q: How many triangles are in this graph?"
        ),
        "cycle_check": (
            "In an undirected graph, (i,j) means that node i and node j are connected with an undirected edge. "
            "G describes a graph among nodes 0, 1, 2, 3, 4, 5, 6, and 7. "
            "The edges in G are: (0, 1) (0, 2) (0, 5) (0, 7) (1, 2) (1, 4) (1, 5) (1, 7) (2, 5) (3, 5) (5, 7). "
            "Q: Is there a cycle in this graph? A:"
        ),
    }
    prompt = prompt_data["triangle_counting"]

    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.responses.create(
                model="gpt-4o-mini",
                input=prompt,
                temperature=0.2,
                max_output_tokens=512 + 256,
            )
            if hasattr(response, "output_text") and response.output_text:  # SDK 便宜メソッド
                return response.output_text.strip()
            # fallback: 手動で content を結合
            parts: list[str] = []
            for item in getattr(response, "output", []) or []:
                for c in getattr(item, "content", []) or []:
                    if getattr(c, "type", None) in {"output_text", "text"}:
                        txt = getattr(c, "text", None)
                        if txt and hasattr(txt, "value"):
                            parts.append(txt.value)
            if parts:
                return "".join(parts).strip()
            return str(response)
        except Exception as e:  # pragma: no cover
            last_error = e
            if attempt == max_retries:
                break
            wait = retry_wait * attempt
            print(
                f"[WARN] 呼び出し失敗 (試行 {attempt}/{max_retries}): {e}. {wait:.1f}s 待機後再試行",
                file=sys.stderr,
            )
            time.sleep(wait)
    raise RuntimeError(f"OpenAI 呼び出し失敗: {last_error}")


def extract_final_answer(text: str) -> Optional[str]:
    """出力テキストから '答え:' 以降を抽出。見つからなければ None。"""
    import re

    m = re.search(r"答え\s*[:：]\s*(.+)$", text.strip(), re.MULTILINE)
    if m:
        return m.group(1).strip()
    return None


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="GPT-4o-mini で簡単な数学問題を解く")
    parser.add_argument(
        "--question",
        "-q",
        type=str,
        default="12 * (7 - 2) を計算せよ",
        help="数学の問題文",
    )
    parser.add_argument(
        "--show-answer-only",
        action="store_true",
        help="抽出した最終答えのみ表示",
    )
    args = parser.parse_args()

    try:
        result = solve_math(args.question)
    except Exception as e:  # pragma: no cover
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    # 保存ディレクトリ (tools/outputs/answers) を作成
    out_dir = Path(__file__).resolve().parent / "outputs" / "answers"
    out_dir.mkdir(parents=True, exist_ok=True)
    # ランダムファイル名生成
    filename = f"answer_{int(time.time())}_{uuid.uuid4().hex[:8]}.txt"
    out_path = out_dir / filename
    # ファイルへ書き込み (UTF-8)
    try:
        out_path.write_text(result, encoding="utf-8")
    except Exception as e:  # 失敗したらエラー表示して終了
        print(f"[ERROR] 回答のファイル保存に失敗しました: {e}", file=sys.stderr)
        sys.exit(1)

    # 標準出力にはパスのみ表示。内容は表示しない。
    print(f"Answer saved to: {out_path}")
    # show-answer-only オプションは後方互換のため残しているが、表示抑制方針により無効化。
    if args.show_answer_only:
        print("(NOTE) --show-answer-only は現在非表示ポリシーにより無効です。")


if __name__ == "__main__":
    main()
