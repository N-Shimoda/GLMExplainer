#!/usr/bin/env bash
#
# Claude Code PostToolUse hook: report ruff diagnostics for edited Python files.
#
# Reports only -- it never rewrites the file. Findings are surfaced twice:
#   * systemMessage      -> shown to the user in the CLI
#   * additionalContext  -> injected into Claude's context (non-blocking)
#
# To make Claude auto-apply ruff's safe fixes instead, uncomment the `--fix`
# line below. To make unfixed findings block Claude (it must resolve them
# before continuing), replace the final `jq -n ...` with `echo "$out" >&2;
# exit 2`. Both are off on purpose: reporting keeps the fix under review
# rather than rewriting code behind the user's back.

set -uo pipefail

RUFF_VERSION="0.16.1"

input=$(cat)
file=$(printf '%s' "$input" | jq -r '.tool_input.file_path // .tool_response.filePath // empty')

[[ "$file" == *.py && -f "$file" ]] || exit 0

# uvx "ruff@${RUFF_VERSION}" check --fix -q "$file" >/dev/null 2>&1

out=$(uvx "ruff@${RUFF_VERSION}" check --output-format=concise "$file" 2>&1) && exit 0

jq -n --arg file "$file" --arg out "$out" '
{
  systemMessage: ("ruff: " + $file + "\n" + $out),
  hookSpecificOutput: {
    hookEventName: "PostToolUse",
    additionalContext: (
      "ruff の指摘です (報告のみ / 自動修正は無効):\n" + $out +
      "\nリポジトリは ruff クリーンな状態を維持しています。" +
      "上記は今回の編集で入った指摘なので、修正してください。"
    )
  }
}'
