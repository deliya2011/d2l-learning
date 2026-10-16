"""把 markdown 里 <details> 折叠块改成 Obsidian 的折叠 callout。

Obsidian 不渲染原生 <details>，折叠内容会直接摊开显示。改写成 callout 的
`-` 形态（默认折叠）：

    > [!note]- 参考答点
    > 折叠起来的内容，每一行都要以 `> ` 开头，
    > 空行写成单独一个 `>`（空行不带 `>` 会把 callout 截断）。

同时把嵌套的列表、表格、代码块、公式原样保留：只是给每一行加上 `> `。

用法：
    .venv/bin/python tools/fix_details.py --dry-run     # 只看会改哪些文件
    .venv/bin/python tools/fix_details.py               # 真正改写
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# <details> 与 <summary> 可以写在同一行，也可以分行、带缩进
OPEN_WITH_SUMMARY = re.compile(
    r"^(?P<indent>[ \t]*)<details[^>]*>\s*<summary>(?P<title>.*?)</summary>\s*$"
)
OPEN_ALONE = re.compile(r"^(?P<indent>[ \t]*)<details[^>]*>\s*$")
SUMMARY_ALONE = re.compile(r"^[ \t]*<summary>(?P<title>.*?)</summary>\s*$")
CLOSE = re.compile(r"^[ \t]*</details>\s*$")
FENCE = re.compile(r"^[ \t]*(```|~~~)")


def to_callout(title: str, body: list[str]) -> list[str]:
    """把一段折叠体转成 callout 行。"""
    out = [f"> [!note]- {title.strip()}"]
    for raw in body:
        line = raw.rstrip("\n")
        out.append(f"> {line}".rstrip() if line.strip() else ">")
    return out


def convert(text: str) -> tuple[str, int]:
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    converted = 0
    in_fence = False

    while i < len(lines):
        line = lines[i]

        # 代码围栏里的一切都不动（AGENTS.md、log/ 里那些是行文里的举例）
        if FENCE.match(line):
            in_fence = not in_fence
            out.append(line)
            i += 1
            continue

        if in_fence:
            out.append(line)
            i += 1
            continue

        m = OPEN_WITH_SUMMARY.match(line)
        title = None
        if m:
            title = m.group("title")
        elif OPEN_ALONE.match(line):
            # 标题可能在下一行的 <summary>
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            sm = SUMMARY_ALONE.match(lines[j]) if j < len(lines) else None
            if sm:
                title = sm.group("title")
                i = j  # 跳过分行写的 <summary>

        if title is None:
            out.append(line)
            i += 1
            continue

        # 收集折叠体，直到配对的 </details>。用深度计数 + 代码围栏跟踪，
        # 这样嵌套折叠不会提前结束（当前仓库没有嵌套，留着防以后手滑）。
        body: list[str] = []
        i += 1
        depth = 1
        depth_fence = False
        while i < len(lines):
            if not depth_fence:
                if OPEN_ALONE.match(lines[i]) or OPEN_WITH_SUMMARY.match(lines[i]):
                    depth += 1
                elif CLOSE.match(lines[i]):
                    depth -= 1
                    if depth == 0:
                        break
            if FENCE.match(lines[i]):
                depth_fence = not depth_fence
            body.append(lines[i])
            i += 1
        if i >= len(lines):
            raise SystemExit(f"有没闭合的 <details>：{title!r}")

        # 去掉首尾空行，紧凑一点
        while body and not body[0].strip():
            body.pop(0)
        while body and not body[-1].strip():
            body.pop()

        out.extend(to_callout(title, body))
        converted += 1
        i += 1  # 跳过 </details>

    result = "\n".join(out)
    if text.endswith("\n"):
        result += "\n"
    return result, converted


def main() -> int:
    dry = "--dry-run" in sys.argv
    files = sorted(
        p for p in REPO.rglob("*.md") if ".git" not in p.parts and ".venv" not in p.parts
    )
    total = 0
    for path in files:
        text = path.read_text(encoding="utf-8")
        if "<details" not in text:
            continue
        new, n = convert(text)
        if n == 0:
            continue
        total += n
        rel = path.relative_to(REPO)
        print(f"{n:3d} 处  {rel}")
        if not dry:
            path.write_text(new, encoding="utf-8")
    print(f"\n共 {total} 处" + ("（dry-run，未写入）" if dry else "，已改写"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
