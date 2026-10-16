"""把 work/a1/ 下的 per-task 文件汇成一个模块，供 tools/check_a1.py 验收。

学习者按 T1.py…T5.py 分文件写（这是他的写法），而断言脚本需要在这些名字的
命名空间里找类与函数。这个文件只做 import 转发，不含任何实现。

用法：
    .venv/bin/python tools/run_a1_checks.py
"""

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WORK = REPO / "work" / "a1"

SHIM = """\
# 由 tools/run_a1_checks.py 生成：只做 import 转发，不含实现
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from T1 import Counter
from T2 import Vec2
from T3 import batch_indices
from T4 import ScaledShift
from T5 import Timer
"""


def main() -> int:
    missing = [f"T{i}.py" for i in range(1, 6) if not (WORK / f"T{i}.py").exists()]
    if missing:
        print(f"缺文件：{', '.join(missing)}")
        return 2

    shim = WORK / "_shim.py"
    shim.write_text(SHIM, encoding="utf-8")
    try:
        return subprocess.run(
            [sys.executable, str(REPO / "tools" / "check_a1.py"), str(shim)],
            cwd=str(REPO),
        ).returncode
    finally:
        shim.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
