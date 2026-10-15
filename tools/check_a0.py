"""A0 验收断言。

用法：
    just check a0                      # 查 work/a0/ 下的两个脚本
    just check a0 solutions/a0         # 查别的目录（自测用）
    .venv/bin/python tools/check_a0.py

查的是两件事：`hello_tensor.py` 打印出来的东西对不对，`sin.py` 有没有生成一张
真的画了曲线的 `sin.png`。

讲义末尾那 9 道自测题是口试，这里查不了，由人工抽查。
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import torch

# 失败信息里可能带学习者的原始输出，控制台编码装不下的字符直接替换，别让报错本身炸掉
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

REPO = Path(__file__).resolve().parent.parent
# 默认查学习者的 work/a0，传一个目录参数可以查别的实现（自测用）
WORK = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "work" / "a0"
if not WORK.is_absolute():
    WORK = REPO / WORK
HELLO = WORK / "hello_tensor.py"
SIN = WORK / "sin.py"
PNG = WORK / "sin.png"

TIMEOUT = 120  # 秒。脚本卡住通常是 plt.show() 打开了窗口在等人关

_failures: list[str] = []
_passed: list[str] = []


def case(name: str):
    def deco(fn):
        try:
            fn()
        except AssertionError as exc:
            _failures.append(f"{name}: {exc}")
        except Exception as exc:  # noqa: BLE001  学习者的代码可能抛任何东西
            _failures.append(f"{name}: 意外异常 {type(exc).__name__}: {exc}")
        else:
            _passed.append(name)
        return fn

    return deco


def run_script(path: Path) -> subprocess.CompletedProcess:
    """用当前解释器跑一个脚本，返回 CompletedProcess（文本模式）。

    强制子进程按 UTF-8 输出：默认它会跟着控制台的代码页走（中文 Windows 是 GBK），
    那样中文在管道里会解不回来。
    """
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    return subprocess.run(
        [sys.executable, str(path)],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=TIMEOUT,
        env=env,
    )


def numbers_in(text: str) -> set[float]:
    """把输出里出现的所有数字抓出来，用来验证打印的内容。"""
    return {float(m) for m in re.findall(r"\d+(?:\.\d+)?", text)}


def main() -> int:
    # ---------------- 任务一：hello_tensor.py ----------------
    hello_out = ""
    cuda_available: bool | None = None

    @case("T1 hello_tensor.py 存在")
    def _t1():
        assert HELLO.exists(), f"找不到 {HELLO}，先照着 notes/a0-setup.md 第 5 节写出来"

    if HELLO.exists():

        @case("T2 hello_tensor.py 能跑完")
        def _t2():
            nonlocal hello_out
            try:
                proc = run_script(HELLO)
            except subprocess.TimeoutExpired:
                raise AssertionError(
                    f"跑了 {TIMEOUT} 秒还没结束。多半是 plt.show() 之类打开了窗口在等人关，"
                    "或者脚本里有 input()"
                ) from None
            hello_out = proc.stdout
            assert proc.returncode == 0, (
                f"退出码 {proc.returncode}，说明脚本报错了。stderr:\n{proc.stderr[-1500:]}"
            )
            assert "Traceback" not in proc.stderr, f"stderr 里有 traceback:\n{proc.stderr[-1500:]}"

        if hello_out:

            @case("T3 打印了 torch 版本号")
            def _t3():
                m = re.search(r"(\d+\.\d+\.\d+)", hello_out)
                assert m, f"输出里找不到形如 2.14.0 的版本号。实际输出：\n{hello_out}"
                assert m.group(1).startswith("2."), (
                    f"版本号看着不对：{m.group(1)}。要打印 torch.__version__"
                )

            @case("T4 打印了 CUDA 是否可用")
            def _t4():
                nonlocal cuda_available
                m = re.search(r"(True|False)", hello_out)
                assert m, f"输出里找不到 True/False，没打印 CUDA 可用性。实际输出：\n{hello_out}"
                cuda_available = m.group(1) == "True"
                # 这台机器没有独立显卡，torch 是 CPU 版，is_available() 恒为 False。
                # 有显卡的机器上才要求打印 True，两种情况都收，不要按机器写死。
                assert cuda_available == torch.cuda.is_available(), (
                    f"脚本打印 CUDA 可用 = {cuda_available}，但 torch.cuda.is_available() 是 "
                    f"{torch.cuda.is_available()}。检查是不是打印了别的东西顶替了它。"
                    f"实际输出：\n{hello_out}"
                )

            @case("T5 打印了形状 (3, 4)")
            def _t5():
                ok = re.search(r"torch\.Size\(\[3,\s*4\]\)", hello_out) or re.search(
                    r"\(\s*3\s*,\s*4\s*\)", hello_out
                )
                assert ok, (
                    "输出里找不到 3 行 4 列的形状，应当是 torch.Size([3, 4]) 或 (3, 4)。"
                    f"实际输出：\n{hello_out}"
                )

            @case("T5b 打印了张量所在设备")
            def _t5b():
                if not cuda_available:
                    # 本机没有显卡。任务一的第 5 步是"如果 CUDA 可用"，条件不成立就不该做那几步，
                    # 能跑到这里说明脚本没有去调 .to("cuda")。人工看它是怎么跳过分支的。
                    return
                # 必须匹配 "cuda:0" 这种形式：只搜 "cuda" 会被版本号 2.14.0+cu126
                # 和 "CUDA可用:True" 那行误命中
                assert re.search(r"cuda:\d", hello_out), (
                    "输出里找不到 cuda:0 这样的设备名。第 5 步要求在张量搬到显卡之后打印它所在的设备；"
                    "注意如果在那之前调过 .cpu() 并接住返回值，设备就变回 cpu 了。"
                    f"实际输出：\n{hello_out}"
                )

            @case("T6 打印了元素个数 12")
            def _t6():
                assert 12.0 in numbers_in(hello_out), (
                    f"输出里没有 12，元素个数没打印。实际输出：\n{hello_out}"
                )

            @case("T7 x 的内容是 0 到 11")
            def _t7():
                nums = numbers_in(hello_out)
                missing = [i for i in range(12) if float(i) not in nums]
                assert not missing, (
                    f"输出里缺这些数：{missing}。0 到 11 都要出现，说明 x 的内容没完整打印出来。"
                    f"实际输出：\n{hello_out}"
                )

            @case("T8 显卡上乘 2 的结果打印了")
            def _t8():
                if not cuda_available:
                    # 同上：没有显卡时第 5 步整体被跳过，没有"显卡上乘 2"的结果可查
                    return
                nums = numbers_in(hello_out)
                missing = [i for i in range(12, 23, 2) if float(i) not in nums]
                assert not missing, (
                    f"输出里缺这些数：{missing}。它们只可能来自乘 2 之后的结果"
                    "（应为 0, 2, ..., 22），说明第 5 步没做全或没打印。"
                    f"实际输出：\n{hello_out}"
                )

    # ---------------- 任务二：sin.py ----------------
    png_mtime_before: float | None = PNG.stat().st_mtime if PNG.exists() else None

    @case("T9 sin.py 存在且能跑完")
    def _t9():
        assert SIN.exists(), f"找不到 {SIN}，先照着 notes/a0-setup.md 第 6 节写出来"
        try:
            proc = run_script(SIN)
        except subprocess.TimeoutExpired:
            raise AssertionError(
                f"跑了 {TIMEOUT} 秒还没结束。多半是没设 Agg 后端，plt.show() 开了窗口在等人关"
            ) from None
        assert proc.returncode == 0, (
            f"退出码 {proc.returncode}，脚本报错了。stderr:\n{proc.stderr[-1500:]}"
        )

    @case("T10 生成了 work/a0/sin.png，且是这次跑出来的")
    def _t10():
        assert PNG.exists(), (
            f"找不到 {PNG}。savefig 的路径要写成 work/a0/sin.png（相对仓库根目录），"
            "写成 out.png 之类会落到别的地方"
        )
        assert PNG.stat().st_size > 5000, (
            f"图片只有 {PNG.stat().st_size} 字节，太小了，多半是被裁空或没画上东西"
        )
        # 目录里可能留着上一次跑的图。只有 mtime 变了才说明这次的 savefig 真的写了这个文件
        if png_mtime_before is not None:
            assert PNG.stat().st_mtime > png_mtime_before, (
                f"{PNG.name} 是上一次留下的，这次跑完没有更新——说明 savefig 写的不是这个路径"
            )

    @case("T11 图里真的有一条曲线")
    def _t11():
        from PIL import Image

        if png_mtime_before is not None:
            assert PNG.stat().st_mtime > png_mtime_before, (
                "这张图是上一次留下的，不是这次跑出来的，内容检查没有意义（见 T10）"
            )
        with Image.open(PNG) as im:
            assert im.format == "PNG", f"文件格式是 {im.format}，要存成 PNG"
            w, h = im.size
            assert w >= 200 and h >= 100, f"图只有 {w}x{h} 像素，太小了"
            gray = im.convert("L")
            px = gray.load()
            # 背景（白）与曲线（深色）都要有；只在某一半有深色说明曲线没铺满横轴
            dark_left = dark_right = light = 0
            for y in range(0, h, max(1, h // 200)):
                for x in range(0, w, max(1, w // 200)):
                    if px[x, y] < 128:
                        if x < w // 2:
                            dark_left += 1
                        else:
                            dark_right += 1
                    elif px[x, y] > 200:
                        light += 1
            assert light > 0, "整张图没有浅色背景，不像是正常的折线图"
            assert dark_left > 5 and dark_right > 5, (
                f"深色像素太少或只在半边（左 {dark_left}、右 {dark_right} 个采样点）。"
                "要画的是 0 到 2π 上的完整 sin 曲线"
            )

    print(f"通过 {len(_passed)} 项")
    if _failures:
        print(f"\n失败 {len(_failures)} 项：")
        for item in _failures:
            print(f"  - {item}")
        print("\n讲义末尾那 9 道自测题是口试，等我来问。")
        return 1

    print("T1-T11 全部通过。")
    print("讲义末尾那 9 道自测题是口试，等我来问。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
