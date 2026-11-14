set shell := ["bash", "-uc"]

# 本仓库在 Linux 上跑，解释器用仓库内的 .venv（依赖清单见 requirements.txt）。
py := "./.venv/bin/python"

# 列出所有命令
default:
    @just --list

# 打印解释器、关键包版本、加速器状态
env:
    {{py}} tools/env.py

# 跑一个脚本：just run work/a0/hello_tensor.py
run f:
    {{py}} {{f}}

# 跑某单元的验收断言：just check b02
# 也可以指定别的实现文件（自测用）：just check b02 solutions/b02/b02.py
check u *args:
    {{py}} tools/check_{{u}}.py {{args}}

# 在 ofalias 的镜像仓库上跑断言（远端是 NixOS，用 nix 提供 libstdc++）
check-remote u *args:
    ssh ofalias 'cd ~/d2l-learning~ && lib=$(nix build nixpkgs#stdenv.cc.cc.lib --no-link --print-out-paths | tail -1) && LD_LIBRARY_PATH=$lib/lib just check {{u}} {{args}}'

# 重建 .venv 并装依赖（换机器或 .venv 坏掉时用）
setup:
    @[ -d .venv ] || python -m venv .venv
    {{py}} -m pip install -r requirements.txt

# 删掉 .venv 与 __pycache__
clean:
    rm -rf .venv
    find . -name __pycache__ -type d -prune -exec rm -rf {} +
