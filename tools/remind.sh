#!/usr/bin/env bash
# 扬声器提醒。在本地跑，用系统的声音设备。
#
# 用法：
#   tools/remind.sh                       一声提示音
#   tools/remind.sh 3                     连响三声，溜号了用这个
#   tools/remind.sh 2 "该看进度了"          响两声，再念出来
#
# 语音用 piper（离线，中文自然）。没装 piper 或没有模型时只响提示音，不报错。
# 装法：
#   pacman -S piper                      # piper 本体
#   模型放 ~/.local/share/piper/zh_CN-huayan-medium.onnx（附 .onnx.json）
set -euo pipefail

# 脚本可能从没有桌面会话的环境里被调用，补上这两个变量，否则 paplay 连不上声音服务。
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=$XDG_RUNTIME_DIR/bus}"

VOL_SOUND="${D2L_VOL_SOUND:-45000}"   # paplay 的音量是 0-65536 的相对值，约 69%
VOL_VOICE="${D2L_VOL_VOICE:-60000}"   # 语音，约 92%

MODEL="$HOME/.local/share/piper/zh_CN-huayan-medium.onnx"
SOUND=/usr/share/sounds/freedesktop/stereo/complete.oga
WAV="${TMPDIR:-/tmp}/d2l-remind.wav"

TIMES=1
if [ $# -gt 0 ] && [[ "$1" =~ ^[0-9]+$ ]]; then
  TIMES="$1"
  shift
fi
MSG="${1:-}"

play_sound() {
  if command -v paplay >/dev/null 2>&1 && [ -f "$SOUND" ]; then
    paplay --volume="$VOL_SOUND" "$SOUND" 2>/dev/null || printf '\a'
  else
    printf '\a'
  fi
}

speak() {
  [ -n "$MSG" ] || return 0
  command -v piper >/dev/null 2>&1 || return 0
  [ -f "$MODEL" ] || return 0
  printf '%s' "$MSG" | piper --model "$MODEL" --output_file "$WAV" 2>/dev/null || return 0
  if command -v paplay >/dev/null 2>&1; then
    paplay --volume="$VOL_VOICE" "$WAV" 2>/dev/null || true
  fi
}

for _ in $(seq "$TIMES"); do
  play_sound
  sleep 1
done
speak
