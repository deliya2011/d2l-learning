---
date: 2026-10-16
unit: ALL
kind: 准备
status: 完成
duration: 40
outcome: 未考核
---

# 折叠块改用 Obsidian 原生 callout

## 目标

仓库里所有讲义与题目的"折叠答案/骨架"原先手写 `<details><summary>` HTML。学习者在 Obsidian
里打开时这些块完全不渲染、内容直接摊开。要求统一改成 Obsidian 能正常显示且默认折叠的写法。

## 讲了什么

### 原因诊断

装的是 `details-markdown` 插件，它的 manifest 写的是
"Render Markdown (headings, lists, tables, links, embeds) inside native HTML details blocks
in Reading view"——它是**在 HTML details 里渲染 Markdown 的渲染器**，而 Obsidian 本身不渲染
`<details>`。两边互相指望，结果两头都不成立。

### 采用的写法

改用 Obsidian **原生 callout** 的折叠形态，不依赖任何插件：

```markdown
> [!note]- 参考答点
> 内容每一行都带 `> `
>
> 空行写成单独一个 `>`（不带 `>` 的空行会截断 callout）
```

`[!note]` 后面的 `-` 表示默认折叠（`+` 是默认展开）。

## 学习者的产出

无。改动由私教完成。

## 考核结果

未考核。做了机器可核的检查：

- 转换 86 处，覆盖 12 个 `.md`；再跑一次脚本为 0 处（幂等）
- 全文没有遗留 `</details>`，没有"只有空白的空行"截断 callout
- 把改后内容按规则还原成 `<details>` 形态与原文件逐行比对，非空内容全部等价
- `AGENTS.md` 与 `log/2026-10-14-ALL-撤销作息限制与A1材料.md` 里的 `<details>` 是行文举例
  （在代码围栏或行内代码里），脚本按围栏状态跳过，未被误改
- 渲染效果本机无法验证（没有 Obsidian / markdown 渲染器），已请学习者目视确认

## 用时对比

| 任务 | 预算 | 实际 | 比值 | 备注 |
|---|---|---|---|---|
| 转换 + 校验 + 提交 | 未给 | 约 40 分钟 | — | 临时要求，没给预算 |

## 暴露的问题

| 问题 | 类型 | 进错题本 |
|---|---|---|
| `exercises/b02/oral.md` 的 Q7 少一个 `</details>` 收尾标签（9 开 8 闭），会让 Q8 的答案被并进 Q7 的折叠块 | 材料缺陷 | 否，已修 |
| 仓库里的折叠语法与 Obsidian 渲染不一致，此前一直没被发现 | 材料缺陷 | 否，已定规范 |

## 下一步

无。规范已写进 `AGENTS.md`（"把答案折进折叠块"一条改成 callout 写法），
批量转换脚本留在 `tools/fix_details.py`，可重跑。
