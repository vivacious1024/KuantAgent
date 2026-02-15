# KuantAgent 创新开发指南 (Innovation Guide)

本文档基于对原版 QuantAgent 代码的深入分析，为您在 `KuantAgent` 分支进行的创新工作提供具体的实施指引。请参考以下三个核心方向进行修改。

## 1. 突破数据视野限制 (Data Horizon Expansion)

**现状 (As-Is):**
原版 `web_interface.py` 中的 `run_analysis` 函数硬编码截取最后 **45 根 K 线**。
```python
# web_interface.py
df_slice = df.tail(45)  # 仅取最后 45 根 K 线
```
这导致 AI 无法感知更宏观的市场结构（如大周期的支撑阻力位），容易陷入局部噪声。

**修改建议 (To-Be):**
1.  **动态数据窗口**：修改 `run_analysis`，不仅传入当前周期的 `df_slice`，还应传入更大周期的数据。
2.  **双周期分析 (Dual-Timeframe Analysis)**：
    -   传入 `macro_df` (如日线级别，取最近 100 根) 和 `micro_df` (如 15 分钟级别，取最近 50 根)。
    -   在 `agent_state.py` 中增加 `macro_trend_report` 字段。
    -   让 `Trend Agent` 先分析宏观趋势，再分析微观趋势。

## 2. AI 驱动的趋势认知 (AI-Driven Trend Analysis)

**现状 (As-Is):**
`graph_util.py` 中的 `generate_trend_image` 完全依赖传统的**最小二乘法 (Least Squares)** 数学拟合来画线。AI 只是对着画好的线“看图说话”。
```python
# graph_util.py
support_coefs, resist_coefs = fit_trendlines_high_low(...)
```
这限制了 AI 的主观能动性，无法识别非线性的形态（如圆弧底、楔形）。

**修改建议 (To-Be):**
1.  **保留 Baseline**：保留现有的数学拟合用于生成 `trend_image`。
2.  **新增 AI 标注**：
    -   新增 `generate_raw_kline_image` 工具（仅画 K 线，不画趋势线）。
    -   修改 `Trends Agent` 的 Prompt，让多模态大模型（如 Qwen-VL, GPT-4o）直接指出图中的关键高低点（Pivot Points）。
    -   让 AI 判断数学拟合的线条是否合理（Critique Mechanism）。

## 3. 引入自我反思机制 (Self-Reflection / Critic Agent)

**现状 (As-Is):**
决策逻辑是线性的：`Indicator` -> `Pattern` -> `Trend` -> `Decision`。
如果前置智能体出错，`Decision Agent` 往往会照单全收。

**修改建议 (To-Be):**
1.  **增加 Critic Agent**：
    -   在 `Decision Agent` 做出决策后，不仅输出结果，还输出 `confidence_score`。
    -   如果置信度低，引入一个 `Critic Agent`（批评家），它会审查所有中间报告，寻找逻辑漏洞（例如：指标显示超买，但趋势显示强劲上涨，是否存在背离？）。
2.  **利用 `custom_qwen.py` 的优势**：
    -   由于您已经实现了 `custom_qwen.py` 的 Prompt 注入技术，可以轻松接入推理能力更强的模型（如 **DeepSeek-R1** 或 **Qwen-Max**）作为 `Critic Agent` 的大脑。
    -   这些模型通常推理能力强，适合做“事后诸葛亮”的审查工作。

## 4. 架构微调建议

-   **文件结构**：建议在 `KuantAgent` 中新建 `agents/` 目录，将 `indicator_agent.py`, `trend_agent.py` 等移动进去，保持根目录整洁。
-   **配置管理**：在 `default_config.py` 中增加 `ENABLE_CRITIC` 和 `USE_DUAL_TIMEFRAME` 开关，方便在不同实验组之间切换。

---
*Created by Antigravity for KuantAgent Innovation Project*
