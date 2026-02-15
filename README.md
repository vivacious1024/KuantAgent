# KuantAgent

**KuantAgent** 是一个基于价格驱动的多智能体（Price-Driven Multi-Agent）量化交易系统，衍生自 [QuantAgent 项目](https://github.com/Y-Research-SBU/QuantAgent.git)。本项目旨在通过引入前沿的 AI 技术和改进的架构设计，突破现有量化分析的局限。

## 🚀 项目愿景

KuantAgent 致力于解决传统量化系统在“数据视野”和“趋势认知”上的痛点，通过以下创新方向提升交易决策的准确性与鲁棒性：

1. **突破数据视野限制 (Horizon Expansion)**：引入双周期（宏观+微观）分析机制，打破单一时间窗口的局限。
2. **AI 驱动的趋势认知 (AI-Driven Trend Analysis)**：结合传统算法与 VL（视觉语言）模型，实现更智能的形态识别与趋势研判。
3. **自我反思机制 (Self-Reflection)**：引入 Critic Agent，对决策进行多维度审查与反思。

## 🛠️ 技术栈

* **核心框架**: Python 3.11+, LangChain, LangGraph
* **大模型支持**: OpenAI (GPT-4o), Anthropic (Claude), Alibaba (Qwen/DashScope)
* **Web 界面**: Flask, HTML/CSS/JS
* **数据源**: Yahoo Finance (yfinance)
* **图表工具**: mplfinance, matplotlib

## 📦 快速开始

1. **克隆仓库**

   ```bash
   git clone https://github.com/vivacious1024/KuantAgent.git
   cd KuantAgent
   ```
2. **安装依赖**

   ```bash
   pip install -r requirements.txt
   ```
3. **配置 API Key**
   请在环境变量中设置您的 LLM API Key，或在 Web 界面中直接输入。
4. **启动 Web 界面**

   ```bash
   python web_interface.py
   ```

   访问 `http://127.0.0.1:5000` 即可使用。

## 📝 创新指南

详细的开发与修改计划，请参考项目根目录下的 [INNOVATION_GUIDE.md](INNOVATION_GUIDE.md)。

---

*Disclaimer: This software is for educational and research purposes only. Do not use it for financial advice.*
