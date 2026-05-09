<img width="337" height="141" alt="image" src="https://github.com/user-attachments/assets/844edb9d-341a-4b9c-a3f2-39fab2aca5ab" />
# AlphaFin Enhancement

## Overview
This project extends the AlphaFin framework for financial forecasting by integrating:
- Baseline & Enhanced Pipeline
- RAG-enhanced data augmentation
- LoRA vs LoRI fine-tuning experiments

Large files (databases, model weights) are excluded and must be downloaded at runtime.



💼 Financial Forecasting with RAG-Enhanced LLMs

📌 *Unlocking the Potential of a Financial Forecasting Benchmark: Data-Enriched Prediction with Retrieval-Augmented Language Models*

This project focused on enhancing the AlphaFin financial forecasting benchmark using Retrieval-Augmented Generation (RAG), financial LLM adaptation, and quantitative backtesting.

Instead of building a new benchmark from scratch, I re-engineered and improved the existing AlphaFin pipeline while preserving its original datasets and evaluation framework for fair comparison and reproducibility.

🔹 Main Contributions

🔹 Designed and implemented a financial RAG framework (RQ1)
• Built a Retrieval-Augmented Generation pipeline integrating:

* TuShare market data
* CNINFO disclosures
* earnings guidance
* audit reports
* financial forecasts

• Implemented:

* stock-code normalization
* time-series aligned retrieval
* disclosure filtering
* prompt injection mechanisms
* anti look-ahead-bias controls

• Redesigned prompt engineering workflows by injecting structured financial evidence directly into LLM reasoning.

📈 Results:
• ARR improved from 13.8% → 25.8%
• Sharpe Ratio improved from 0.67 → 1.20
• Directional Accuracy improved from 52.33% → 55.07%

The experiments showed that RAG improves not only prediction accuracy, but also interpretability and portfolio-level trading robustness.

🔹 Conducted LoRA vs LoRI fine-tuning research (RQ2)
• Replaced the original StockGPT adapter with FinGPT-6B on ChatGLM2-6B.

• Implemented and compared:

* LoRA
* LoRI (LoRA with Reduced Interference)

• Evaluated:

* training efficiency
* GPU memory usage
* convergence behavior
* financial backtesting performance

🔍 Key Finding:
LoRI reduced training overhead while maintaining competitive forecasting performance, showing strong potential for scalable multi-task financial NLP systems.

🔹 Model replacement & benchmark enhancement (RQ2)
• Investigated replacing AlphaFin’s original financial adapter with FinGPT-based adapters.

• FinGPT + adapters modestly outperformed several benchmark indices and improved reasoning stability in financial forecasting tasks.

• Identified future opportunities to replace the full backbone model with stronger LLMs such as Qwen or DeepSeek for larger performance gains.

💡 Key Research Insight
Accuracy ≠ Profitability.
A model with slightly lower prediction accuracy may still generate superior trading performance if: • drawdowns are smaller • risk-adjusted returns are higher • long/short allocation is better optimized
The experiments also showed that different model families excel in different financial tasks:
• LLMs → textual reasoning & disclosure interpretation 
• LSTMs → temporal sequence robustness 
• ML models → technical indicator extraction

🔮 Future Work: 
Building hybrid ensemble systems combining: 
• LLM reasoning 
• LSTM sequence learning 
• ML-based quantitative indicators
may produce more robust financial forecasting systems.

🛠 Tech Stack
Python, PyTorch, Hugging Face Transformers, PEFT, LoRA, LoRI, RAG, FinGPT, ChatGLM2-6B, Financial NLP, Quantitative Backtesting, TuShare, CNINFO
<img width="1432" height="1059" alt="AlphaFin_Structure" src="https://github.com/user-attachments/assets/9512fb57-432f-48e3-a25a-7e7d8b02fedb" />


