# Setup and Run Instructions

## Project Structure
```
.
├── analysis.py              # Part A: Data analysis script
├── agent/
│   └── rag_agent.py        # Part B: RAG QA agent
├── sales_2024Q3.csv        # Sales data
├── support_tickets_2024Q3.csv  # Support tickets data
├── rag_eval_questions.csv  # Evaluation questions
├── docs/                   # Documentation for RAG
│   ├── onboarding_guide.md
│   ├── pricing.md
│   ├── product_specs_widget.md
│   ├── refund_policy.md
│   ├── release_notes.md
│   └── troubleshooting.md
├── INSIGHTS.md            # Generated insights (created by analysis.py)
├── analysis_plots.png     # Generated visualizations
└── evaluation_results.csv # RAG evaluation results

```

## Prerequisites
- Python 3.10+
- pip package manager

## Setup

1. **Create virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install pandas numpy matplotlib seaborn sentence-transformers faiss-cpu
   ```

## Part A: Data Analysis

Run the sales and support data analysis:

```bash
python analysis.py
```

This will:
- Load and analyze `sales_2024Q3.csv` and `support_tickets_2024Q3.csv`
- Display detailed metrics in the console
- Generate `analysis_plots.png` with 4 key visualizations
- Create `INSIGHTS.md` with business insights (<250 words)

## Part B: RAG QA Agent

1. **Ensure documents are in the `docs/` folder** with the exact filenames listed above.

2. **Run the RAG agent**:
   ```bash
   python agent/rag_agent.py
   ```

   This will:
   - Load all documents from the `docs/` folder
   - Build embeddings and vector index
   - Evaluate against questions in `rag_eval_questions.csv`
   - Save results to `evaluation_results.csv`
   - Start an interactive Q&A session

3. **Using the interactive mode**:
   - After evaluation, you can ask custom questions
   - Type your question and press Enter
   - Type 'quit' to exit

## Expected Outputs

### Part A:
- Console output with detailed metrics
- `analysis_plots.png` - 4 visualization panels
- `INSIGHTS.md` - Business insights summary

### Part B:
- Console output showing evaluation results
- `evaluation_results.csv` - Detailed evaluation metrics
- Interactive Q&A interface

## Troubleshooting

1. **Import errors**: Ensure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. **File not found errors**: Verify all CSV files and docs are in the correct locations.

3. **Memory issues**: The default embedding model is lightweight. For larger documents, consider chunking more aggressively.

## Notes

- The RAG agent uses a simple extractive approach for demonstration. In production, integrate with an LLM API for better answer generation.
- Analysis focuses on key business metrics. Additional analyses can be added as needed.
- All code follows PEP8 standards and includes comprehensive docstrings.