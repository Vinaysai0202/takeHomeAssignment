# Project Setup and Instructions

## Project Structure
```
.
├── analysis.py                  # Part A: Sales and support data analysis
├── agent/
│   └── agent.py                 # Part B: Retrieval-Augmented QA Agent
├── sales_2024Q3.csv             # Sales data
├── support_tickets_2024Q3.csv  # Support tickets data
├── docs/                        # Documentation for RAG agent
│   ├── onboarding_guide.md
│   ├── pricing.md
│   ├── product_specs_widget.md
│   ├── refund_policy.md
│   ├── release_notes.md
│   └── troubleshooting.md
├── analysis_plots.png           # Generated visualizations
├── INSIGHTS.md                  # Generated insights from analysis
└── README.md                    # Project instructions and setup
```

## Prerequisites
- Python 3.10+
- pip package manager

## Setup

1. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate    # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install pandas numpy matplotlib seaborn sentence-transformers faiss-cpu openai
   ```

## Part A: Data Analysis

Run the sales and support ticket analysis:

```bash
python analysis.py
```

This will:
- Load and analyze `sales_2024Q3.csv` and `support_tickets_2024Q3.csv`
- Display key metrics and summaries in the console
- Generate visualizations in `analysis_plots.png`
- Save business insights in `INSIGHTS.md`

## Part B: Retrieval-Augmented QA Agent

1. **Ensure documents are present in the `docs/` folder**.

2. **Run the RAG agent**:
   ```bash
   python agent/agent.py
   ```

   This will:
   - Load all documents from `docs/`
   - Chunk documents and create embeddings using `sentence-transformers/all-MiniLM-L6-v2`
   - Build an in-memory vector index
   - Start an interactive question-answering session

3. **Using interactive mode**:
   - Type your question and press Enter
   - The agent retrieves relevant chunks and provides answers
   - If `OPENAI_API_KEY` is set, answers are generated with OpenAI's GPT-3.5-Turbo
   - Type `exit` or `quit` to close the agent

## Expected Outputs

### Part A:
- Printed sales and support analysis
- `analysis_plots.png` with charts
- `INSIGHTS.md` containing business insights

### Part B:
- Interactive Q&A session in terminal
- Citations of source files for each answer

## Troubleshooting

- **Missing dependencies**:
  ```bash
  pip install -r requirements.txt
  ```

- **File not found**: Ensure all CSV files and documents exist in their correct folders.
- **Memory issues**: Chunk size and embedding model are designed for efficiency; increase chunk size if needed for performance.

## Notes

- The RAG agent uses a simple cosine similarity search approach for demonstration.
- Ensure all documents in `docs/` are text-based `.md` files for best results.
- The analysis script follows PEP8 standards for clarity and maintainability.
