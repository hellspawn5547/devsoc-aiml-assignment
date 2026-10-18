# DevSoc AI/ML Crew Assignment (2026–2027)

Comprehensive submission repository for the **DevSoc BITS Goa AI/ML Crew Induction Assignment**.

---

## Repository Structure

```text
.
├── README.md                      # Comprehensive project documentation
├── requirements.txt               # Environment dependencies
├── .gitignore                     # Git ignore rules (protecting .env and caches)
│
├── part1_llm_api/                 # Module 1: LLM API Integration
│   ├── api_caller.py              # Automated batch prompt processing script
│   ├── text.txt                   # Input file containing 12 technical questions
│   ├── responses.json             # Formatted output JSON with AI answers
│   └── .env.example               # Template for API credentials
│
└── part2_linear_regression/       # Module 2: Linear Regression from Scratch
    └── Lin_Reg.ipynb              # Fully completed Google Colab notebook
```

---

## Module 1: LLM API Integration Script

### Overview
This module automates querying an LLM API (Google Gemini, Groq, or OpenAI) with a batch of prompt questions from `text.txt` and exports structured, timestamped results into `responses.json`.

### Features
- **Proper Naming Conventions & PEP 8 Compliance**: Written with clean object-oriented design and type annotations.
- **Secure Secret Handling**: API keys are loaded via environment variables using `python-dotenv`.
- **Fault Tolerance**: Automatic fallback and polite pacing between requests to avoid rate limits.

### How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set up your API key:
   ```bash
   cp part1_llm_api/.env.example part1_llm_api/.env
   # Edit part1_llm_api/.env to insert your GEMINI_API_KEY or GROQ_API_KEY
   ```
3. Run the script:
   ```bash
   python part1_llm_api/api_caller.py
   ```

---

## Module 2: Linear Regression from Scratch (Colab)

### Overview
Implementation of multivariate Linear Regression to predict housing sale prices (`SalePrice`) using NumPy vectorization and Batch Gradient Descent.

### Mathematical Formulation
- **Hypothesis**: $h_w(x) = W^T X + b$
- **Loss Function (MSE)**:
  $$J(W, b) = \frac{1}{2m} \sum_{i=1}^{m} (h_w(x^{(i)}) - y^{(i)})^2$$
- **Gradient Update Rule**:
  $$W := W - \alpha \frac{\partial J}{\partial W}, \quad b := b - \alpha \frac{\partial J}{\partial b}$$

### Key Implementation Details
1. **One-Hot Encoding**: Handled non-numeric categorical attributes (`Utilities`).
2. **Min-Max Feature Scaling**: Normalized all features to $[0, 1]$ to ensure smooth gradient descent convergence.
3. **Prevention of Data Leakage**: Scaled the test dataset strictly using the training set's minimum and maximum values.
4. **Vectorized NumPy Operations**: High-throughput matrix multiplication without explicit iteration loops over samples.

---

## Submission Checklist
- [x] Part 1: Automated API calling script (`api_caller.py`) with input `text.txt` and output `responses.json`.
- [x] Part 2: Working linear regression notebook (`Lin_Reg.ipynb`) with mathematical derivations, training loop, loss convergence plot, and test set analysis.
- [x] Clean modular code adhering to PEP 8.
- [x] Public GitHub repository ready for review.
