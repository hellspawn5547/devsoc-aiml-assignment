# DevSoc AI/ML Crew Induction Assignment (2026–2027)

This repository contains the complete solution for the **DevSoc BITS Goa AI/ML Crew Assignment**.

---

## Files in this Repository

| File | Description |
| :--- | :--- |
| **`Lin_Reg.ipynb`** | Completed Google Colab notebook for the Linear Regression task (data preprocessing, Min-Max scaling, loss function, vectorized gradient descent, and test evaluation). |
| **`api_caller.py`** | Python script that reads questions from `text.txt`, calls an LLM API, and outputs structured responses. |
| **`text.txt`** | Input text file containing the 12 technical questions assigned by DevSoc. |
| **`responses.json`** | Generated JSON file containing structured questions, responses, model metadata, and timestamps. |
| **`requirements.txt`** | Python dependencies required to run the scripts. |

---

## Part 1: LLM API Caller

### Overview
The `api_caller.py` script automates sending queries to an LLM API and saves the responses into a formatted `responses.json` file.

### How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set your API key in your terminal (e.g. Gemini, Groq, or OpenAI):
   ```bash
   export GEMINI_API_KEY="your_api_key_here"
   ```
3. Execute the script:
   ```bash
   python api_caller.py
   ```

---

## Part 2: Linear Regression from Scratch

### Overview
A complete implementation of multivariate Linear Regression to predict housing sale prices (`SalePrice`) using NumPy vectorization and Batch Gradient Descent.

### Key Highlights
- **Min-Max Normalization**: Normalized features to $[0, 1]$ to ensure smooth gradient descent convergence.
- **Prevention of Data Leakage**: Reused the training dataset's min/max values for transforming the test dataset.
- **Vectorized Gradient Descent**: Optimized weights and bias iteratively using the Mean Squared Error (MSE) loss function.
- **Evaluation**: Visualized loss convergence over epochs and evaluated predictions on the unseen test set.
