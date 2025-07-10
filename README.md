# Math Helper Prototype

This project is a prototype Math Helper tool that classifies math questions into categories and generates step-by-step solution.

---

## Project Overview

### 1. Math Category Classification
- Model: **Hugging Face BERT (bert-base-uncased)**
- Fine-tuned on: [Kaggle AIMO External Dataset](https://www.kaggle.com/datasets)  
  File used: `/kaggle/input/aimo-external-dataset/external_df.csv`
- Task: Classify user-submitted math questions into categories (e.g., Geometry, Algebra, etc.)
- Confidence score: Displayed alongside predicted category
- Fine-tuned model link: `https://drive.google.com/drive/folders/1YNDlhmk-Sr2RavT51QfmA40LC_COZpck?usp=drive_link`

### 2. Step-by-Step Solution Generation
- Model: **LLaMA 3.3 70B - Versatile**, accessed via **GroqCloud**
- Task: Generate plain-text, step-by-step solutions based on user input
- Prompting: Instructed to avoid LaTeX and format solutions clearly
- API key required via environment variable: `GROQ_API_KEY`

---

##  Data Preprocessing

The file `data.py` contains preprocessing steps applied to the raw dataset before fine-tuning the BERT model.

## How It Works

1. User enters a math question in the Streamlit interface.
2. Fine-tuned BERT model predicts the question's category and confidence.
3. The question is then passed to the LLaMA model for a step-by-step solution.
4. Both the predicted category and the generated solution are shown in the app.

---


