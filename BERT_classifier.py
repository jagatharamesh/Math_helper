from transformers import BertTokenizerFast, BertForSequenceClassification
import sys
import torch

print("Using Python interpreter:", sys.executable)

def get_model():
    # Huggingface BERT model fine tuned on Math dataset
    model_path = "Add the downloaded model path here(MATH_HELPER/The downloadable link is given in README.md )"
    try:
        tokenizer = BertTokenizerFast.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path)
        model.eval()
        print("Model and tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
    return tokenizer,model
    
if __name__ == "__main__":
    get_model()

        
# sentences = [
#     "Kevin Kangaroo begins hopping on a number line at 0. He wants to get to 1, but he can hop only $\\frac{1}{3}$ of the distance. Each hop tires him out so that he continues to hop $\\frac{1}{3}$ of the remaining distance. How far has he hopped after five hops? Express your answer as a common fraction.",
#     "Let $\\mathbf{a} = \\begin{pmatrix} -3 \\\\ 10 \\\\ 1 \\end{pmatrix},$ $\\mathbf{b} = \\begin{pmatrix} 5 \\\\ \\pi \\\\ 0 \\end{pmatrix},$ and $\\mathbf{c} = \\begin{pmatrix} -2 \\\\ -2 \\\\ 7 \\end{pmatrix}.$  Compute\\[(\\mathbf{a} - \\mathbf{b}) \\cdot [(\\mathbf{b} - \\mathbf{c}) \\times (\\mathbf{c} - \\mathbf{a})].\\]",
# ]


