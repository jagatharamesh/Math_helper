import torch
import torch.nn.functional as F

def get_predicted_class(text,tokenizer,model):
   # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Inference
    with torch.no_grad():
         outputs = model(**inputs)
         logits = outputs.logits

    # Convert logits to probabilities
    probs = F.softmax(logits, dim=1)
    # Get predicted class and confidence
    predicted_class_id = torch.argmax(probs, dim=1).item()
    confidence = probs[0][predicted_class_id].item()
    return(predicted_class_id,confidence)
