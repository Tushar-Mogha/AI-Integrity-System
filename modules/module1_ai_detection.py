# AI Text Detection - Module 1
# Model: Fine-tuned RoBERTa on DAIGT-V2 dataset
# Training Accuracy: 98.95% (achieved on Google Colab with T4 GPU)
# Dataset: DAIGT-V2 - 10,000 essays (5000 Human + 5000 AI)
# Model saved at: https://huggingface.co/Tushar101/module1-roberta
# Status: Deployment pending due to version compatibility issues
#         Module 2 is currently handling AI detection as fallback

# ── Future Implementation ─────────────────────────────────────────────────────
# Once version compatibility is resolved, load model like this:
#
# from transformers import RobertaTokenizer, RobertaForSequenceClassification
# import torch
#
# tokenizer = RobertaTokenizer.from_pretrained("Tushar101/module1-roberta")
# model = RobertaForSequenceClassification.from_pretrained("Tushar101/module1-roberta")
# model.eval()
#
# def predict_ai_or_human(essay_text):
#     inputs = tokenizer(
#         essay_text,
#         max_length=256,
#         padding='max_length',
#         truncation=True,
#         return_tensors='pt'
#     )
#     with torch.no_grad():
#         outputs = model(**inputs)
#         probs   = torch.softmax(outputs.logits, dim=1)
#         pred    = torch.argmax(probs, dim=1).item()
#     label = "AI" if pred == 1 else "Human"
#     return {
#         'prediction'       : label,
#         'ai_probability'   : round(probs[0][1].item() * 100, 2),
#         'human_probability': round(probs[0][0].item() * 100, 2)
#     }

# print("Module 1 - RoBERTa AI Text Detection")
# print("Training Accuracy: 98.95%")
# print("Model: huggingface.co/Tushar101/module1-roberta")
# print("Status: Deployment in progress")
# print("Fallback: Module 2 handling AI detection at 98.17% accuracy")

# Module 1 - AI Text Detection
# Model: Fine-tuned RoBERTa on DAIGT-V2 dataset (44,868 essays)
# Training Accuracy: 99.60% achieved on Google Colab
# Model saved at: https://huggingface.co/Tushar101/module1-roberta
# Team - Abhinandan, Stuti, Tushar

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

os.system('cls')

print("Loading Module 1 - RoBERTa model...")
model_path = "Tushar101/module1-roberta"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model     = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()
print("Model loaded!")

def predict_ai_or_human(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)

    probs      = torch.nn.functional.softmax(outputs.logits, dim=1)
    ai_prob    = probs[0][1].item()
    human_prob = probs[0][0].item()
    label      = "AI" if ai_prob > 0.5 else "Human"

    return {
        "prediction"       : label,
        "ai_probability"   : round(ai_prob * 100, 2),
        "human_probability": round(human_prob * 100, 2)
    }

if __name__ == "__main__":
    human = "I am a good boy , my name is Tushar."
    ai    = "The implementation of comprehensive educational frameworks necessitates systematic evaluation of pedagogical methodologies to optimize student learning outcomes."

    print("\n── Testing Module 1 ────────────────────────────────")
    print("\nHuman Text:")
    print(predict_ai_or_human(human))
    print("\nAI Text:")
    print(predict_ai_or_human(ai))