from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F

# Load model (use small one for your system)
model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Input text
text = "The capital of France is"

inputs = tokenizer(text, return_tensors="pt")

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits

# Take last token logits
last_token_logits = logits[0, -1, :]

# Convert to log probabilities
log_probs = F.log_softmax(last_token_logits, dim=-1)

# Get top 5 tokens
top_k = 5
values, indices = torch.topk(log_probs, top_k)

print("Top predictions:\n")

for i in range(top_k):
    token = tokenizer.decode([indices[i]])
    print(f"{token} → log_prob = {values[i].item():.4f}")