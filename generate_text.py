from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"  # You can replace this with other models like "distilgpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

input_text = "Once upon a time"
# Convert text to token IDs
input_ids = tokenizer.encode(input_text, return_tensors="pt")

print(input_ids)

output = model.generate(
    input_ids,
    max_length=50,  # Maximum length of the generated text
    num_return_sequences=1,  # Number of sequences to generate
    no_repeat_ngram_size=2,  # Prevent repetition of 2-grams
    top_k=50,  # Top-k sampling
    top_p=0.95,  # Nucleus sampling
    temperature=0.7,  # Controls randomness (lower = more deterministic)
)

# Decode and print the output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Text:")
print(generated_text)
