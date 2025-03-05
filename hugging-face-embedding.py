from transformers import pipeline
import numpy as np

# Load the feature extraction pipeline
feature_extractor = pipeline("feature-extraction", model="gpt2")

# Sample input text
input_text = "Generate embeddings using GPT-2."

# Generate embeddings
embeddings = feature_extractor(input_text)

# The output is a list of embeddings for each token in the input text
print("Embeddings shape:", len(embeddings[0]), len(
    embeddings[0][0]))  # (sequence_length, hidden_size)

# Get embeddings for the first token
first_token_embedding = embeddings[0][0]
print("First token embedding:", first_token_embedding)

# Compute average embedding for the entire input text
average_embedding = np.mean(embeddings[0], axis=0)
print("Average embedding:", average_embedding)
