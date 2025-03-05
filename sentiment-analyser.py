from transformers import pipeline

# Load the sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

# Sample input text
input_text = "I absolutely love this product! It's amazing."

# Detect sentiment
result = sentiment_analyzer(input_text)

# Print the result
print("Sentiment Detection Result:")
print(result)

# Handle multiple texts
input_texts = [
    "I absolutely love this product! It's amazing.",
    "I hate this product. It's terrible.",
    "The product is okay, but it could be better."
]

# Detect sentiment for multiple texts
results = sentiment_analyzer(input_texts)

# Print the results
print("\nSentiment Detection Results:")
for text, result in zip(input_texts, results):
    print(f"Text: {text}")
    print(f"Sentiment: {result['label']}, Confidence: {result['score']:.4f}")
    print()
