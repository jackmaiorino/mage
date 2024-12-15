import openai
import json

# Set your OpenAI API key
openai.api_key = "your-api-key-here"

# Example card texts
card_texts = [
    "Draw two cards.",
    "Deal 3 damage to any target.",
    # Add more card texts here
]

# Dictionary to store embeddings
embeddings = {}

for text in card_texts:
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text
    )
    embeddings[text] = response['data'][0]['embedding']

# Save embeddings to a JSON file
with open('card_embeddings.json', 'w') as f:
    json.dump(embeddings, f)
