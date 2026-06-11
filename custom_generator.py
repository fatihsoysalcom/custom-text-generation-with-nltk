import nltk
from nltk.tokenize import word_tokenize
from collections import defaultdict
import random

# Download necessary NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

class CustomTextGenerator:
    def __init__(self, corpus):
        self.corpus = corpus
        self.model = self.build_model()

    def build_model(self):
        """Builds a simple Markov chain model from the corpus."""
        model = defaultdict(list)
        tokens = word_tokenize(self.corpus.lower())

        # Create pairs of (current_word, next_word)
        for i in range(len(tokens) - 1):
            current_word = tokens[i]
            next_word = tokens[i+1]
            model[current_word].append(next_word)
        return model

    def generate_text(self, length=50, start_word=None):
        """Generates text of a specified length using the trained model."""
        if not self.model:
            return "Model not trained. Please provide a corpus."

        if start_word:
            current_word = start_word.lower()
            if current_word not in self.model:
                # If start_word is not in our vocabulary, pick a random one
                current_word = random.choice(list(self.model.keys()))
        else:
            # Start with a random word from the model
            current_word = random.choice(list(self.model.keys()))

        generated_words = [current_word]

        for _ in range(length - 1):
            if current_word in self.model and self.model[current_word]:
                # Choose a random next word from the possibilities
                next_word = random.choice(self.model[current_word])
                generated_words.append(next_word)
                current_word = next_word
            else:
                # If no next word is found, break or pick another random word
                break

        return " ".join(generated_words)

# --- Example Usage ---

# A small corpus to train our custom generator
# In a real scenario, this would be a much larger and specific dataset.
custom_corpus = ""
This is a sample text to demonstrate how a custom AI tool can be built. ""
Instead of relying on general models, we can train our own for specific tasks. ""
This approach offers more control and potentially better performance for niche applications. ""
Building your own AI is a rewarding experience. ""
It allows for fine-tuning and optimization based on your unique needs.

# Initialize and train the custom generator
my_generator = CustomTextGenerator(custom_corpus)

# Generate text starting with a specific word
print("--- Generating text starting with 'custom' ---")
generated_output_specific = my_generator.generate_text(length=40, start_word="custom")
print(generated_output_specific)

# Generate text starting with a random word
print("\n--- Generating text with a random start ---")
generated_output_random = my_generator.generate_text(length=30)
print(generated_output_random)

# Generate text with a word not in the corpus (will pick a random start)
print("\n--- Generating text with a non-corpus start word ---")
generated_output_unknown = my_generator.generate_text(length=25, start_word="unrelated")
print(generated_output_unknown)
