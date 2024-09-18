import nltk
from nltk.corpus import brown
import string
import gensim.downloader as api
from transformers import BertTokenizer, BertForMaskedLM
import torch
from difflib import get_close_matches
import tkinter as tk
from tkinter import messagebox

#bypass SSL certificate verification
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Download necessary NLTK resources
nltk.download('brown')
nltk.download('punkt')

# Load and preprocess the Brown Corpus
brown_words = brown.words()

def preprocess_word(word):
    return word.lower().translate(str.maketrans('', '', string.punctuation))

vocabulary = set(preprocess_word(word) for word in brown_words if word.isalpha())

# Load Word Embeddings (Word2Vec)
word_vectors = api.load("glove-wiki-gigaword-100")

# Function to find similar words using word embeddings
def get_similar_words(word):
    if word in word_vectors:
        return [w for w, _ in word_vectors.most_similar(word, topn=5)]
    return []

# Load BERT model and tokenizer for masked language model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# Non-Word Error Detection
def correct_non_word_error(word, vocabulary, n=3):
    word = preprocess_word(word)
    suggestions = get_close_matches(word, vocabulary, n=n)
    return suggestions

# BERT-based Real-Word Error Detection
def correct_real_word_error(sentence, word, position):
    """
    Use BERT to predict the correct word in the given position of the sentence.
    """
    tokens = tokenizer.tokenize(sentence)
    
    # Mask the word at the given position
    tokens[position] = '[MASK]'
    
    # Reconstruct the sentence
    masked_sentence = tokenizer.convert_tokens_to_string(tokens)
    
    # Tokenize the masked sentence and create tensor
    input_ids = tokenizer.encode(masked_sentence, return_tensors='pt')
    
    # Predict the masked word
    with torch.no_grad():
        outputs = model(input_ids)
        predictions = outputs.logits
    
    # Get the predicted token ID for the masked position
    masked_index = input_ids[0].tolist().index(tokenizer.mask_token_id)
    predicted_id = torch.argmax(predictions[0, masked_index]).item()
    
    # Convert the predicted token ID to a word
    predicted_word = tokenizer.decode([predicted_id]).strip()

    # Ensure we are not splitting the word into characters (e.g., a, t, e for 'ate')
    if len(predicted_word) > 1 and predicted_word != word:
        return predicted_word
    return None

# Real-Word Error Detection using Word Embeddings + BERT
def spell_check_real_word_errors(text):
    tokens = nltk.word_tokenize(text)
    real_word_errors = []

    # Iterate over each token to see if it's a real-word error
    for i, word in enumerate(tokens):
        if word.isalpha() and word.lower() in vocabulary:
            # Check context using BERT for real-word correction
            correction = correct_real_word_error(text, word, i)
            if correction:
                real_word_errors.append((word, correction))
            
            # Additionally, use word embeddings for semantic similarity suggestions
            similar_words = get_similar_words(word.lower())
            if similar_words:
                real_word_errors.append((word, similar_words))
    
    return real_word_errors

# Full Spell Check System
def spell_check_text(text, vocabulary):
    tokens = nltk.word_tokenize(text)
    errors = []

    # Check non-word errors first
    non_word_errors = False
    for word in tokens:
        if preprocess_word(word) not in vocabulary:
            suggestions = correct_non_word_error(word, vocabulary)
            if suggestions:
                errors.append((word, suggestions))
                non_word_errors = True
    
    # Check real-word errors only if no non-word errors are found
    if not non_word_errors:
        real_word_errors = spell_check_real_word_errors(text)
        errors.extend(real_word_errors)

    return errors

# Tkinter GUI for Spell Checker
class SpellCheckerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Spell Checker with BERT and Word Embeddings")

        # Create a Text widget for input
        self.text = tk.Text(self.root, height=15, width=80)
        self.text.pack(padx=10, pady=10)

        # Button to trigger spell check
        self.check_button = tk.Button(self.root, text="Check Spelling", command=self.check_spelling)
        self.check_button.pack(pady=5)

    def check_spelling(self):
        # Get input text
        input_text = self.text.get("1.0", "end-1c")

        # Run spell check
        errors = spell_check_text(input_text, vocabulary)

        # Display results
        if errors:
            for word, suggestions in errors:
                messagebox.showinfo("Spelling Error", f"Word: {word}\nSuggestions: {', '.join(suggestions)}")
        else:
            messagebox.showinfo("No Errors", "No spelling errors found!")

# Initialize Tkinter window
root = tk.Tk()
gui = SpellCheckerGUI(root)
root.mainloop()
