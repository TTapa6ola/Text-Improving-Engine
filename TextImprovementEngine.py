import re
from sentence_transformers import SentenceTransformer, util
import numpy as np
import os

def load_text_from_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def load_phrases_from_file(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file.readlines()]

def get_user_input():
    while True:
        standard_phrases_path = input("Enter the path to the file containing standard phrases: ")
        sample_text_path = input("Enter the path to the file containing sample text: ")

        if not os.path.exists(standard_phrases_path) or not os.path.exists(sample_text_path):
            print("Invalid file path. Please provide valid paths.")
            try_again = input("Do you want to try again? (yes/no): ")
            if try_again.lower() != 'yes' and try_again.lower() != 'y':
                return None, None
        else:
            standard_phrases = load_phrases_from_file(standard_phrases_path)
            sample_text = load_text_from_file(sample_text_path)
            return standard_phrases, sample_text

def embed_text(model, text):
    sentences = re.split(r'[.!?]', text)
    return model.encode(sentences)

def find_similar_phrases(sentence_embeddings, standard_phrases_embeddings, standard_phrases, sentences, threshold=0.3):
    similar_phrases = []

    for i, sentence_emb in enumerate(sentence_embeddings):
        scores = util.pytorch_cos_sim(sentence_emb, standard_phrases_embeddings)[0]
        top_match = np.argmax(scores)

        if scores[top_match] >= threshold:
            similar_phrases.append({
                'original': sentences[i],
                'replacement': standard_phrases[top_match],
                'similarity_score': scores[top_match]
            })

    return similar_phrases

def format_sentence(sentence, max_width=90):
    if len(sentence) > max_width:
        return sentence[:max_width-3] + "..."
    else:
        return sentence

def main():
    standard_phrases, sample_text = get_user_input()

    if standard_phrases is not None and sample_text is not None:
        model = SentenceTransformer('all-MiniLM-L6-v2')

        standard_embeddings = model.encode(standard_phrases)
        sentence_embeddings = embed_text(model, sample_text)
        sentences = re.split(r'[.!?]', sample_text)

        suggestions = find_similar_phrases(sentence_embeddings, standard_embeddings, standard_phrases, sentences)

        print("\nStandard Phrases:")
        print(standard_phrases)
        print("\nSample Text:")
        print(sample_text)
        print("\nAnalysis Results:")
        print("{:<90} {:<30} {:<10}".format("Original Phrase", "Suggestion", "Score"))
        print("-" * 130)

        for suggestion in suggestions:
            formatted_original = format_sentence(suggestion['original'])
            formatted_replacement = format_sentence(suggestion['replacement'], max_width=30)
            print("{:<90} {:<30} {:.2f}".format(formatted_original, formatted_replacement, suggestion['similarity_score']))

if __name__ == "__main__":
    main()
