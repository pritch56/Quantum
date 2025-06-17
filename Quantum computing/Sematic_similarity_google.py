import os
import warnings
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

def load_use_model():
    """Load the Universal Sentence Encoder model from TensorFlow Hub."""
    print("Loading Universal Sentence Encoder model...")
    return hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

def sentence_to_embedding(model, sentence):
    """Convert a sentence to a normalized embedding vector."""
    embedding = model([sentence])[0].numpy()
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 0 else embedding

def compute_similarity(vec1, vec2):
    """Compute fidelity and cosine similarity."""
    fidelity = np.dot(vec1, vec2) ** 2
    cos_sim = 1 - cosine(vec1, vec2)
    return fidelity, cos_sim

def pairwise_similarity_matrix(sentences, model):
    """
    Compute cosine similarity matrix for a list of sentences.
    Returns a 2D numpy array (matrix) and a fidelity matrix.
    """
    embeddings = [sentence_to_embedding(model, s) for s in sentences]
    num_sent = len(embeddings)
    cosine_matrix = np.zeros((num_sent, num_sent))
    fidelity_matrix = np.zeros((num_sent, num_sent))

    for i in range(num_sent):
        for j in range(num_sent):
            fidelity, cos_sim = compute_similarity(embeddings[i], embeddings[j])
            cosine_matrix[i, j] = cos_sim
            fidelity_matrix[i, j] = fidelity

    return cosine_matrix, fidelity_matrix

def plot_heatmap(matrix, labels, title="Cosine Similarity Heatmap"):
    """Display a heatmap of the similarity matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt=".2f", cmap="viridis")
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def main():
    use_model = load_use_model()

    # Define a list of sentences to compare
    sentences = [
        "the cat sleeps",
        "the cac sleeps",
        "a dog is barking",
        "the kitten is resting",
        "the animal is sleeping",
        "the cat eats fish"
    ]

    print("\nSentences to compare:")
    for idx, s in enumerate(sentences):
        print(f"  [{idx+1}] {s}")

    cos_matrix, fid_matrix = pairwise_similarity_matrix(sentences, use_model)

    print("\nCosine Similarity Matrix:\n", np.round(cos_matrix, 4))
    print("\nFidelity (cosine²) Matrix:\n", np.round(fid_matrix, 4))

    plot_heatmap(cos_matrix, sentences, title="Cosine Similarity Heatmap")
    plot_heatmap(fid_matrix, sentences, title="Fidelity (Cosine²) Heatmap")

if __name__ == "__main__":
    main()
