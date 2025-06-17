import warnings
import numpy as np
from scipy.spatial.distance import cosine
from scipy.linalg import norm
from lambeq import BobcatParser, AtomicType, IQPAnsatz, NumpyModel

# Ignore parser warnings
warnings.filterwarnings('ignore')

# Set up the parser and ansatz 

# Initialise the Bobcat parser for sentence parsing
parser = BobcatParser()

# Define a basic IQP ansatz for generating parameterised quantum circuits
ansatz = IQPAnsatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1}, n_layers=1)

# Function to process sentences

def sentence_to_state(sentence):
    # Parse the sentence into a DisCoCat diagram
    diagram = parser.sentence2diagram(sentence)

    # Generate the parameterised quantum circuit
    circuit = ansatz(diagram)

    # Create a model from the circuit
    model = NumpyModel.from_diagrams([circuit])

    # Initialise the model weights
    model.initialise_weights()

    # Evaluate the circuit to get the state vector
    state_vector = model([circuit])[0]

    # Normalise and return the state vector
    return state_vector / norm(state_vector)

# Function to compute similarity 

def compute_similarity(vec1, vec2):
    # Calculate fidelity similarity
    fidelity = np.abs(np.vdot(vec1, vec2))**2
    # Calculate cosine similarity
    cos_sim = 1 - cosine(vec1, vec2)
    return fidelity, cos_sim

# Input and execution

if __name__ == '__main__':
    # Input two English sentences
    sentence1 = 'the cat sleeps'
    sentence2 = 'the cat sleeps'

    fidelities = []
    cos_sims = []

    for i in range(1000):
        try:
            # Convert sentences to quantum states
            state1 = sentence_to_state(sentence1)
            state2 = sentence_to_state(sentence2)

            # Compute similarity
            fidelity, cos_sim = compute_similarity(state1, state2)

            fidelities.append(fidelity)
            cos_sims.append(cos_sim)

            print(f"[Run {i+1}] Fidelity: {fidelity:.4f}, Cosine: {cos_sim:.4f}")

        except Exception as e:
            print(f"[Run {i+1}] Error processing sentences: {e}")

    # Compute and print averages if successful runs exist
    if fidelities and cos_sims:
        avg_fidelity = sum(fidelities) / len(fidelities)
        avg_cos_sim = sum(cos_sims) / len(cos_sims)

        print("\nAverages over 1000 runs")
        print(f"Average Fidelity: {avg_fidelity:.4f}")
        print(f"Average Cosine Similarity: {avg_cos_sim:.4f}")
    else:
        print("\nNo successful runs to compute averages.") 