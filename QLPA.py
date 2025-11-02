import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit import transpile
from qiskit.quantum_info import Statevector, partial_trace, DensityMatrix
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import warnings
import re
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset
warnings.filterwarnings('ignore')
class GeneralSemanticTokenizer:
    """General-purpose tokenizer for natural language"""
    def __init__(self, vocab_size: int = 1024):
        self.vocab_size = vocab_size
        self.word_to_id = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
        self.id_to_word = {0: '<PAD>', 1: '<UNK>', 2: '<BOS>', 3: '<EOS>'}
        self.next_id = 4
        self.word_freq = {}
    def add_word(self, word: str) -> int:
        """Add word to vocabulary"""
        if word not in self.word_to_id and self.next_id < self.vocab_size:
            self.word_to_id[word] = self.next_id
            self.id_to_word[self.next_id] = word
            self.next_id += 1
            return self.next_id - 1
        return self.word_to_id.get(word, 1)
    def build_vocab_from_dataset(self, texts: List[str], max_words: int = 500):
        """Build vocabulary from dataset texts"""
        word_counts = {}
        for text in texts:
            words = re.findall(r'\b[a-z]+\b', text.lower())
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        for word, count in sorted_words[:max_words]:
            self.add_word(word)
            self.word_freq[word] = count
        print(f"[Tokenizer] Built vocabulary: {self.next_id} tokens")
    def encode(self, text: str, max_length: int = 8) -> List[int]:
        """Encode text to token IDs"""
        text = text.strip().lower()
        words = re.findall(r'\b[a-z]+\b', text)
        stop_words = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'in', 'on', 'at'}
        tokens = []
        for word in words:
            if len(tokens) >= max_length:
                break
            if word in stop_words and len(tokens) > 0:
                continue
            if word in self.word_to_id:
                tokens.append(self.word_to_id[word])
            else:
                token_id = self.add_word(word)
                tokens.append(token_id)
        while len(tokens) < max_length:
            tokens.append(0)
        return tokens[:max_length]
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text"""
        words = [self.id_to_word.get(tid, '<UNK>') for tid in token_ids if tid != 0]
        return ' '.join(words)
class SemanticEmbedding:
    """FIXED: Semantic embeddings with cluster structure (simulates pre-trained)"""
    def __init__(self, vocab_size: int, embedding_dim: int = 64):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embeddings = self._initialize_semantic_embeddings(vocab_size, embedding_dim)
    def _initialize_semantic_embeddings(self, vocab_size: int, dim: int) -> np.ndarray:
        embeddings = np.random.randn(vocab_size, dim) * 0.05
        n_clusters = 8
        cluster_size = dim // n_clusters
        for i in range(vocab_size):
            cluster = i % n_clusters
            start_dim = cluster * cluster_size
            end_dim = start_dim + cluster_size
            embeddings[i, start_dim:end_dim] += 0.3
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-10)
        return embeddings
    def get_embedding(self, token_id: int) -> np.ndarray:
        """Get embedding vector for token"""
        if token_id < self.vocab_size:
            return self.embeddings[token_id]
        return self.embeddings[1]
    def compute_similarity(self, token_id1: int, token_id2: int) -> float:
        """Compute semantic similarity"""
        emb1 = self.get_embedding(token_id1).reshape(1, -1)
        emb2 = self.get_embedding(token_id2).reshape(1, -1)
        return float(cosine_similarity(emb1, emb2)[0, 0])
class QuantumSemanticEncoder:
    """Quantum encoder implementing TRUE amplitude encoding"""
    def __init__(self, n_qubits: int, embedding_dim: int = 64):
        self.n_qubits = n_qubits
        self.embedding_dim = embedding_dim
        self.importance_weights = np.random.randn(n_qubits) * 0.1
    def compute_amplitude_vector(
        self,
        embeddings: List[np.ndarray],
        positions: List[int]
    ) -> np.ndarray:
        n_states = 2 ** self.n_qubits
        amplitudes = np.zeros(n_states, dtype=complex)
        for idx, (embedding, position) in enumerate(zip(embeddings, positions)):
            if idx >= n_states:
                break
            norm = np.linalg.norm(embedding)
            if norm > 0:
                normalized_emb = embedding / norm
            else:
                normalized_emb = embedding
            token_amplitude = np.mean(normalized_emb[:8])
            position_phase = 2 * np.pi * position / len(embeddings)
            importance = np.exp(self.importance_weights[idx % self.n_qubits])
            amplitude = importance * token_amplitude * np.exp(1j * position_phase)
            state_index = idx
            amplitudes[state_index] = amplitude
        norm = np.linalg.norm(amplitudes)
        if norm > 1e-10:
            amplitudes = amplitudes / norm
        else:
            amplitudes[0] = 1.0
        return amplitudes
    def create_amplitude_encoding_circuit(
        self,
        embeddings: List[np.ndarray],
        positions: List[int]
    ) -> QuantumCircuit:
        """Create quantum circuit with TRUE amplitude encoding"""
        qc = QuantumCircuit(self.n_qubits, name='QSEL')
        amplitudes = self.compute_amplitude_vector(embeddings, positions)
        qc.initialize(amplitudes, range(self.n_qubits))
        qc.barrier(label='Encoded')
        return qc
    def create_entanglement_layer(self, qc: QuantumCircuit, layer: int = 0) -> QuantumCircuit:
        """Create entanglement layer for position-token correlation"""
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)
        if self.n_qubits > 2:
            qc.cx(self.n_qubits - 1, 0)
        return qc
class QSEL_Component:
    """Complete QSEL Component (Paper Section 3.5) - FIXED"""
    def __init__(self, max_sequence_length: int = 8, embedding_dim: int = 64):
        self.max_seq_len = max_sequence_length
        self.embedding_dim = embedding_dim
        self.n_qubits = int(np.ceil(np.log2(max(max_sequence_length, 2))))
        self.tokenizer = GeneralSemanticTokenizer(vocab_size=1024)
        self.embeddings = SemanticEmbedding(vocab_size=1024, embedding_dim=embedding_dim)
        self.encoder = QuantumSemanticEncoder(n_qubits=self.n_qubits, embedding_dim=embedding_dim)
    def encode_text(self, text: str, num_layers: int = 2) -> Tuple[QuantumCircuit, Dict]:
        """Encode text to quantum state using TRUE amplitude encoding"""
        tokens = self.tokenizer.encode(text, max_length=self.max_seq_len)
        token_embeddings = [self.embeddings.get_embedding(t) for t in tokens]
        positions = list(range(len(tokens)))
        qc = QuantumCircuit(self.n_qubits, name='QSEL')
        encoding_circuit = self.encoder.create_amplitude_encoding_circuit(
            token_embeddings, positions
        )
        qc.compose(encoding_circuit, inplace=True)
        for layer in range(num_layers):
            qc.barrier(label=f'Ent-{layer+1}')
            qc = self.encoder.create_entanglement_layer(qc, layer)
        metadata = {
            'text': text,
            'tokens': tokens,
            'words': [self.tokenizer.id_to_word.get(t, '<PAD>') for t in tokens],
            'n_qubits': self.n_qubits,
            'depth': qc.depth(),
            'gates': qc.size(),
            'encoding_type': 'TRUE_AMPLITUDE_ENCODING'
        }
        return qc, metadata
class SemanticSimilarityGraph:
    """FIXED: Builds semantic similarity graph with proper threshold"""
    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold
        self.graph = None
        self.similarity_matrix = None
    def compute_similarity_matrix(self, token_embeddings: List[np.ndarray]) -> np.ndarray:
        n_tokens = len(token_embeddings)
        sim_matrix = np.zeros((n_tokens, n_tokens))
        for i in range(n_tokens):
            for j in range(n_tokens):
                if i == j:
                    sim_matrix[i, j] = 1.0
                else:
                    emb_i = token_embeddings[i].reshape(1, -1)
                    emb_j = token_embeddings[j].reshape(1, -1)
                    sim_matrix[i, j] = float(cosine_similarity(emb_i, emb_j)[0, 0])
        self.similarity_matrix = sim_matrix
        return sim_matrix
    def build_graph(self, token_embeddings: List[np.ndarray], token_ids: List[int]) -> nx.Graph:
        sim_matrix = self.compute_similarity_matrix(token_embeddings)
        n_tokens = len(token_ids)
        G = nx.Graph()
        for i in range(n_tokens):
            G.add_node(i, token_id=token_ids[i])
        edge_count = 0
        for i in range(n_tokens):
            for j in range(i + 1, n_tokens):
                if (sim_matrix[i, j] > self.threshold and
                    token_ids[i] != 0 and token_ids[j] != 0):
                    G.add_edge(i, j, weight=sim_matrix[i, j])
                    edge_count += 1
        self.graph = G
        return G
    def get_entanglement_pairs(self) -> List[Tuple[int, int, float]]:
        if self.graph is None:
            return []
        pairs = []
        for (i, j, data) in self.graph.edges(data=True):
            pairs.append((i, j, data['weight']))
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs
class QuantumEntanglementLayer:
    """FIXED: Quantum entanglement with cleaner graph state implementation"""
    def __init__(self, n_qubits: int, n_tokens: int):
        self.n_qubits = n_qubits
        self.n_tokens = n_tokens
    def similarity_to_angle(self, similarity: float) -> float:
        """Convert similarity to entanglement angle"""
        return (similarity ** 2) * np.pi / 2
    def create_graph_state_entanglement(
        self,
        qc: QuantumCircuit,
        entanglement_pairs: List[Tuple[int, int, float]]
    ) -> QuantumCircuit:
        qc.barrier(label='Graph-Init')
        for q in range(self.n_qubits):
            qc.h(q)
        qc.barrier(label='Graph-Entangle')
        for (token_i, token_j, similarity) in entanglement_pairs:
            qubit_i = token_i % self.n_qubits
            qubit_j = token_j % self.n_qubits
            if qubit_i != qubit_j:
                qc.cz(qubit_i, qubit_j)
                theta = self.similarity_to_angle(similarity)
                qc.rz(theta, qubit_i)
                qc.rz(theta, qubit_j)
        qc.barrier(label='Graph-Complete')
        return qc
class QSEN_Component:
    """Complete QSEN Component (Paper Section 3.6) - FIXED"""
    def __init__(self, n_tokens: int, embedding_dim: int = 64):
        self.max_tokens = n_tokens
        self.n_qubits = int(np.ceil(np.log2(max(n_tokens, 2))))
        self.embedding_dim = embedding_dim
        self.similarity_graph = SemanticSimilarityGraph(threshold=0.3)
        self.entanglement_layer = QuantumEntanglementLayer(self.n_qubits, n_tokens)
    def apply_attention(
        self,
        qc: QuantumCircuit,
        token_embeddings: List[np.ndarray],
        token_ids: List[int]
    ) -> Tuple[QuantumCircuit, Dict]:
        """Apply quantum attention via graph state entanglement"""
        graph = self.similarity_graph.build_graph(token_embeddings, token_ids)
        pairs = self.similarity_graph.get_entanglement_pairs()
        qc.barrier(label='QSEN-Start')
        qc = self.entanglement_layer.create_graph_state_entanglement(qc, pairs)
        qc.barrier(label='QSEN-End')
        metadata = {
            'n_edges': len(pairs),
            'graph_density': nx.density(graph) if graph.number_of_nodes() > 1 else 0,
            'pairs': pairs[:5],
            'entanglement_type': 'GRAPH_STATE_FORMULA',
            'similarity_matrix': self.similarity_graph.similarity_matrix
        }
        return qc, metadata
class QLPAValidator:
    """Validation suite for QLPA components"""
    def __init__(self, qlpa_pipeline):
        self.pipeline = qlpa_pipeline
        self.simulator = AerSimulator(method='statevector')
    def validate_qsen_entanglement(self, text: str) -> Dict:
        print(f"\n[Validation] Testing entanglement for: '{text[:50]}...'")
        qc, meta = self.pipeline.process_text(text, verbose=False)
        qc_sv = qc.copy()
        qc_sv.save_statevector()
        result = self.simulator.run(qc_sv).result()
        statevector = result.get_statevector()
        n_qubits = self.pipeline.qsel.n_qubits
        qubits_to_trace = list(range(n_qubits // 2, n_qubits))
        rho = DensityMatrix(statevector)
        rho_reduced = partial_trace(rho, qubits_to_trace)
        eigenvalues = np.linalg.eigvalsh(rho_reduced.data)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        entropy = -np.sum(eigenvalues * np.log2(eigenvalues + 1e-10))
        max_entropy = np.log2(2 ** (n_qubits // 2))
        entanglement_ratio = entropy / max_entropy if max_entropy > 0 else 0
        result = {
            'entropy': entropy,
            'max_entropy': max_entropy,
            'entanglement_ratio': entanglement_ratio,
            'n_edges': meta.get('qsen_edges', 0),
            'is_entangled': entropy > 0.1
        }
        print(f"  Entanglement Entropy: {entropy:.4f} bits")
        print(f"  Maximum Entropy: {max_entropy:.4f} bits")
        print(f"  Entanglement Ratio: {entanglement_ratio:.2%}")
        print(f"  Status: {' ENTANGLED' if result['is_entangled'] else ' NOT ENTANGLED'}")
        return result
    def measure_actual_complexity(self) -> List[Dict]:
      print("\n[Validation] Measuring complexity scaling...")
      results = []
      sequence_lengths = [4, 8, 16, 32]
      old_max_len = self.pipeline.max_seq_len
      old_qsel_qubits = self.pipeline.qsel.n_qubits
      old_encoder_weights = self.pipeline.qsel.encoder.importance_weights.copy()
      old_qsen_qubits = self.pipeline.qsen.n_qubits
      old_ent_layer = self.pipeline.qsen.entanglement_layer
      for n in sequence_lengths:
        test_text = " ".join(["word"] * n)
        self.pipeline.max_seq_len = n
        self.pipeline.qsel.max_seq_len = n
        new_qubits = int(np.ceil(np.log2(max(n, 2))))
        self.pipeline.qsel.n_qubits = new_qubits
        self.pipeline.qsel.encoder.n_qubits = new_qubits
        self.pipeline.qsel.encoder.importance_weights = np.random.randn(new_qubits) * 0.1
        self.pipeline.qsen.n_qubits = new_qubits
        self.pipeline.qsen.max_tokens = n
        self.pipeline.qsen.entanglement_layer = QuantumEntanglementLayer(new_qubits, n)
        qc, meta = self.pipeline.process_text(test_text, verbose=False)
        theoretical_depth = int(np.ceil(np.log2(n))) * 5
        actual_depth = meta['total_depth']
        result = {
            'n': n,
            'theoretical_log_n': int(np.ceil(np.log2(n))),
            'theoretical_depth': theoretical_depth,
            'actual_depth': actual_depth,
            'ratio': actual_depth / theoretical_depth if theoretical_depth > 0 else 0
        }
        results.append(result)
        print(f"  n={n:3d}: log(n)={result['theoretical_log_n']:2d}, "
              f"theoretical={theoretical_depth:3d}, actual={actual_depth:3d}, "
              f"ratio={result['ratio']:.2f}x")
        self.pipeline.max_seq_len = old_max_len
        self.pipeline.qsel.max_seq_len = old_max_len
        self.pipeline.qsel.n_qubits = old_qsel_qubits
        self.pipeline.qsel.encoder.n_qubits = old_qsel_qubits
        self.pipeline.qsel.encoder.importance_weights = old_encoder_weights
        self.pipeline.qsen.n_qubits = old_qsen_qubits
        self.pipeline.qsen.max_tokens = old_max_len
        self.pipeline.qsen.entanglement_layer = old_ent_layer
      return results
    def validate_similarity_graph(self, text: str) -> Dict:
        """Validate semantic similarity graph construction"""
        print(f"\n[Validation] Testing similarity graph for: '{text[:50]}...'")
        qc, meta = self.pipeline.process_text(text, verbose=False)
        tokens = meta['tokens']
        token_embeddings = [self.pipeline.qsel.embeddings.get_embedding(t) for t in tokens]
        sim_matrix = self.pipeline.qsen.similarity_graph.compute_similarity_matrix(token_embeddings)
        non_pad_tokens = [i for i, t in enumerate(tokens) if t != 0]
        n_non_pad = len(non_pad_tokens)
        if n_non_pad > 1:
            similarities = []
            for i in non_pad_tokens:
                for j in non_pad_tokens:
                    if i < j:
                        similarities.append(sim_matrix[i, j])
            result = {
                'n_tokens': n_non_pad,
                'n_edges': meta['qsen_edges'],
                'avg_similarity': np.mean(similarities),
                'max_similarity': np.max(similarities),
                'min_similarity': np.min(similarities),
                'threshold': self.pipeline.qsen.similarity_graph.threshold
            }
            print(f"  Non-padding tokens: {n_non_pad}")
            print(f"  Semantic edges: {result['n_edges']}")
            print(f"  Avg similarity: {result['avg_similarity']:.3f}")
            print(f"  Max similarity: {result['max_similarity']:.3f}")
            print(f"  Threshold: {result['threshold']:.3f}")
        else:
            result = {'n_tokens': n_non_pad, 'n_edges': 0}
        return result
class QLPA_Pipeline:
    """FIXED QLPA Pipeline with proper QSEN and validation"""
    def __init__(self, max_seq_len: int = 8, embedding_dim: int = 64):
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        print(f"Max Sequence Length: {max_seq_len}")
        print(f"Embedding Dimension: {embedding_dim}")
        self.qsel = QSEL_Component(max_seq_len, embedding_dim)
        print(f"QSEL initialized ({self.qsel.n_qubits} qubits) - TRUE AMPLITUDE ENCODING")
        self.qsen = QSEN_Component(max_seq_len, embedding_dim)
        print(f"QSEN initialized - GRAPH STATE (threshold=0.3)")
        self.validator = QLPAValidator(self)
        print(f" Validator initialized")
    def load_dataset(self, dataset_name: str = "wikitext", split: str = "train", num_samples: int = 100):
        """Load dataset"""
        print(f"[Dataset] Loading {dataset_name}...")
        try:
            if dataset_name == "wikitext":
                dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split, streaming=True)
            else:
                dataset = load_dataset(dataset_name, split=split, streaming=True)
            texts = []
            for i, item in enumerate(dataset):
                if i >= num_samples:
                    break
                text = item.get('text', '')
                if len(text) > 50:
                    texts.append(text[:200])
            print(f"Loaded {len(texts)} samples from {dataset_name}")
            self.qsel.tokenizer.build_vocab_from_dataset(texts, max_words=500)
            return texts
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
            print("Using fallback dataset...")
            texts = [
                "The quick brown fox jumps over the lazy dog in the garden",
                "Machine learning algorithms process large amounts of data efficiently",
                "Natural language processing enables computers to understand human language",
                "Quantum computing harnesses quantum mechanics for computation",
                "Deep neural networks learn hierarchical representations from data",
                "The weather forecast predicts sunny skies and warm temperatures",
                "Scientists discovered a new method for solving complex problems",
                "Technology advances rapidly transforming society and economy",
            ]
            self.qsel.tokenizer.build_vocab_from_dataset(texts, max_words=200)
            return texts
    def process_text(self, text: str, verbose: bool = True) -> Tuple[QuantumCircuit, Dict]:
        """Complete QLPA processing pipeline"""
        if verbose:
            print(f"\n[QLPA] Processing: '{text[:60]}...'")
        qc, qsel_meta = self.qsel.encode_text(text, num_layers=2)
        if verbose:
            print(f"  QSEL: {qsel_meta['depth']} depth, {qsel_meta['gates']} gates")
        tokens = qsel_meta['tokens']
        token_embeddings = [self.qsel.embeddings.get_embedding(t) for t in tokens]
        qc, qsen_meta = self.qsen.apply_attention(qc, token_embeddings, tokens)
        if verbose:
            print(f"  QSEN: {qsen_meta['n_edges']} semantic edges")
            print(f"  Final: {qc.depth()} depth, {qc.size()} gates")
        metadata = {
            **qsel_meta,
            'qsen_edges': qsen_meta['n_edges'],
            'qsen_density': qsen_meta['graph_density'],
            'qsen_pairs': qsen_meta['pairs'],
            'total_depth': qc.depth(),
            'total_gates': qc.size(),
        }
        return qc, metadata
    def visualize_circuit(self, qc: QuantumCircuit, metadata: Dict):
        """Visualize quantum circuit"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        qc.draw(output='mpl', ax=axes[0], style={'backgroundcolor': '#FFFFFF'})
        axes[0].set_title(f'QLPA Circuit (FIXED)\n"{metadata["text"][:40]}..."',
                         fontsize=11, fontweight='bold')
        metrics = [
            f"Tokens: {len([t for t in metadata['tokens'] if t != 0])}",
            f"Qubits: {metadata['n_qubits']}",
            f"Circuit Depth: {metadata['total_depth']}",
            f"Total Gates: {metadata['total_gates']}",
            f"QSEN Edges: {metadata['qsen_edges']}",
            f"Graph Density: {metadata['qsen_density']:.3f}",
        ]
        axes[1].axis('off')
        axes[1].text(0.1, 0.9, 'QLPA Metrics (FIXED)', fontsize=14, fontweight='bold',
                    transform=axes[1].transAxes)
        for i, metric in enumerate(metrics):
            axes[1].text(0.1, 0.80 - i*0.10, f"• {metric}", fontsize=10,
                        transform=axes[1].transAxes)
        plt.tight_layout()
        plt.show()
    def run_validation_suite(self, texts: List[str]):
        """Run complete validation suite"""
        self.validator.validate_qsen_entanglement(texts[0])
        self.validator.validate_similarity_graph(texts[0])
        self.validator.measure_actual_complexity()
def run_qlpa_fixed_demo():
    """Complete QLPA demonstration with fixes"""
    qlpa = QLPA_Pipeline(max_seq_len=8, embedding_dim=64)
    texts = qlpa.load_dataset("wikitext", num_samples=50)
    print(f"\n[Sample Texts]")
    for i, text in enumerate(texts[:5]):
        print(f"  {i+1}. {text[:70]}...")
    results = []
    for i, text in enumerate(texts[:5]):
        print(f"\n--- Text {i+1} ---")
        qc, metadata = qlpa.process_text(text, verbose=True)
        results.append((qc, metadata))
        if i == 0:
            print("\n[Circuit Diagram]")
            print(qc.draw(output='text', fold=-1))
            qlpa.visualize_circuit(qc, metadata)
    qlpa.run_validation_suite(texts[:3])
    avg_edges = np.mean([m['qsen_edges'] for _, m in results])
    avg_depth = np.mean([m['total_depth'] for _, m in results])
    avg_gates = np.mean([m['total_gates'] for _, m in results])
    print(f"Average QSEN Edges: {avg_edges:.1f}")
    print(f"Average Circuit Depth: {avg_depth:.1f}")
    print(f"Average Gates: {avg_gates:.1f}")
if __name__ == "__main__":
    run_qlpa_fixed_demo()
