import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter, ParameterVector
from qiskit_aer import AerSimulator
from qiskit import transpile
from datasets import load_dataset
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')
class CustomTokenizer:
  def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.word_to_id = {'<PAD>': 0, '<UNK>': 1}
        self.id_to_word = {0: '<PAD>', 1: '<UNK>'}
        self.next_id = 2
        medical_terms = [
            'patient', 'woman', 'man', 'infant', 'baby', 'child', 'doctor',
            'diagnosis', 'treatment', 'symptoms', 'disease', 'condition',
            'blood', 'heart', 'pressure', 'pain', 'fever', 'weeks', 'months',
            'years', 'presents', 'history', 'examination', 'test',
            'pregnant', 'gestation', 'delivery', 'surgery', 'medication',
            'diabetes', 'hypertension', 'pneumonia', 'cancer', 'died',
            'emergency', 'autopsy', 'died', 'suddenly', 'brings', 'physician'
        ]
        self.stop_words = {'a', 'an', 'the', 'at', 'to', 'for', 'in', 'on', 'of', 'is', 'and', 'or'}        
        for term in medical_terms:
            self.word_to_id[term] = self.next_id
            self.id_to_word[self.next_id] = term
            self.next_id += 1   
  def encode(self, text: str, max_length: int = 8) -> List[int]:
        """Convert text to token IDs (word-level), filtering stop words"""
        text = text.replace('Q: Q:', '').replace('Q:', '').strip()
        all_words = text.lower().split()
        words = []
        for word in all_words:
            # Clean punctuation
            word = ''.join(c for c in word if c.isalnum())
            # Skip stop words
            if word and word not in self.stop_words:
                words.append(word)
                if len(words) >= max_length:
                    break     
        tokens = []
        for word in words:
            # Get token ID
            if word in self.word_to_id:
                tokens.append(self.word_to_id[word])
            elif word:  # Not in vocab
                # Add to vocabulary
                if self.next_id < self.vocab_size:
                    self.word_to_id[word] = self.next_id
                    self.id_to_word[self.next_id] = word
                    tokens.append(self.next_id)
                    self.next_id += 1
                else:
                    tokens.append(1)  # <UNK>      
        # Pad to max_length
        while len(tokens) < max_length:
            tokens.append(0)  # <PAD>       
        return tokens[:max_length]   
  def decode(self, tokens: List[int]) -> str:
        """Convert token IDs back to text"""
        return ''.join([self.id_to_char.get(t, ' ') for t in tokens])
class QuantumSemanticEncoder:
    """
    Encode text into quantum states using rotation gates
    Creates semantic representations through quantum interference
    """  
    def __init__(self, n_qubits: int, vocab_size: int = 256):
        self.n_qubits = n_qubits
        self.vocab_size = vocab_size       
        # Learnable parameters for semantic encoding
        self.theta_params = np.random.randn(n_qubits, 3) * 0.1  # RY, RZ, RY rotations
        self.learning_rate = 0.02  # Reduced for stability       
        print(f"[Quantum Semantic Encoder] Initialized")
        print(f"  Qubits: {n_qubits}")
        print(f"  Learnable parameters: {self.theta_params.size}")    
    def token_to_angles(self, token_id: int, position: int) -> Tuple[float, float, float]:
        """
        Convert token ID and position to rotation angles
        This creates the semantic representation
        """
        # Normalize token ID to [0, 2π]
        base_angle = (token_id / self.vocab_size) * 2 * np.pi       
        # Position-dependent modulation
        pos_factor = (position + 1) / (self.n_qubits + 1)        
        # Three rotation angles for semantic encoding
        theta_y1 = base_angle * pos_factor
        theta_z = base_angle * (1 - pos_factor) + np.pi/4
        theta_y2 = base_angle * 0.5        
        return theta_y1, theta_z, theta_y2    
    def create_encoding_circuit(self, tokens: List[int]) -> QuantumCircuit:
        """
        Create quantum circuit that encodes tokens
        Uses rotation gates to create semantic superposition
        """
        qc = QuantumCircuit(self.n_qubits, name='Semantic_Encoding')        
        # Initialize to uniform superposition
        for i in range(self.n_qubits):
            qc.h(i)        
        # Encode each token using rotation gates
        for pos, token_id in enumerate(tokens[:self.n_qubits]):
            theta_y1, theta_z, theta_y2 = self.token_to_angles(token_id, pos)          
            # Add learnable parameters
            theta_y1 += self.theta_params[pos, 0]
            theta_z += self.theta_params[pos, 1]
            theta_y2 += self.theta_params[pos, 2]           
            # Apply rotation gates
            qc.ry(theta_y1, pos)
            qc.rz(theta_z, pos)
            qc.ry(theta_y2, pos)       
        return qc   
    def create_entanglement_layer(self, qc: QuantumCircuit) -> QuantumCircuit:
        """
        Add entanglement to capture relationships between tokens
        """
        # Circular entanglement pattern
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)
        qc.cx(self.n_qubits - 1, 0)  # Close the circle       
        return qc   
    def update_parameters(self, gradient: np.ndarray):
        """Update learnable parameters using gradient"""
        self.theta_params -= self.learning_rate * gradient
class VariationalQSEL:
    """
    Complete QSEL with Variational Quantum Circuits (VQC)
    All components built from quantum gates
    """    
    def __init__(self, max_sequence_length: int = 8):
        self.max_seq_len = max_sequence_length
        self.n_qubits = int(np.ceil(np.log2(max_sequence_length)))
        self.state_dim = 2 ** self.n_qubits        
        print(f"\n{'='*70}")
        print(f"[VARIATIONAL QSEL] Initialization")
        print(f"{'='*70}")
        print(f"  Max sequence length: {max_sequence_length}")
        print(f"  Quantum qubits: {self.n_qubits}")
        print(f"  State dimension: {self.state_dim}")       
        # Custom tokenizer (no pre-trained models)
        self.tokenizer = CustomTokenizer(vocab_size=256)       
        # Quantum semantic encoder
        self.encoder = QuantumSemanticEncoder(self.n_qubits, vocab_size=256)       
        # Importance weight parameters
        self.importance_weights = np.random.randn(max_sequence_length) * 0.1        
        print(f"  ✓ Custom tokenizer initialized")
        print(f"  ✓ Quantum encoder with {self.encoder.theta_params.size} parameters")
        print(f"{'='*70}\n")   
    def create_importance_weighting_circuit(self, qc: QuantumCircuit) -> QuantumCircuit:
        """
        Apply importance weights using controlled rotations
        Implements learned α_i from paper
        """
        for i in range(self.n_qubits):
            weight = self.importance_weights[i]
            # Use RY gate with importance weight
            qc.ry(weight, i)       
        return qc   
    def create_positional_encoding_circuit(self, qc: QuantumCircuit) -> QuantumCircuit:
        """
        Add positional information using phase gates
        Implements |position_i⟩ from paper
        """
        for i in range(self.n_qubits):
            phase = 2 * np.pi * i / self.n_qubits
            qc.p(phase, i)  # Phase gate       
        return qc   
    def encode_text_to_quantum_state(self, text: str, verbose: bool = False) -> Tuple[QuantumCircuit, Dict]:
        """
        Complete QSEL encoding pipeline with custom circuits
        """
        # Step 1: Tokenization
        tokens = self.tokenizer.encode(text, max_length=self.max_seq_len)    
        if verbose:
            print(f"\n[Encoding] '{text[:50]}...'")
            print(f"  Tokens: {tokens}")     
        # Step 2: Create base quantum circuit
        qc = QuantumCircuit(self.n_qubits, self.n_qubits, name='QSEL')     
        # Step 3: Semantic encoding
        encoding_circuit = self.encoder.create_encoding_circuit(tokens)
        qc.compose(encoding_circuit, inplace=True)
        qc.barrier()       
        # Step 4: Entanglement layer (capture token relationships)
        qc = self.encoder.create_entanglement_layer(qc)
        qc.barrier()     
        # Step 5: Importance weighting
        qc = self.create_importance_weighting_circuit(qc)
        qc.barrier()     
        # Step 6: Positional encoding
        qc = self.create_positional_encoding_circuit(qc)      
        metadata = {
            'tokens': tokens,
            'text': text,
            'num_qubits': self.n_qubits,
            'circuit_depth': qc.depth(),
            'gate_count': qc.size()
        }        
        if verbose:
            print(f"  Circuit depth: {qc.depth()}")
            print(f"  Total gates: {qc.size()}")     
        return qc, metadata  
    def measure_semantic_similarity(self, text1: str, text2: str, shots: int = 1024) -> float:
        """
        Measure similarity between two texts using quantum fidelity
        """
        qc1, _ = self.encode_text_to_quantum_state(text1)
        qc2, _ = self.encode_text_to_quantum_state(text2)   
        simulator = AerSimulator()    
        qc1_copy = qc1.copy()
        qc2_copy = qc2.copy()
        qc1_copy.measure_all()
        qc2_copy.measure_all()       
        qc1_t = transpile(qc1_copy, simulator)
        qc2_t = transpile(qc2_copy, simulator)      
        result1 = simulator.run(qc1_t, shots=shots).result()
        result2 = simulator.run(qc2_t, shots=shots).result()      
        counts1 = result1.get_counts()
        counts2 = result2.get_counts()     
        all_states = set(counts1.keys()) | set(counts2.keys())
        overlap = 0
        for state in all_states:
            p1 = counts1.get(state, 0) / shots
            p2 = counts2.get(state, 0) / shots
            overlap += np.sqrt(p1 * p2)      
        return overlap ** 2   
    def train_on_medical_pairs(self, pairs: List[Tuple[str, str, float]], epochs: int = 20):
        """
        Train quantum parameters on medical text pairs
        """
        print(f"\n[TRAINING] Learning quantum parameters...")
        print(f"  Training pairs: {len(pairs)}")
        print(f"  Epochs: {epochs}\n")       
        losses = []
        best_loss = float('inf')
        best_params = self.encoder.theta_params.copy()
        patience = 5
        no_improve = 0       
        for epoch in range(epochs):
            epoch_loss = 0.0          
            for text1, text2, target_sim in pairs:
                current_sim = self.measure_semantic_similarity(text1, text2, shots=1024)               
                loss = (current_sim - target_sim) ** 2
                epoch_loss += loss              
                gradient = np.zeros_like(self.encoder.theta_params)
                epsilon = 0.005               
                for i in range(self.encoder.theta_params.shape[0]):
                    for j in range(self.encoder.theta_params.shape[1]):
                        self.encoder.theta_params[i, j] += epsilon
                        sim_plus = self.measure_semantic_similarity(text1, text2, shots=512)
                        loss_plus = (sim_plus - target_sim) ** 2                       
                        self.encoder.theta_params[i, j] -= epsilon
                        gradient[i, j] = (loss_plus - loss) / epsilon               
                gradient = np.clip(gradient, -1.0, 1.0)               
                self.encoder.theta_params -= self.encoder.learning_rate * gradient           
            avg_loss = epoch_loss / len(pairs)
            losses.append(avg_loss)           
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_params = self.encoder.theta_params.copy()
                no_improve = 0
            else:
                no_improve += 1           
            print(f"  Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f} {'✓ (best)' if avg_loss == best_loss else ''}")          
            if no_improve >= patience:
                print(f"\n  Early stopping: No improvement for {patience} epochs")
                break      
        self.encoder.theta_params = best_params
        print(f"\n  ✓ Training complete! Best loss: {best_loss:.6f}\n")     
        plt.figure(figsize=(10, 5))
        plt.plot(losses, linewidth=2, color='steelblue', marker='o')
        plt.axhline(y=best_loss, color='red', linestyle='--', label=f'Best Loss: {best_loss:.4f}')
        plt.xlabel('Epoch', fontsize=12, fontweight='bold')
        plt.ylabel('Loss', fontsize=12, fontweight='bold')
        plt.title('QSEL Variational Training on Medical Text', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()       
        return losses
class MedicalDatasetLoader:
    """Load medical datasets from HuggingFace"""    
    @staticmethod
    def load_medical_qa(num_samples: int = 20) -> List[Dict]:
        """Load medical Q&A dataset"""
        print(f"\n[DATASET] Loading medical Q&A from HuggingFace...")        
        try:
            dataset = load_dataset("medalpaca/medical_meadow_medqa", split="train", streaming=True)
            samples = []          
            for i, item in enumerate(dataset):
                if i >= num_samples:
                    break              
                question = item['input']
                clean_question = question.replace('Q:', '').strip()             
                clean_question = clean_question.split('.')[0][:200]               
                answer = item['output'][:100] if 'output' in item else ""               
                samples.append({
                    'question': clean_question,
                    'answer': answer,
                    'text': clean_question
                })                
                print(f"  [{i+1}/{num_samples}] {clean_question[:60]}...")           
            print(f"  ✓ Loaded {len(samples)} medical Q&A samples\n")
            return samples           
        except Exception as e:
            print(f"  Error: {e}")
            return MedicalDatasetLoader.get_fallback_medical_data(num_samples)   
    @staticmethod
    def get_fallback_medical_data(num_samples: int = 20) -> List[Dict]:
        """Fallback medical examples"""
        print(f"  Using fallback medical examples...\n")      
        examples = [
            {
                'question': 'What is hypertension?',
                'answer': 'High blood pressure condition',
                'text': 'Q: What is hypertension? A: High blood pressure condition'
            },
            {
                'question': 'What causes diabetes?',
                'answer': 'Insulin deficiency or resistance',
                'text': 'Q: What causes diabetes? A: Insulin deficiency or resistance'
            },
            {
                'question': 'What is a myocardial infarction?',
                'answer': 'Heart attack due to blocked artery',
                'text': 'Q: What is a myocardial infarction? A: Heart attack due to blocked artery'
            },
            {
                'question': 'What is cancer?',
                'answer': 'Abnormal cell growth and division',
                'text': 'Q: What is cancer? A: Abnormal cell growth and division'
            },
            {
                'question': 'What is pneumonia?',
                'answer': 'Lung infection causing inflammation',
                'text': 'Q: What is pneumonia? A: Lung infection causing inflammation'
            }
        ]        
        samples = []
        for i in range(min(num_samples, len(examples) * 4)):
            idx = i % len(examples)
            samples.append(examples[idx])
            print(f"  [{i+1}/{num_samples}] {examples[idx]['question'][:60]}...")      
        return samples
class QSELVisualizer:
    """Visualization tools for QSEL circuits"""   
    @staticmethod
    def visualize_circuit(qc: QuantumCircuit, metadata: Dict):
        print(f"Text: {metadata['text'][:60]}...")
        print(f"Tokens: {metadata['tokens']}")
        print(f"\n[Circuit Properties]")
        print(f"  Qubits: {metadata['num_qubits']}")
        print(f"  Depth: {metadata['circuit_depth']}")
        print(f"  Gates: {metadata['gate_count']}")      
        gate_counts = qc.count_ops()
        print(f"\n[Gate Distribution]")
        for gate, count in sorted(gate_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {gate.upper()}: {count}")       
        print(f"\n[Circuit Diagram]")
        print(qc.draw(output='text', fold=-1))       
        decomposed = qc.decompose().decompose()
        print(f"\n[Decomposed Circuit]")
        print(f"  Elementary gates: {decomposed.size()}")
        print(f"  Decomposed depth: {decomposed.depth()}")      
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))           
            qc.draw(output='mpl', ax=axes[0, 0], style={'backgroundcolor': '#FFFFFF'})
            axes[0, 0].set_title('QSEL Circuit - High Level', fontsize=12, fontweight='bold')           
            gates = list(gate_counts.keys())
            counts = list(gate_counts.values())
            axes[0, 1].bar(gates, counts, color='steelblue', alpha=0.7, edgecolor='black')
            axes[0, 1].set_xlabel('Gate Type', fontsize=11)
            axes[0, 1].set_ylabel('Count', fontsize=11)
            axes[0, 1].set_title('Gate Distribution', fontsize=12, fontweight='bold')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(axis='y', alpha=0.3)           
            temp_qc = QuantumCircuit(decomposed.num_qubits)
            for i, instruction in enumerate(decomposed.data[:50]):
                temp_qc.append(instruction)
            temp_qc.draw(output='mpl', ax=axes[1, 0], style={'backgroundcolor': '#FFFFFF'}, fold=20)
            axes[1, 0].set_title('Decomposed Circuit (First 50 Gates)', fontsize=12, fontweight='bold')           
            metrics = ['Qubits', 'Depth', 'Gates']
            original = [qc.num_qubits, qc.depth(), qc.size()]
            decomp = [decomposed.num_qubits, decomposed.depth(), decomposed.size()]
            x = np.arange(len(metrics))
            width = 0.35
            axes[1, 1].bar(x - width/2, original, width, label='Original', color='coral', alpha=0.7)
            axes[1, 1].bar(x + width/2, decomp, width, label='Decomposed', color='lightgreen', alpha=0.7)
            axes[1, 1].set_xlabel('Metric', fontsize=11)
            axes[1, 1].set_ylabel('Count', fontsize=11)
            axes[1, 1].set_title('Circuit Complexity', fontsize=12, fontweight='bold')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(metrics)
            axes[1, 1].legend()
            axes[1, 1].grid(axis='y', alpha=0.3)           
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"\n  Matplotlib visualization failed: {e}")   
    @staticmethod
    def visualize_measurements(qc: QuantumCircuit, text: str, shots: int = 4096):
        print(f"Text: {text[:60]}...")
        print(f"Shots: {shots}\n")       
        qc_measured = qc.copy()
        qc_measured.measure_all()      
        simulator = AerSimulator()
        transpiled = transpile(qc_measured, simulator, optimization_level=3)       
        print(f"[Transpilation]")
        print(f"  Optimized depth: {transpiled.depth()}")
        print(f"  Optimized gates: {transpiled.size()}")       
        job = simulator.run(transpiled, shots=shots)
        result = job.result()
        counts = result.get_counts()       
        print(f"\n[Execution Results]")
        print(f"  Total states measured: {len(counts)}")
        print(f"  Top 5 states:")       
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        for i, (state, count) in enumerate(sorted_counts[:5]):
            prob = count / shots * 100
            print(f"    {i+1}. |{state}⟩: {count} ({prob:.2f}%)")     
        top_states = sorted_counts[:12]
        states = [s[0] for s in top_states]
        values = [s[1] for s in top_states]       
        plt.figure(figsize=(14, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(states)))
        plt.bar(range(len(states)), values, color=colors, alpha=0.8, edgecolor='black')
        plt.xlabel('Quantum State', fontsize=12, fontweight='bold')
        plt.ylabel('Measurement Counts', fontsize=12, fontweight='bold')
        plt.title(f'Quantum Measurement Distribution\n"{text[:50]}..."',
                  fontsize=13, fontweight='bold')
        plt.xticks(range(len(states)), states, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()       
        return counts
def run_complete_medical_demonstration():
    qsel = VariationalQSEL(max_sequence_length=8)  
    medical_data = MedicalDatasetLoader.load_medical_qa(num_samples=10)   
    sample_text = medical_data[1]['text']
    qc, metadata = qsel.encode_text_to_quantum_state(sample_text, verbose=True) 
    QSELVisualizer.visualize_circuit(qc, metadata)
    QSELVisualizer.visualize_measurements(qc, sample_text)
    print("[Tokenization Check]")
    for i in range(3):
        tokens = qsel.tokenizer.encode(medical_data[i]['text'])
        print(f"Text {i+1}: {medical_data[i]['text'][:50]}...")
        print(f"  Tokens: {tokens}")
        print(f"  Words: {[qsel.tokenizer.id_to_word.get(t, '?') for t in tokens if t != 0]}\n")
    test_pairs = [
        (medical_data[0]['text'], medical_data[1]['text'], "Different scenarios"),
        (medical_data[0]['text'], medical_data[2]['text'], "Different scenarios"),
        (medical_data[0]['text'], medical_data[0]['text'], "Same text (should be ~1.0)"),
    ] 
    for text1, text2, description in test_pairs:
        similarity = qsel.measure_semantic_similarity(text1, text2, shots=1024)
        print(f"Similarity: {similarity:.4f} - {description}")
        print(f"  Text 1: {text1[:50]}...")
        print(f"  Text 2: {text2[:50]}...\n")  
    training_pairs = [
        (medical_data[0]['question'], medical_data[0]['question'], 0.95), 
        (medical_data[0]['question'], medical_data[2]['question'], 0.65),  
        (medical_data[0]['question'], medical_data[1]['question'], 0.45), 
        (medical_data[1]['question'], medical_data[5]['question'], 0.30),
        (medical_data[2]['question'], medical_data[0]['question'], 0.60), 
    ]   
    losses = qsel.train_on_medical_pairs(training_pairs, epochs=15) 
    for i, sample in enumerate(medical_data[:3]):
        print(f"\n[Sample {i+1}]")
        qc, metadata = qsel.encode_text_to_quantum_state(sample['text'], verbose=True)   
    print(f"\n Features Demonstrated:")
    print(f"  Custom quantum circuits (no pre-trained models)")
    print(f" Rotation gates for semantic encoding")
    print(f" Entanglement for token relationships")
    print(f" Variational quantum learning")
    print(f" Medical dataset from HuggingFace")
    print(f" Quantum measurement and analysis")
    print(f" O(log n) qubit scaling demonstrated")
if __name__ == "__main__":
    run_complete_medical_demonstration()
