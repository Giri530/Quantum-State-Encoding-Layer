import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import StatePreparation
from qiskit_ibm_runtime import QiskitRuntimeService, Session, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from datasets import load_dataset
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')
class QuantumStateEncodingLayer:    
    def __init__(self, max_sequence_length: int = 8, vocab_size: int = 1000):
        self.max_seq_len = max_sequence_length
        self.vocab_size = vocab_size
        self.n_qubits = int(np.ceil(np.log2(max_sequence_length)))       
        print(f"[QSEL] Initialized with:")
        print(f"  - Max sequence length: {max_sequence_length}")
        print(f"  - Required qubits: {self.n_qubits} (log2({max_sequence_length}))")
        print(f"  - Vocabulary size: {vocab_size}")      
    def normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        flat_embeddings = embeddings.flatten()
        target_size = 2 ** self.n_qubits
        
        if len(flat_embeddings) < target_size:
            flat_embeddings = np.pad(flat_embeddings, 
                                     (0, target_size - len(flat_embeddings)), 
                                     mode='constant')
        else:
            flat_embeddings = flat_embeddings[:target_size]
        norm = np.linalg.norm(flat_embeddings)
        if norm > 0:
            return flat_embeddings / norm
        else:
            return np.ones(target_size) / np.sqrt(target_size) 
    def encode_text_to_quantum_state(self, 
                                    text: str, 
                                    tokenizer) -> Tuple[QuantumCircuit, np.ndarray]:   
        tokens = tokenizer.encode(text, max_length=self.max_seq_len, 
                                 truncation=True, padding='max_length')
        tokens = np.array(tokens[:self.max_seq_len])
        embeddings = tokens / self.vocab_size
        positions = np.arange(len(embeddings))
        positional_phase = np.exp(2j * np.pi * positions / len(positions))
        complex_embeddings = embeddings * positional_phase
        amplitudes = self.normalize_embeddings(complex_embeddings)
        qc = QuantumCircuit(self.n_qubits, name='QSEL')
        state_prep = StatePreparation(amplitudes)
        qc.append(state_prep, range(self.n_qubits))      
        return qc, amplitudes  
    def decompose_circuit(self, qc: QuantumCircuit, levels: int = 3) -> QuantumCircuit:
        decomposed = qc.decompose()
        for _ in range(levels - 1):
            decomposed = decomposed.decompose()
        return decomposed   
    def visualize_circuit_detailed(self, qc: QuantumCircuit, text: str):
        decomposed_qc = self.decompose_circuit(qc, levels=3)        
        print(f"\n{'='*80}")
        print(f"CIRCUIT ANALYSIS: '{text[:50]}...'")
        print(f"{'='*80}")       
        print(f"\n[Original Circuit]")
        print(f"  - Qubits: {qc.num_qubits}")
        print(f"  - Depth: {qc.depth()}")
        print(f"  - Operations: {qc.size()}")       
        print(f"\n[Decomposed Circuit - Elementary Gates]")
        print(f"  - Qubits: {decomposed_qc.num_qubits}")
        print(f"  - Depth: {decomposed_qc.depth()}")
        print(f"  - Elementary gates: {decomposed_qc.size()}")
        gate_counts = decomposed_qc.count_ops()
        print(f"\n[Gate Breakdown]")
        for gate, count in sorted(gate_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {gate.upper()}: {count}")
        print(f"\n[High-Level Circuit Diagram]")
        print(qc.draw(output='text', fold=-1))      
        print(f"\n[Decomposed Circuit - First 100 Gates]")
        temp_qc = QuantumCircuit(decomposed_qc.num_qubits)
        gate_limit = min(100, len(decomposed_qc.data))
        for i in range(gate_limit):
            temp_qc.append(decomposed_qc.data[i])
        print(temp_qc.draw(output='text', fold=100))       
        if len(decomposed_qc.data) > 100:
            print(f"\n  ... (showing first 100 of {len(decomposed_qc.data)} gates)")
        try:
            fig = plt.figure(figsize=(16, 8))
            ax1 = plt.subplot(2, 2, 1)
            try:
                qc.draw(output='mpl', ax=ax1, style={'backgroundcolor': '#FFFFFF'})
                ax1.set_title(f'Original QSEL Circuit\n{text[:40]}...', fontsize=10)
            except Exception as e:
                ax1.text(0.5, 0.5, f'Circuit visualization unavailable\nUse text output above\n\nInstall: pip install pylatexenc', 
                        ha='center', va='center', fontsize=10, transform=ax1.transAxes)
                ax1.set_title('Original Circuit (see text output)', fontsize=10)
                ax1.axis('off')
            ax2 = plt.subplot(2, 2, 2)
            gates = list(gate_counts.keys())
            counts = list(gate_counts.values())
            colors = plt.cm.Set3(range(len(gates)))
            ax2.bar(gates, counts, color=colors, alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Gate Type', fontsize=10)
            ax2.set_ylabel('Count', fontsize=10)
            ax2.set_title('Elementary Gate Distribution', fontsize=12, fontweight='bold')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(axis='y', alpha=0.3)
            ax3 = plt.subplot(2, 2, 3)
            metrics = ['Qubits', 'Depth', 'Gates']
            original_values = [qc.num_qubits, qc.depth(), qc.size()]
            decomposed_values = [decomposed_qc.num_qubits, decomposed_qc.depth(), decomposed_qc.size()]           
            x = np.arange(len(metrics))
            width = 0.35
            ax3.bar(x - width/2, original_values, width, label='Original', color='steelblue', alpha=0.7)
            ax3.bar(x + width/2, decomposed_values, width, label='Decomposed', color='coral', alpha=0.7)
            ax3.set_xlabel('Metric', fontsize=10)
            ax3.set_ylabel('Count', fontsize=10)
            ax3.set_title('Circuit Complexity Comparison', fontsize=12, fontweight='bold')
            ax3.set_xticks(x)
            ax3.set_xticklabels(metrics)
            ax3.legend()
            ax3.grid(axis='y', alpha=0.3)
            ax4 = plt.subplot(2, 2, 4)
            try:
                if gate_limit < len(decomposed_qc.data):
                    temp_qc.draw(output='mpl', ax=ax4, style={'backgroundcolor': '#FFFFFF'}, fold=20)
                    ax4.set_title(f'Decomposed Circuit (First {gate_limit} gates)', fontsize=10)
                else:
                    decomposed_qc.draw(output='mpl', ax=ax4, style={'backgroundcolor': '#FFFFFF'}, fold=20)
                    ax4.set_title('Fully Decomposed Circuit', fontsize=10)
            except Exception as e:
                ax4.text(0.5, 0.5, f'Decomposed circuit visualization\nSee text output above\n\nInstall: pip install pylatexenc', 
                        ha='center', va='center', fontsize=10, transform=ax4.transAxes)
                ax4.set_title('Decomposed Circuit (see text output)', fontsize=10)
                ax4.axis('off')           
            plt.tight_layout()
            plt.show()           
        except Exception as e:
            print(f"\n[Matplotlib Error] {e}")
            print("  → Circuit diagrams shown in text format above")    
    def visualize_quantum_state(self, amplitudes: np.ndarray, title: str = "QSEL State"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        basis_states = [f"|{i:0{self.n_qubits}b}⟩" for i in range(len(amplitudes))]
        colors = plt.cm.viridis(np.linspace(0, 1, len(amplitudes)))
        ax1.bar(range(len(amplitudes)), np.abs(amplitudes), color=colors, alpha=0.8, edgecolor='black')
        ax1.set_xlabel('Basis State', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Amplitude |α_i|', fontsize=11, fontweight='bold')
        ax1.set_title(f'{title} - Amplitudes', fontsize=13, fontweight='bold')
        ax1.set_xticks(range(len(amplitudes)))
        ax1.set_xticklabels(basis_states, rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)
        probabilities = np.abs(amplitudes) ** 2
        colors = plt.cm.plasma(np.linspace(0, 1, len(probabilities)))
        ax2.bar(range(len(probabilities)), probabilities, color=colors, alpha=0.8, edgecolor='black')
        ax2.set_xlabel('Basis State', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Probability |α_i|²', fontsize=11, fontweight='bold')
        ax2.set_title(f'{title} - Measurement Probabilities', fontsize=13, fontweight='bold')
        ax2.set_xticks(range(len(probabilities)))
        ax2.set_xticklabels(basis_states, rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)       
        plt.tight_layout()
        plt.show()
class MedicalDatasetLoader:
    @staticmethod
    def load_medical_abstracts(num_samples: int = 50) -> List[Dict]:
        print(f"\n[Medical Dataset] Loading PubMed abstracts...")
        try:
            dataset = load_dataset("pubmed", split="train", streaming=True)            
            samples = []
            for i, item in enumerate(dataset):
                if i >= num_samples:
                    break
                samples.append({
                    'text': item['MedlineCitation']['Article']['Abstract']['AbstractText'][0],
                    'title': item['MedlineCitation']['Article']['ArticleTitle'],
                    'source': 'PubMed'
                })
                print(f"  Sample {i+1}: {item['MedlineCitation']['Article']['ArticleTitle'][:70]}...")           
            return samples
        except:
            print("  → PubMed dataset unavailable, trying alternative...")
            return MedicalDatasetLoader.load_medical_questions(num_samples)    
    @staticmethod
    def load_medical_questions(num_samples: int = 50) -> List[Dict]:
        print(f"\n[Medical Dataset] Loading medical Q&A dataset...")
        try:
            dataset = load_dataset("medalpaca/medical_meadow_medqa", split="train")            
            samples = []
            for i in range(min(num_samples, len(dataset))):
                item = dataset[i]
                text = f"Question: {item['input']} Answer: {item['output']}"
                samples.append({
                    'text': text[:500], 
                    'title': item['input'][:100],
                    'source': 'Medical Q&A'
                })
                print(f"  Sample {i+1}: {item['input'][:70]}...")           
            return samples
        except:
            print("  → Medical Q&A dataset unavailable, trying alternative...")
            return MedicalDatasetLoader.load_healthfact(num_samples)   
    @staticmethod
    def load_healthfact(num_samples: int = 50) -> List[Dict]:
        print(f"\n[Medical Dataset] Loading HealthFact dataset...")
        try:
            dataset = load_dataset("health_fact", split="train")            
            samples = []
            for i in range(min(num_samples, len(dataset))):
                item = dataset[i]
                samples.append({
                    'text': item['main_text'][:500],
                    'title': item['claim'][:100],
                    'source': 'HealthFact'
                })
                print(f"  Sample {i+1}: {item['claim'][:70]}...")            
            return samples
        except Exception as e:
            print(f"  → Error loading datasets: {e}")
            print("  → Using fallback medical examples...")
            return MedicalDatasetLoader.get_fallback_medical_data(num_samples)   
    @staticmethod
    def get_fallback_medical_data(num_samples: int = 50) -> List[Dict]:
        print(f"\n[Fallback] Using curated medical examples...")        
        medical_texts = [
            {
                'text': 'Diabetes mellitus is characterized by chronic hyperglycemia with disturbances of carbohydrate, fat and protein metabolism resulting from defects in insulin secretion, insulin action, or both.',
                'title': 'Diabetes Mellitus Overview',
                'source': 'Medical Encyclopedia'
            },
            {
                'text': 'Hypertension or high blood pressure is a chronic medical condition in which the blood pressure in the arteries is persistently elevated, leading to increased risk of heart disease and stroke.',
                'title': 'Hypertension Definition',
                'source': 'Medical Encyclopedia'
            },
            {
                'text': 'Cancer is a group of diseases involving abnormal cell growth with the potential to invade or spread to other parts of the body through metastasis.',
                'title': 'Cancer Pathophysiology',
                'source': 'Medical Encyclopedia'
            },
            {
                'text': 'Myocardial infarction occurs when blood flow decreases or stops to a part of the heart causing damage to the heart muscle, commonly known as a heart attack.',
                'title': 'Myocardial Infarction',
                'source': 'Medical Encyclopedia'
            },
            {
                'text': 'Alzheimers disease is a progressive neurodegenerative disorder that destroys memory and other important mental functions through the degeneration of brain cells.',
                'title': 'Alzheimers Disease',
                'source': 'Medical Encyclopedia'
            }
        ]        
        samples = []
        for i in range(min(num_samples, len(medical_texts) * 10)):
            idx = i % len(medical_texts)
            samples.append(medical_texts[idx])
            print(f"  Sample {i+1}: {medical_texts[idx]['title']}")       
        return samples
class QLPAMedicalExperiment:
    def __init__(self, ibm_token: str = None, platform: str = 'simulator'):
        self.qsel = QuantumStateEncodingLayer(max_sequence_length=8, vocab_size=30000)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.ibm_token = ibm_token
        self.platform = platform       
        if ibm_token and platform == 'ibm':
            print("\n[IBM Quantum] Connecting to IBM Quantum services...")
            try:
                QiskitRuntimeService.save_account(channel="ibm_quantum", 
                                                 token=ibm_token, 
                                                 overwrite=True)
                print("  ✓ IBM Quantum connection successful")
            except Exception as e:
                print(f"  ✗ IBM Quantum connection failed: {e}")
                print("  → Falling back to local simulator")
                self.platform = 'simulator'
        else:
            print(f"\n[Backend] Using {platform} for quantum execution")   
    def run_complete_analysis(self, text: str, title: str):
        print(f"\n{'='*80}")
        print(f"[QUANTUM ENCODING ANALYSIS]")
        print(f"Title: {title}")
        print(f"Text: {text[:100]}...")
        print(f"{'='*80}")
        qc, amplitudes = self.qsel.encode_text_to_quantum_state(text, self.tokenizer)
        self.qsel.visualize_quantum_state(amplitudes, f"Medical: {title[:30]}")
        self.qsel.visualize_circuit_detailed(qc, title)       
        return qc, amplitudes    
    def run_quantum_backend(self, text: str, shots: int = 2048):
        print(f"\n[Quantum Execution] Running on {self.platform}...")       
        qc, amplitudes = self.qsel.encode_text_to_quantum_state(text, self.tokenizer)
        qc.measure_all()       
        try:
            from qiskit_aer import AerSimulator
            from qiskit import transpile           
            simulator = AerSimulator()
            transpiled_qc = transpile(qc, simulator, optimization_level=3)            
            print(f"  - Transpiled depth: {transpiled_qc.depth()}")
            print(f"  - Transpiled gates: {transpiled_qc.size()}")           
            job = simulator.run(transpiled_qc, shots=shots)
            result = job.result()
            counts = result.get_counts()           
            print(f"  ✓ Execution completed ({shots} shots)")
            sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:16]
            states = [s[0] for s in sorted_counts]
            values = [s[1] for s in sorted_counts]           
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(states)), values, color='mediumseagreen', alpha=0.7, edgecolor='black')
            plt.xlabel('Quantum State', fontsize=12, fontweight='bold')
            plt.ylabel('Measurement Counts', fontsize=12, fontweight='bold')
            plt.title(f'Quantum Measurement Results (Medical Text)\n{text[:50]}...', 
                     fontsize=13, fontweight='bold')
            plt.xticks(range(len(states)), states, rotation=45)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.show()          
            return counts            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return None
def main():
    experiment = QLPAMedicalExperiment(platform='simulator')
    medical_data = MedicalDatasetLoader.load_medical_questions(num_samples=10)    
    for i, sample in enumerate(medical_data[:3]): 
        print(f"SAMPLE {i+1}/{min(3, len(medical_data))}")
        qc, amps = experiment.run_complete_analysis(
            text=sample['text'],
            title=sample['title']
        )
    counts = experiment.run_quantum_backend(medical_data[0]['text'], shots=4096)
    print(f" Dataset: {medical_data[0]['source']}")
    print(f" Samples processed: {len(medical_data)}")
    print(f" Qubits: {experiment.qsel.n_qubits}")
    print(f" Circuit depth: Shown in visualizations")
    print(f" Elementary gates: Decomposed and counted")
    print(f" Platform: {experiment.platform}")
if __name__ == "__main__":
    main()
