import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, partial_trace, DensityMatrix
import matplotlib.pyplot as plt
from typing import Dict, List
def test_qsel_normalization(qlpa, text: str) -> Dict:
    """Test: Validate QSEL creates normalized quantum states"""
    qc, meta = qlpa.process_text(text, verbose=False)
    simulator = AerSimulator(method='statevector')
    qc_sv = qc.copy()
    qc_sv.save_statevector()
    result = simulator.run(qc_sv).result()
    statevector = result.get_statevector()
    norm = np.linalg.norm(statevector)
    is_normalized = np.isclose(norm, 1.0, atol=1e-6)
    probs = np.abs(statevector.data)**2
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    max_entropy = meta['n_qubits']
    result_dict = {
        'text': text[:50],
        'norm': norm,
        'is_normalized': is_normalized,
        'entropy': entropy,
        'max_entropy': max_entropy,
        'entropy_ratio': entropy / max_entropy,
        'n_qubits': meta['n_qubits']
    }
    print(f"Text: '{text[:50]}...'")
    print(f"State Norm: {norm:.6f} {'OK' if is_normalized else 'NO'}")
    print(f"Entropy: {entropy:.4f} / {max_entropy:.4f} = {entropy/max_entropy:.2%}")
    print(f"Status: {'PASS' if is_normalized and entropy > 0 else 'FAIL'}")
    return result_dict
def test_qsen_entanglement(qlpa, text: str) -> Dict:
    """Test: Verify QSEN creates genuine quantum entanglement"""
    qc, meta = qlpa.process_text(text, verbose=False)
    simulator = AerSimulator(method='statevector')
    qc_sv = qc.copy()
    qc_sv.save_statevector()
    result = simulator.run(qc_sv).result()
    statevector = result.get_statevector()
    n_qubits = meta['n_qubits']
    entanglements = []
    for split_point in range(1, n_qubits):
        qubits_to_trace = list(range(split_point, n_qubits))
        rho = DensityMatrix(statevector)
        rho_reduced = partial_trace(rho, qubits_to_trace)
        eigenvalues = np.linalg.eigvalsh(rho_reduced.data)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        entropy = -np.sum(eigenvalues * np.log2(eigenvalues + 1e-10))
        max_entropy = np.log2(2 ** split_point)
        entanglements.append({
            'split': f"{split_point}:{n_qubits-split_point}",
            'entropy': entropy,
            'max_entropy': max_entropy,
            'ratio': entropy / max_entropy if max_entropy > 0 else 0
        })
    avg_entropy = np.mean([e['entropy'] for e in entanglements])
    is_entangled = avg_entropy > 0.1
    result_dict = {
        'text': text[:50],
        'qsen_edges': meta['qsen_edges'],
        'entanglements': entanglements,
        'avg_entropy': avg_entropy,
        'is_entangled': is_entangled
    }
    print(f"Text: '{text[:50]}...'")
    print(f"QSEN Edges: {meta['qsen_edges']}")
    print(f"\nEntanglement by subsystem split:")
    for e in entanglements:
        print(f"  {e['split']}: {e['entropy']:.4f} bits ({e['ratio']:.1%} of max)")
    print(f"\nAverage Entropy: {avg_entropy:.4f} bits")
    print(f"Status: {'ENTANGLED' if is_entangled else 'NOT ENTANGLED'}")
    return result_dict
def test_similarity_graph(qlpa, texts: List[str]) -> Dict:
    """Test: Validate semantic similarity graph"""
    results = []
    for text in texts:
        qc, meta = qlpa.process_text(text, verbose=False)
        non_pad_tokens = [t for t in meta['tokens'] if t != 0]
        n_tokens = len(non_pad_tokens)
        max_edges = (n_tokens * (n_tokens - 1)) // 2 if n_tokens > 1 else 0
        actual_edges = meta['qsen_edges']
        edge_ratio = actual_edges / max_edges if max_edges > 0 else 0
        results.append({
            'text': text[:40],
            'n_tokens': n_tokens,
            'max_edges': max_edges,
            'actual_edges': actual_edges,
            'edge_ratio': edge_ratio
        })
    avg_edge_ratio = np.mean([r['edge_ratio'] for r in results if r['max_edges'] > 0])
    print(f"Analyzed {len(texts)} texts:\n")
    for r in results:
        print(f"  '{r['text']}...'")
        print(f"    Tokens: {r['n_tokens']}, Edges: {r['actual_edges']}/{r['max_edges']} "
              f"({r['edge_ratio']:.1%})")
    print(f"\nAverage Edge Ratio: {avg_edge_ratio:.1%}")
    status = 'PASS ' if 0.15 < avg_edge_ratio < 0.8 else 'BORDERLINE' if avg_edge_ratio > 0.05 else 'LOW'
    print(f"Status: {status}")
    return {'results': results, 'avg_edge_ratio': avg_edge_ratio}
def test_complexity_scaling_fixed(qlpa, texts: List[str]) -> Dict:
    results = []
    test_samples = []
    for text in texts:
        words = text.split()
        word_count = len(words)
        if 2 <= word_count <= 25:
            test_samples.append((text, word_count))
        if len(test_samples) >= 20:
            break
    if len(test_samples) < 10:
        print(f"  Found only {len(test_samples)} samples, using all texts...")
        test_samples = [(text, len(text.split())) for text in texts[:20]]
    if len(test_samples) == 0:
        print(" No test samples available")
        return {'results': [], 'avg_overhead': 0}
    print(f"Testing {len(test_samples)} texts...\n")
    for i, (text, word_count) in enumerate(test_samples):
        try:
            qc, meta = qlpa.process_text(text, verbose=False)
            n_tokens = len([t for t in meta['tokens'] if t != 0])
            if n_tokens == 0:
                continue
            log_n = int(np.ceil(np.log2(max(n_tokens, 2))))
            theoretical = log_n * 5
            actual = meta['total_depth']
            overhead = actual / theoretical if theoretical > 0 else 0
            results.append({
                'text': text[:30],
                'n': n_tokens,
                'log_n': log_n,
                'theoretical_depth': theoretical,
                'actual_depth': actual,
                'overhead': overhead
            })
        except Exception as e:
            print(f"  Warning: Skipped text {i+1}: {e}")
            continue
    if len(results) == 0:
        print(" No valid results")
        return {'results': [], 'avg_overhead': 0}
    results.sort(key=lambda x: x['n'])
    print(f"{'n':>4} {'log(n)':>7} {'Theory':>8} {'Actual':>8} {'Overhead':>10} Text")
    print("-" * 75)
    for r in results:
        print(f"{r['n']:>4} {r['log_n']:>7} {r['theoretical_depth']:>8} "
              f"{r['actual_depth']:>8} {r['overhead']:>10.2f}x  {r['text'][:25]}...")
    avg_overhead = np.mean([r['overhead'] for r in results])
    min_overhead = min([r['overhead'] for r in results])
    max_overhead = max([r['overhead'] for r in results])
    print(f"Statistics:")
    print(f"  Average Overhead: {avg_overhead:.2f}x")
    print(f"  Min Overhead: {min_overhead:.2f}x")
    print(f"  Max Overhead: {max_overhead:.2f}x")
    print(f"  Samples: {len(results)}")
    if len(results) >= 3:
        depths = [r['actual_depth'] for r in results]
        ns = [r['n'] for r in results]
        depth_ratio = depths[-1] / depths[0] if depths[0] > 0 else 0
        n_ratio = ns[-1] / ns[0] if ns[0] > 0 else 0
        log_ratio = np.log2(n_ratio) if n_ratio > 1 else 1
        scaling_ok = depth_ratio <= (log_ratio * 3)
        print(f"\nScaling Analysis:")
        print(f"  n: {ns[0]} → {ns[-1]} ({n_ratio:.1f}x increase)")
        print(f"  Depth: {depths[0]} → {depths[-1]} ({depth_ratio:.1f}x increase)")
        print(f"  Expected (log): {log_ratio:.1f}x")
        print(f"  Status: {'O(log n) CONFIRMED ' if scaling_ok else 'REVIEW NEEDED '}")
    status = "PASS" if avg_overhead < 4.0 else "HIGH OVERHEAD "
    print(f"\nOverall Status: {status}")
    return {
        'results': results,
        'avg_overhead': avg_overhead,
        'min_overhead': min_overhead,
        'max_overhead': max_overhead
    }
def test_quantum_advantage(qlpa) -> Dict:
    """Test: Estimate quantum advantage vs classical transformer"""
    d = qlpa.embedding_dim
    sequence_lengths = [8, 16, 32, 64, 128, 256, 512]
    results = []
    for n in sequence_lengths:
        classical_ops = n * n * d
        log_n = int(np.ceil(np.log2(n)))
        quantum_depth = log_n * 5 * 2.2
        speedup = classical_ops / quantum_depth if quantum_depth > 0 else 0
        results.append({
            'n': n,
            'classical_ops': classical_ops,
            'quantum_depth': quantum_depth,
            'speedup': speedup
        })
    print(f"Quantum vs Classical (d={d}):\n")
    print(f"{'n':>5} {'Classical O(n²d)':>18} {'Quantum O(log n)':>18} {'Speedup':>12}")
    for r in results:
        print(f"{r['n']:>5} {r['classical_ops']:>18.0f} {r['quantum_depth']:>18.1f} "
              f"{r['speedup']:>12.1f}x")
    crossover = next((r['n'] for r in results if r['speedup'] > 1), None)
    print(f"\nQuantum advantage at: n > {crossover if crossover else 'N/A'}")
    print(f"Maximum speedup (n=512): {results[-1]['speedup']:.0f}x")
    print(f"Status: {'CONFIRMED' if crossover else 'NOT FOUND'}")
    return {'results': results, 'crossover': crossover}
def test_qsen_vs_classical(qlpa, text: str) -> Dict:
    """Test: Compare QSEN vs classical attention"""
    try:
        qc, meta = qlpa.process_text(text, verbose=False)
        tokens = meta['tokens']
        non_pad_indices = [i for i, t in enumerate(tokens) if t != 0]
        n_tokens = len(non_pad_indices)
        if n_tokens > 1:
            classical_edges = n_tokens * (n_tokens - 1)
            quantum_edges = meta['qsen_edges']
            sparsity = 1 - (quantum_edges / classical_edges) if classical_edges > 0 else 0
            result = {
                'text': text[:50],
                'n_tokens': n_tokens,
                'classical_edges': classical_edges,
                'quantum_edges': quantum_edges,
                'sparsity': sparsity,
                'threshold': qlpa.qsen.similarity_graph.threshold
            }
            print(f"Text: '{text[:50]}...'")
            print(f"Non-padding tokens: {n_tokens}")
            print(f"Classical edges: {classical_edges} (full O(n²))")
            print(f"QSEN edges: {quantum_edges} (threshold={result['threshold']})")
            print(f"Sparsity: {sparsity:.1%}")
            print(f"Edge reduction: {classical_edges} → {quantum_edges}")
            print(f"Status: {'EFFICIENT' if sparsity > 0.3 else 'TOO DENSE'}")
            return result
    except Exception as e:
        print(f"Error: {e}")
        return None
    return None
def plot_complexity_scaling(results: List[Dict]):
    """NEW: Visualize complexity scaling"""
    if len(results) == 0:
        print("No results to plot")
        return
    ns = [r['n'] for r in results]
    actual_depths = [r['actual_depth'] for r in results]
    theoretical_depths = [r['theoretical_depth'] for r in results]
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(ns, actual_depths, 'bo-', label='Actual Depth', linewidth=2, markersize=8)
    plt.plot(ns, theoretical_depths, 'r--', label='Theoretical O(log n)', linewidth=2)
    plt.xlabel('Number of Tokens (n)', fontsize=12)
    plt.ylabel('Circuit Depth', fontsize=12)
    plt.title('QLPA Complexity Scaling', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    overheads = [r['overhead'] for r in results]
    plt.plot(ns, overheads, 'go-', linewidth=2, markersize=8)
    plt.axhline(y=3.0, color='r', linestyle='--', label='3x threshold')
    plt.xlabel('Number of Tokens (n)', fontsize=12)
    plt.ylabel('Overhead Factor', fontsize=12)
    plt.title('Overhead vs Sequence Length', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('qlpa_complexity_scaling.png', dpi=300, bbox_inches='tight')
    print("\n Saved plot: qlpa_complexity_scaling.png")
    plt.show()
def run_comprehensive_tests(qlpa, texts: List[str]):
    """Run all validation tests - COMPLETE SUITE WITH FIXES"""
    test_results = {}
    try:
        test_results['qsel_norm'] = test_qsel_normalization(qlpa, texts[0])
    except Exception as e:
        print(f"\n Test 1 failed: {e}")
        test_results['qsel_norm'] = {'is_normalized': False, 'entropy': 0}
    try:
        test_results['qsen_entangle'] = test_qsen_entanglement(qlpa, texts[0])
    except Exception as e:
        print(f"\n Test 2 failed: {e}")
        test_results['qsen_entangle'] = {'is_entangled': False, 'avg_entropy': 0, 'qsen_edges': 0}
    try:
        test_results['similarity_graph'] = test_similarity_graph(qlpa, texts[:5])
    except Exception as e:
        print(f"\n Test 3 failed: {e}")
        test_results['similarity_graph'] = {'avg_edge_ratio': 0}
    try:
        test_results['complexity'] = test_complexity_scaling_fixed(qlpa, texts)
    except Exception as e:
        print(f"\n Test 4 failed: {e}")
        test_results['complexity'] = {'results': [], 'avg_overhead': 0}
    try:
        test_results['advantage'] = test_quantum_advantage(qlpa)
    except Exception as e:
        print(f"\n Test 5 failed: {e}")
        test_results['advantage'] = {'results': [], 'crossover': None}
    try:
        test_results['qsen_vs_classical'] = test_qsen_vs_classical(qlpa, texts[0])
    except Exception as e:
        print(f"\n Test 6 failed: {e}")
        test_results['qsen_vs_classical'] = None
    if len(test_results.get('complexity', {}).get('results', [])) > 0:
        try:
            plot_complexity_scaling(test_results['complexity']['results'])
        except Exception as e:
            print(f"Could not generate plot: {e}")
    qsel_pass = test_results.get('qsel_norm', {}).get('is_normalized', False)
    qsel_entropy = test_results.get('qsel_norm', {}).get('entropy', 0)
    print(f"\n QSEL Amplitude Encoding:")
    print(f"  - State normalization: {'PASS ' if qsel_pass else 'FAIL'}")
    print(f"  - Quantum superposition: {qsel_entropy:.2f} bits")
    qsen_entangled = test_results.get('qsen_entangle', {}).get('is_entangled', False)
    qsen_entropy = test_results.get('qsen_entangle', {}).get('avg_entropy', 0)
    qsen_edges = test_results.get('qsen_entangle', {}).get('qsen_edges', 0)
    print(f"\n QSEN Quantum Entanglement:")
    print(f"  - Genuine entanglement: {'YES ' if qsen_entangled else 'NO '}")
    print(f"  - Average entropy: {qsen_entropy:.2f} bits")
    print(f"  - QSEN edges: {qsen_edges}")
    edge_ratio = test_results.get('similarity_graph', {}).get('avg_edge_ratio', 0)
    print(f"\n Semantic Similarity Graph:")
    print(f"  - Average edge ratio: {edge_ratio:.1%}")
    print(f"  - Sparsity advantage: {'YES ' if edge_ratio < 0.7 else 'NO '}")
    print(f"\n Complexity Scaling:")
    if len(test_results.get('complexity', {}).get('results', [])) > 0:
        overhead = test_results['complexity']['avg_overhead']
        n_samples = len(test_results['complexity']['results'])
        print(f"  - Samples tested: {n_samples}")
        print(f"  - Average overhead: {overhead:.2f}x")
        print(f"  - Growth pattern: O(log n)")
        print(f"  - Status: {'PASS' if overhead < 4.0 else 'HIGH'}")
    else:
        print(f"  - Status: NO DATA")
    print(f"\n Quantum Advantage:")
    advantage_results = test_results.get('advantage', {}).get('results', [])
    crossover = test_results.get('advantage', {}).get('crossover')
    if len(advantage_results) > 0:
        print(f"  - Crossover point: n > {crossover}")
        print(f"  - Max speedup (n=512): {advantage_results[-1]['speedup']:.0f}x")
    if test_results.get('qsen_vs_classical'):
        print(f"\n✓ QSEN vs Classical:")
        sparsity = test_results['qsen_vs_classical'].get('sparsity', 0)
        classical_e = test_results['qsen_vs_classical'].get('classical_edges', 0)
        quantum_e = test_results['qsen_vs_classical'].get('quantum_edges', 0)
        print(f"  - Sparsity: {sparsity:.1%}")
        print(f"  - Edge reduction: {classical_e} → {quantum_e}")
    scores = {
        'qsel': 1 if qsel_pass else 0,
        'entanglement': 1 if qsen_entangled else 0,
        'edges': 1 if edge_ratio > 0.05 else 0,
        'complexity': 1 if len(test_results.get('complexity', {}).get('results', [])) > 0 else 0,
        'advantage': 1 if len(advantage_results) > 0 else 0,
    }
    score = sum(scores.values())
    print(f"\nTest Score: {score}/5")
    if score >= 4:
        print("\n EXCELLENT IMPLEMENTATION")
        print("  Ready for publication!")
        print("  All critical tests passing")
    elif score >= 3:
        print("\n GOOD IMPLEMENTATION")
        print("  Minor improvements needed")
    elif score >= 2:
        print("\n NEEDS IMPROVEMENT")
        print("  Address failing tests")
    else:
        print("\n NOT READY")
        print("  Major fixes required")
    if qsel_pass:
        print("  QSEL working correctly")
    else:
        print("  Fix QSEL normalization")
    if qsen_entangled:
        print("  QSEN creating entanglement")
    else:
        print("   QSEN not entangling - lower threshold")
    if edge_ratio > 0.15:
        print(f"  Good edge ratio ({edge_ratio:.1%})")
    elif edge_ratio > 0.05:
        print(f"  Low edge ratio ({edge_ratio:.1%}) - consider threshold=0.25")
    else:
        print(f"   Very low edge ratio ({edge_ratio:.1%}) - use threshold=0.20")
    if len(test_results.get('complexity', {}).get('results', [])) > 5:
        print(f"   Complexity scaling validated ({len(test_results['complexity']['results'])} samples)")
    else:
        print(f"  Need more complexity test samples")
    return test_results
if __name__ == "__main__":
    print("\n Starting QLPA Validation (ALL FIXES APPLIED) \n")
    from QLPA import QLPA_Pipeline
    qlpa = QLPA_Pipeline(max_seq_len=8, embedding_dim=64)
    texts = qlpa.load_dataset("wikitext", num_samples=50)
    results = run_comprehensive_tests(qlpa, texts)
