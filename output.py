
======================================================================
[VARIATIONAL QSEL] Initialization
======================================================================
  Max sequence length: 8
  Quantum qubits: 3
  State dimension: 8
[Quantum Semantic Encoder] Initialized
  Qubits: 3
  Learnable parameters: 9
  ✓ Custom tokenizer initialized
  ✓ Quantum encoder with 9 parameters
======================================================================


[DATASET] Loading medical Q&A from HuggingFace...
  [1/10] A 23-year-old pregnant woman at 22 weeks gestation presents ...
  [2/10] A 3-month-old baby died suddenly at night while asleep...
  [3/10] A mother brings her 3-week-old infant to the pediatrician's ...
  [4/10] A pulmonary autopsy specimen from a 58-year-old woman who di...
  [5/10] A 20-year-old woman presents with menorrhagia for the past s...
  [6/10] A 40-year-old zookeeper presents to the emergency department...
  [7/10] A 25-year-old primigravida presents to her physician for a r...
  [8/10] A 3900-g (8...
  [9/10] A 62-year-old woman presents for a regular check-up...
  [10/10] A 35-year-old male presents to his primary care physician wi...
  ✓ Loaded 10 medical Q&A samples


[Encoding] 'A 3-month-old baby died suddenly at night while as...'
  Tokens: [42, 6, 38, 39, 43, 44, 45, 0]
  Circuit depth: 9
  Total gates: 21
Text: A 3-month-old baby died suddenly at night while asleep...
Tokens: [42, 6, 38, 39, 43, 44, 45, 0]

[Circuit Properties]
  Qubits: 3
  Depth: 9
  Gates: 21

[Gate Distribution]
  RY: 9
  H: 3
  RZ: 3
  BARRIER: 3
  CX: 3
  P: 3

[Circuit Diagram]
     ┌───┐┌─────────────┐  ┌────────────┐┌─────────────┐ ░           ┌───┐ ░  ┌─────────────┐  ░   ┌──────┐ 
q_0: ┤ H ├┤ Ry(0.21801) ├──┤ Rz(1.5729) ├┤ Ry(0.61566) ├─░───■───────┤ X ├─░──┤ Ry(0.01939) ├──░───┤ P(0) ├─
     ├───┤├─────────────┴┐┌┴────────────┤├─────────────┤ ░ ┌─┴─┐     └─┬─┘ ░ ┌┴─────────────┴┐ ░ ┌─┴──────┴┐
q_1: ┤ H ├┤ Ry(0.027805) ├┤ Rz(0.80007) ├┤ Ry(0.07095) ├─░─┤ X ├──■────┼───░─┤ Ry(-0.018521) ├─░─┤ P(2π/3) ├
     ├───┤├─────────────┬┘└─┬──────────┬┘├─────────────┤ ░ └───┘┌─┴─┐  │   ░ ├───────────────┤ ░ ├─────────┤
q_2: ┤ H ├┤ Ry(0.63722) ├───┤ Rz(1.06) ├─┤ Ry(0.59638) ├─░──────┤ X ├──■───░─┤ Ry(-0.086157) ├─░─┤ P(4π/3) ├
     └───┘└─────────────┘   └──────────┘ └─────────────┘ ░      └───┘      ░ └───────────────┘ ░ └─────────┘
c: 3/═══════════════════════════════════════════════════════════════════════════════════════════════════════
                                                                                                            

[Decomposed Circuit]
  Elementary gates: 21
  Decomposed depth: 9

Text: A 3-month-old baby died suddenly at night while asleep...
Shots: 4096

[Transpilation]
  Optimized depth: 7
  Optimized gates: 14

[Execution Results]
  Total states measured: 8
  Top 5 states:
    1. |100 000⟩: 1098 (26.81%)
    2. |011 000⟩: 972 (23.73%)
    3. |010 000⟩: 793 (19.36%)
    4. |101 000⟩: 631 (15.41%)
    5. |001 000⟩: 220 (5.37%)

[Tokenization Check]
Text 1: A 23-year-old pregnant woman at 22 weeks gestation...
  Tokens: [46, 26, 3, 47, 19, 27, 22, 48]
  Words: ['23yearold', 'pregnant', 'woman', '22', 'weeks', 'gestation', 'presents', 'with']

Text 2: A 3-month-old baby died suddenly at night while as...
  Tokens: [42, 6, 38, 39, 43, 44, 45, 0]
  Words: ['3monthold', 'baby', 'died', 'suddenly', 'night', 'while', 'asleep']

Text 3: A mother brings her 3-week-old infant to the pedia...
  Tokens: [49, 40, 50, 51, 5, 52, 53, 54]
  Words: ['mother', 'brings', 'her', '3weekold', 'infant', 'pediatricians', 'office', 'because']

Similarity: 0.8947 - Different scenarios
  Text 1: A 23-year-old pregnant woman at 22 weeks gestation...
  Text 2: A 3-month-old baby died suddenly at night while as...

Similarity: 0.9190 - Different scenarios
  Text 1: A 23-year-old pregnant woman at 22 weeks gestation...
  Text 2: A mother brings her 3-week-old infant to the pedia...

Similarity: 0.9920 - Same text (should be ~1.0)
  Text 1: A 23-year-old pregnant woman at 22 weeks gestation...
  Text 2: A 23-year-old pregnant woman at 22 weeks gestation...


[TRAINING] Learning quantum parameters...
  Training pairs: 5
  Epochs: 15

  Epoch 1/15 - Loss: 0.152325 ✓ (best)
  Epoch 2/15 - Loss: 0.149195 ✓ (best)
  Epoch 3/15 - Loss: 0.155141 
  Epoch 4/15 - Loss: 0.171448 
  Epoch 5/15 - Loss: 0.171950 
  Epoch 6/15 - Loss: 0.177862 
  Epoch 7/15 - Loss: 0.177804 

  Early stopping: No improvement for 5 epochs

  ✓ Training complete! Best loss: 0.149195



[Sample 1]

[Encoding] 'A 23-year-old pregnant woman at 22 weeks gestation...'
  Tokens: [46, 26, 3, 47, 19, 27, 22, 48]
  Circuit depth: 9
  Total gates: 21

[Sample 2]

[Encoding] 'A 3-month-old baby died suddenly at night while as...'
  Tokens: [42, 6, 38, 39, 43, 44, 45, 0]
  Circuit depth: 9
  Total gates: 21

[Sample 3]

[Encoding] 'A mother brings her 3-week-old infant to the pedia...'
  Tokens: [49, 40, 50, 51, 5, 52, 53, 54]
  Circuit depth: 9
  Total gates: 21
