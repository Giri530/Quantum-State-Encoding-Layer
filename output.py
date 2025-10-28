[QSEL] Initialized with:
  - Max sequence length: 8
  - Required qubits: 3 (log2(8))
  - Vocabulary size: 30000

[Backend] Using simulator for quantum execution

[Medical Dataset] Loading medical Q&A dataset...
  Sample 1: Q:A 23-year-old pregnant woman at 22 weeks gestation presents with bur...
  Sample 2: Q:A 3-month-old baby died suddenly at night while asleep. His mother n...
  Sample 3: Q:A mother brings her 3-week-old infant to the pediatrician's office b...
  Sample 4: Q:A pulmonary autopsy specimen from a 58-year-old woman who died of ac...
  Sample 5: Q:A 20-year-old woman presents with menorrhagia for the past several y...
  Sample 6: Q:A 40-year-old zookeeper presents to the emergency department complai...
  Sample 7: Q:A 25-year-old primigravida presents to her physician for a routine p...
  Sample 8: Q:A 3900-g (8.6-lb) male infant is delivered at 39 weeks' gestation vi...
  Sample 9: Q:A 62-year-old woman presents for a regular check-up. She complains o...
  Sample 10: Q:A 35-year-old male presents to his primary care physician with compl...
SAMPLE 1/3

================================================================================
[QUANTUM ENCODING ANALYSIS]
Title: Q:A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. She state
Text: Question: Q:A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination....
================================================================================


================================================================================
CIRCUIT ANALYSIS: 'Q:A 23-year-old pregnant woman at 22 weeks gestati...'
================================================================================

[Original Circuit]
  - Qubits: 3
  - Depth: 1
  - Operations: 1

[Decomposed Circuit - Elementary Gates]
  - Qubits: 3
  - Depth: 9
  - Elementary gates: 11

[Gate Breakdown]
  - UNITARY: 7
  - CX: 4

[High-Level Circuit Diagram]
     ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
q_0: ┤0                                                                                                                                 ├
     │                                                                                                                                  │
q_1: ┤1 State Preparation(0.022007,0.48687+0.48687j,0.22312j,-0.16224+0.16224j,-0.22312,-0.15977-0.15977j,-0.56717j,0.015715-0.015715j) ├
     │                                                                                                                                  │
q_2: ┤2                                                                                                                                 ├
     └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

[Decomposed Circuit - First 100 Gates]
     ┌─────────┐                ┌───┐┌─────────┐┌───┐┌─────────┐┌───┐┌─────────┐
q_0: ┤ Unitary ├────────────────┤ X ├┤ Unitary ├┤ X ├┤ Unitary ├┤ X ├┤ Unitary ├
     ├─────────┤┌───┐┌─────────┐└─┬─┘└─────────┘└─┬─┘└─────────┘└─┬─┘└─────────┘
q_1: ┤ Unitary ├┤ X ├┤ Unitary ├──■───────────────┼───────────────■─────────────
     ├─────────┤└─┬─┘└─────────┘                  │                             
q_2: ┤ Unitary ├──■───────────────────────────────■─────────────────────────────
     └─────────┘                                                                

SAMPLE 2/3

================================================================================
[QUANTUM ENCODING ANALYSIS]
Title: Q:A 3-month-old baby died suddenly at night while asleep. His mother noticed that he had died only a
Text: Question: Q:A 3-month-old baby died suddenly at night while asleep. His mother noticed that he had d...
================================================================================


================================================================================
CIRCUIT ANALYSIS: 'Q:A 3-month-old baby died suddenly at night while ...'
================================================================================

[Original Circuit]
  - Qubits: 3
  - Depth: 1
  - Operations: 1

[Decomposed Circuit - Elementary Gates]
  - Qubits: 3
  - Depth: 9
  - Elementary gates: 11

[Gate Breakdown]
  - UNITARY: 7
  - CX: 4

[High-Level Circuit Diagram]
     ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
q_0: ┤0                                                                                                                               ├
     │                                                                                                                                │
q_1: ┤1 State Preparation(0.025803,0.57084+0.57084j,0.2616j,-0.19022+0.19022j,-0.2616,-0.18733-0.18733j,-0.25982j,0.018426-0.018426j) ├
     │                                                                                                                                │
q_2: ┤2                                                                                                                               ├
     └────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

[Decomposed Circuit - First 100 Gates]
     ┌─────────┐                ┌───┐┌─────────┐┌───┐┌─────────┐┌───┐┌─────────┐
q_0: ┤ Unitary ├────────────────┤ X ├┤ Unitary ├┤ X ├┤ Unitary ├┤ X ├┤ Unitary ├
     ├─────────┤┌───┐┌─────────┐└─┬─┘└─────────┘└─┬─┘└─────────┘└─┬─┘└─────────┘
q_1: ┤ Unitary ├┤ X ├┤ Unitary ├──■───────────────┼───────────────■─────────────
     ├─────────┤└─┬─┘└─────────┘                  │                             
q_2: ┤ Unitary ├──■───────────────────────────────■─────────────────────────────
     └─────────┘                                                                

SAMPLE 3/3

================================================================================
[QUANTUM ENCODING ANALYSIS]
Title: Q:A mother brings her 3-week-old infant to the pediatrician's office because she is concerned about 
Text: Question: Q:A mother brings her 3-week-old infant to the pediatrician's office because she is concer...
================================================================================


================================================================================
CIRCUIT ANALYSIS: 'Q:A mother brings her 3-week-old infant to the ped...'
================================================================================

[Original Circuit]
  - Qubits: 3
  - Depth: 1
  - Operations: 1

[Decomposed Circuit - Elementary Gates]
  - Qubits: 3
  - Depth: 9
  - Elementary gates: 11

[Gate Breakdown]
  - UNITARY: 7
  - CX: 4

[High-Level Circuit Diagram]
     ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
q_0: ┤0                                                                                                                            ├
     │                                                                                                                             │
q_1: ┤1 State Preparation(0.02259,0.49976+0.49976j,0.22903j,-0.16654+0.16654j,-0.22903,-0.164-0.164j,-0.53411j,0.016132-0.016132j) ├
     │                                                                                                                             │
q_2: ┤2                                                                                                                            ├
     └─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

[Decomposed Circuit - First 100 Gates]
     ┌─────────┐                ┌───┐┌─────────┐┌───┐┌─────────┐┌───┐┌─────────┐
q_0: ┤ Unitary ├────────────────┤ X ├┤ Unitary ├┤ X ├┤ Unitary ├┤ X ├┤ Unitary ├
     ├─────────┤┌───┐┌─────────┐└─┬─┘└─────────┘└─┬─┘└─────────┘└─┬─┘└─────────┘
q_1: ┤ Unitary ├┤ X ├┤ Unitary ├──■───────────────┼───────────────■─────────────
     ├─────────┤└─┬─┘└─────────┘                  │                             
q_2: ┤ Unitary ├──■───────────────────────────────■─────────────────────────────
     └─────────┘                                                                


[Quantum Execution] Running on simulator...
  - Transpiled depth: 10
  - Transpiled gates: 14
  ✓ Execution completed (4096 shots)

 Dataset: Medical Q&A
 Samples processed: 10
 Qubits: 3
 Circuit depth: Shown in visualizations
 Elementary gates: Decomposed and counted
 Platform: simulator
