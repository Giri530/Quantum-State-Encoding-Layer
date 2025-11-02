Max Sequence Length: 8
Embedding Dimension: 64
QSEL initialized (3 qubits) - TRUE AMPLITUDE ENCODING
QSEN initialized - GRAPH STATE (threshold=0.3)
 Validator initialized
[Dataset] Loading wikitext...
README.md: 10.5kB [00:00, 31.0MB/s]
Loaded 22 samples from wikitext
[Tokenizer] Built vocabulary: 342 tokens

[Sample Texts]
  1.  Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場のヴァルキュリア3 ...
  2.  The game began development in 2010 , carrying over a large portion of...
  3.  It met with positive sales in Japan , and was praised by both Japanes...
  4.  As with previous Valkyira Chronicles games , Valkyria Chronicles III ...
  5.  The game 's battle system , the BliTZ system , is carried over direct...

--- Text 1 ---

[QLPA] Processing: ' Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場...'
  QSEL: 7 depth, 7 gates
  QSEN: 5 semantic edges
  Final: 18 depth, 25 gates

[Circuit Diagram]
     ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐ Encoded  Ent-1           ┌───┐ Ent-2           ┌───┐ QSEN-Start  Graph-Init ┌───┐ Graph-Entangle    ┌─────────┐   ┌────────────┐                    ┌────────────┐   ┌────────────┐ Graph-Complete  QSEN-End 
q_0: ┤0                                                                                                                                ├────░───────░─────■───────┤ X ├───░─────■───────┤ X ├─────░───────────░──────┤ H ├───────░─────────■─┤ Rz(π/2) ├─■─┤ Rz(1.1942) ├──────────────────■─┤ Rz(1.1155) ├─■─┤ Rz(1.0773) ├───────░────────────░─────
     │                                                                                                                                 │    ░       ░   ┌─┴─┐     └─┬─┘   ░   ┌─┴─┐     └─┬─┘     ░           ░      ├───┤       ░         │ ├─────────┤ │ └────────────┘   ┌────────────┐ │ └────────────┘ │ ├────────────┤       ░            ░     
q_1: ┤1 Initialize(-0.040647,-0.038811-0.038811j,0.48259j,-0.44014+0.44014j,-0.60936,0.010858+0.010858j,0.056592j,-0.010786+0.010786j) ├────░───────░───┤ X ├──■────┼─────░───┤ X ├──■────┼───────░───────────░──────┤ H ├───────░─────────■─┤ Rz(π/2) ├─┼────────────────■─┤ Rz(1.1763) ├─┼────────────────■─┤ Rz(1.0773) ├───────░────────────░─────
     │                                                                                                                                 │    ░       ░   └───┘┌─┴─┐  │     ░   └───┘┌─┴─┐  │       ░           ░      ├───┤       ░           └─────────┘ │ ┌────────────┐ │ ├────────────┤ │ ┌────────────┐   └────────────┘       ░            ░     
q_2: ┤2                                                                                                                                ├────░───────░────────┤ X ├──■─────░────────┤ X ├──■───────░───────────░──────┤ H ├───────░───────────────────────■─┤ Rz(1.1942) ├─■─┤ Rz(1.1763) ├─■─┤ Rz(1.1155) ├────────────────────────░────────────░─────
     └─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘    ░       ░        └───┘        ░        └───┘          ░           ░      └───┘       ░                         └────────────┘   └────────────┘   └────────────┘                        ░            ░     
Figure(1400x500)

--- Text 2 ---

[QLPA] Processing: ' The game began development in 2010 , carrying over a large ...'
  QSEL: 7 depth, 7 gates
  QSEN: 2 semantic edges
  Final: 10 depth, 13 gates

--- Text 3 ---

[QLPA] Processing: ' It met with positive sales in Japan , and was praised by bo...'
  QSEL: 7 depth, 7 gates
  QSEN: 2 semantic edges
  Final: 10 depth, 13 gates

--- Text 4 ---

[QLPA] Processing: ' As with previous Valkyira Chronicles games , Valkyria Chron...'
  QSEL: 7 depth, 7 gates
  QSEN: 3 semantic edges
  Final: 10 depth, 13 gates

--- Text 5 ---

[QLPA] Processing: ' The game 's battle system , the BliTZ system , is carried o...'
  QSEL: 7 depth, 7 gates
  QSEN: 7 semantic edges
  Final: 20 depth, 28 gates

[Validation] Testing entanglement for: ' Senjō no Valkyria 3 : Unrecorded Chronicles ( Jap...'
  Entanglement Entropy: 0.9403 bits
  Maximum Entropy: 1.0000 bits
  Entanglement Ratio: 94.03%
  Status:  ENTANGLED

[Validation] Testing similarity graph for: ' Senjō no Valkyria 3 : Unrecorded Chronicles ( Jap...'
  Non-padding tokens: 8
  Semantic edges: 5
  Avg similarity: 0.153
  Max similarity: 1.000
  Threshold: 0.300

[Validation] Measuring complexity scaling...
  n=  4: log(n)= 2, theoretical= 10, actual= 12, ratio=1.20x
  n=  8: log(n)= 3, theoretical= 15, actual= 50, ratio=3.33x
  n= 16: log(n)= 4, theoretical= 20, actual=188, ratio=9.40x
  n= 32: log(n)= 5, theoretical= 25, actual=770, ratio=30.80x
Average QSEN Edges: 3.8
Average Circuit Depth: 13.6
Average Gates: 18.4
