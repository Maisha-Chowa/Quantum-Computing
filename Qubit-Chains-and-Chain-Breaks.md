িভিন্ন ধরনের সলভার আছে‌। hybrid solver, advantage system 1, advantage system 2 ঐ API use করে দেখো তো তোমাদের প্রবলেম টা এগুলো তে সলভ করতে পারো কিনা

# Qubit Chains and Chain Breaks

This note answers:

1. What are qubit chain length and chain breaks?
2. Why do chain breaks occur in quantum annealing but not, under the same
   terminology, in gate-based quantum computing?

The phrase **gate annealing based** mixes two computing models. The usual
comparison is **quantum annealing** versus **gate-based quantum computing**.

## 1. Logical variables and physical qubits

A QUBO is written using **logical binary variables**. For example, a Max-Cut
variable (x_A) says which partition contains vertex A.

A D-Wave QPU contains **physical qubits** connected by a fixed hardware graph.
A physical coupler exists only between certain pairs. The QUBO graph and the
hardware graph are usually different, so Ocean performs **minor embedding**:

- a logical variable may map to one physical qubit when connectivity permits;
- a highly connected logical variable may map to several connected qubits;
- those qubits are coupled so that they prefer the same value;
- this connected group is called a **chain**.

Conceptually:

    Logical variable xA  ->  physical qubits {q12, q31, q44}

These physical qubits are not extra decision variables. Together, they are
intended to behave as one logical variable.

## 2. What is qubit chain length?

**Chain length is the number of physical qubits used to represent one logical
variable.**

Examples:

    xA -> {q12}            chain length = 1
    xB -> {q27, q31}       chain length = 2
    xC -> {q4, q9, q15}    chain length = 3

A length-one chain cannot internally break. A chain of length two or more can.
Variables in one problem can have different chain lengths.

Short chains are generally preferable because they use fewer physical qubits,
require fewer chain couplers, and offer fewer places for disagreement.
Nevertheless, chain length alone does not determine solution quality.

Useful embedding statistics are:

- maximum and average chain length;
- total physical qubits used;
- distribution of chain lengths;
- fraction of returned samples with broken chains.

## 3. What is chain strength?

The physical qubits in a chain are joined by strong agreement-promoting
couplings. Their magnitude is called **chain strength**.

In the Ising representation, a ferromagnetic chain coupling gives lower energy
when neighboring chain qubits have the same spin:

    q1 = +1, q2 = +1     agreement
    q1 = -1, q2 = -1     agreement
    q1 = +1, q2 = -1     disagreement

There is a trade-off:

- If chain strength is too weak, logical interactions can pull different
  portions of a chain toward different values.
- If it is excessively strong, the QPU may rescale the complete problem into
  its supported bias range. The meaningful logical coefficients then use less
  of the available analog precision.

The best value maintains agreement without overwhelming the logical problem.
Ocean can choose a default, but difficult embeddings often benefit from
experimentation.

## 4. What is a chain break?

A **chain break** occurs when physical qubits representing the same logical
variable do not all return the same value at the end of an anneal.

    xA -> {q12=1, q31=1, q44=1}    unbroken; decode xA as 1
    xA -> {q12=1, q31=0, q44=1}    broken; no unanimous value

Ocean must resolve the broken result before returning a logical sample.
Common approaches include:

- **majority vote:** select the value returned by most chain members;
- **energy minimization:** resolve broken variables while considering the
  logical BQM and neighboring variables;
- **discarding samples:** reject samples with unacceptable chain breaking.

These are post-processing methods. They do not undo the physical break.
Always recompute the logical energy and check the original constraints after
decoding.

### Chain-break fraction

For one sample, chain-break fraction is the fraction of logical chains that
were broken before unembedding. If one of four chains broke:

    chain_break_fraction = 1 / 4 = 0.25

It is not the fraction of incorrect physical qubits. It is also not proof that
the decoded answer is wrong. A broken sample can decode to an optimum, while an
unbroken sample can still be a nonoptimal excited state.

## 5. Why do chain breaks happen in quantum annealing?

### 5.1 Competing logical interactions

Different chain members connect to different neighboring variables. Those
external interactions can favor conflicting values in different parts of the
chain. If they compete strongly with the agreement coupling, the chain can
split.

### 5.2 Insufficient chain strength

A weak chain penalty does not make disagreement expensive enough. Increasing
the strength may help, subject to the coefficient-rescaling trade-off.

### 5.3 Long or poor embeddings

A longer chain contains more qubits and couplers that must remain consistent.
Long and badly balanced chains are often more vulnerable.

### 5.4 Analog control error and hardware noise

Biases and couplings are programmed physical analog quantities. Calibration
error, thermal effects, noise, and device imperfections can shift the realized
energy landscape away from the requested landscape.

### 5.5 Freeze-out and nonadiabatic behavior

Annealing takes finite time. Parts of the system can stop equilibrating before
reaching the desired ground state, or transitions can leave the system in an
excited state. A broken chain is one possible excited-state configuration.

This is why an annealer is read repeatedly: some reads may be optimal and
unbroken, while others may be excited or broken.

## 6. Why does ordinary gate-based mapping not have chain breaks?

The two models handle limited hardware connectivity differently.

### Quantum annealing

The optimization problem requests many logical interactions simultaneously.
Minor embedding represents one logical variable by a simultaneously coupled
group of physical qubits. Agreement inside the group is part of the embedded
energy function. Every member is measured at the end, so disagreement is
visible as a chain break.

    one logical variable
             |
             v
    several coupled physical qubits
             |
             v
    measure every member and check agreement

### Gate-based quantum computing

In the usual unencoded circuit model, a virtual circuit qubit maps to one
physical hardware qubit at a particular point in the circuit. When two
nonadjacent qubits must interact, the compiler inserts operations such as SWAP
gates to route the quantum states through the coupling graph.

    virtual qubit state
             |
             v
    one physical location at a time
             |
          SWAP gates
             |
             v
    a different location later in the circuit

There is no group of ferromagnetically coupled replicas required to return the
same final bit. Therefore **chain break** is not the applicable failure mode.

This does not mean gate-based computers avoid connectivity problems or errors.
They experience:

- imperfect one-qubit and two-qubit gates;
- decoherence;
- measurement error;
- crosstalk and leakage;
- extra noise and circuit depth from SWAP routing.

These can produce wrong measurement distributions, but they are not annealing
chain breaks.

## 7. Important exception: quantum error correction

Gate-based quantum error correction can encode one logical qubit across many
physical qubits. This superficially resembles a chain, but the mechanisms are
different:

- an annealing chain represents one Ising/QUBO variable using strong agreement
  couplings;
- an error-correcting code stores a quantum state in an entangled code space;
- syndrome measurements diagnose faults without simply measuring all data
  qubits and assigning a final binary variable;
- the relevant terms are physical error, syndrome event, leakage, and logical
  error, rather than QUBO chain break.

Repetition codes can use majority-like decoding, but that does not make them
minor-embedding chains.

## 8. Inspecting chains in the Max-Cut Ocean notebook

The QPU call in MaxCut-QUBO-DWave-Ocean.ipynb requests embedding information
with return_embedding set to True.

Inspect the selected embedding and its chain lengths:

    embedding = qpu_sampleset.info["embedding_context"]["embedding"]
    chain_lengths = {
        variable: len(physical_qubits)
        for variable, physical_qubits in embedding.items()
    }

    print("Embedding:", embedding)
    print("Chain lengths:", chain_lengths)
    print("Maximum chain length:", max(chain_lengths.values()))

Inspect the chain-break fractions returned by Ocean:

    if "chain_break_fraction" in qpu_sampleset.record.dtype.names:
        fractions = qpu_sampleset.record.chain_break_fraction
        print("Fractions:", fractions)
        print("Mean fraction:", fractions.mean())

When comparing experiments, record:

1. the embedding and chain lengths;
2. chain strength;
3. number of reads;
4. chain-break fraction distribution;
5. frequency of the lowest logical energy;
6. satisfaction of the original constraints.

Do not judge a run using chain-break fraction alone.

## 9. Ways to reduce chain breaks

Possible experiments include:

1. Find embeddings with shorter and more balanced chains.
2. Increase chain strength gradually while measuring both chain breaks and
   logical solution quality.
3. Avoid strength so large that rescaling loses useful coefficient precision.
4. Increase the number of reads.
5. Compare majority-vote and energy-minimization decoding.
6. Apply spin-reversal transforms when supported.
7. Inspect QPU calibration and embedding placement.
8. Scale the BQM so meaningful coefficients use the hardware range well.

The goal is not zero broken chains at any cost. The practical objective is a
high probability of correct, low-energy logical solutions.

## 10. Summary

| Concept | Quantum annealing | Gate-based computing |
|---|---|---|
| Limited-connectivity solution | Minor embedding and chains | Layout and gate routing, often using SWAPs |
| Representation | One logical binary variable may occupy several physical qubits simultaneously | One unencoded virtual qubit maps to one physical qubit at a time |
| Consistency mechanism | Chain couplers favor equal final values | Quantum gates transport and transform the state |
| Characteristic failure | Chain members return different values | Gate, routing, decoherence, leakage, or measurement errors |
| Term “chain break” | Yes | Normally no |
| Multi-qubit logical encoding | QUBO/Ising embedding | Possible through error correction, but it is a different construction |

In one sentence: **chain breaks occur in quantum annealing because minor
embedding deliberately represents one logical variable with multiple physical
qubits whose agreement is energetic rather than guaranteed; ordinary
gate-based routing does not create this representation, so its errors appear
in different forms.**

## References

- [D-Wave concepts: chains and chain breaks](https://docs.dwavequantum.com/en/latest/concepts/index.html)
- [D-Wave introduction to minor embedding](https://docs.dwavequantum.com/en/latest/quantum_research/embedding_intro.html)
- [D-Wave AND-gate embedding example](https://docs.dwavequantum.com/en/latest/quantum_research/example_and.html)
- [IBM Quantum transpiler stages and SWAP routing](https://docs.quantum.ibm.com/guides/transpiler-stages)
- [IBM Quantum hardware connectivity and mapping](https://docs.quantum.ibm.com/guides/represent-quantum-computers)
