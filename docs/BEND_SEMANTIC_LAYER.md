# Bend 2 as a Semantic and Proof Layer Above the AOI Translator

## Thesis

Bend 2 should **not replace Tinygrad as this project's primary execution backend**.

Tinygrad and Bend are solving different problems.

Tinygrad is primarily a compact tensor compiler/runtime whose great strength is that a small canonical operation set can be lowered onto a wide and growing range of hardware. Bend 2 is a dependently typed, affine programming language that combines executable programs, parallel CPU/GPU execution, and machine-checkable laws/proofs.

For this repository, the most promising relationship is therefore **synergistic rather than competitive**:

```text
                    CANONICAL AOI OBJECT
                 "what this thing means"
                           |
        +------------------+------------------+
        |                  |                  |
   mathematical       semantic laws       graph/type
   definition          / contracts          contract
        |                  |                  |
        +------------------+------------------+
                           |
                  canonical tg-aoi IR
                           |
        +------------------+------------------+------------------+
        |                  |                  |                  |
     Tinygrad            C++              CUDA               Bend
   preferred exec    reference impl    native tuned impl   proof/spec impl
        |
   hardware backends
 NV / AMD / QCOM / Metal / OpenCL / CPU / WebGPU / future ASICs
```

The important object is not any one implementation.

The important object is **SOFTMAX**, **RMSNORM**, **FLASH_ATTENTION**, **TRANSFORMER_BLOCK**, **ADAM_STEP**, or some future discovered AOI as a canonical computational idea.

Implementations are realizations of that object.

Bend may eventually let us attach a stronger statement to those objects:

> This implementation is not merely named SOFTMAX. It satisfies the laws that define what SOFTMAX is.

That is a powerful addition to the translator, but it should sit **above or beside Tinygrad**, not displace it.

---

## Why Tinygrad remains the preferred execution substrate

Tinygrad is unusually aligned with the central goal of this repository: reduce neural architectures into a small, inspectable language and lower that language onto real hardware.

Its current runtime surface already includes NVIDIA, AMD, Qualcomm, Metal, OpenCL, CPU/LLVM, WebGPU, and multiple lower-level interfaces. Tinygrad's documentation also explicitly treats new accelerators as a relatively small backend problem: an accelerator needs to support only a small set of low-level operations.

That matters enormously.

The future of AI compute will not be only CUDA and Metal. It already includes AMD GPUs, Qualcomm devices, integrated accelerators, WebGPU-class environments, and increasingly specialized ASICs. A research language intended to survive changes in hardware should avoid binding its semantic model to one vendor execution model.

Tinygrad's architecture is attractive precisely because it attempts to make:

```text
model
  -> tensor operations
  -> UOps / canonical low-level computation
  -> renderer/runtime
  -> hardware
```

small enough to understand end-to-end.

That makes Tinygrad a natural execution target for tg-aoi.

### Strategic consequence

The project should continue to treat:

```text
canonical AOI graph -> Tinygrad
```

as the **first-class execution path**.

C++ and CUDA implementations are still highly valuable, especially as:

- readable reference implementations,
- high-performance baselines,
- comparison points against compiler-generated output,
- sources from which the translator can learn structural equivalence,
- targets for agent-generated optimization experiments.

But Tinygrad has a broader architectural role: it can potentially carry one canonical graph across many hardware families.

That is difficult to replace.

---

## What Bend 2 is trying to do instead

Bend 2 combines several ideas that normally live in different systems:

1. executable functional programs,
2. dependent types,
3. propositions and proofs expressed in the language,
4. explicit parallel computation,
5. compilation to CPU and GPU targets.

Its stated execution targets currently include C/CPU, CUDA, Metal, and JavaScript.

Its more unusual contribution is **law-driven development**.

A project can define laws describing behavior that must remain true, then require a proof before accepting an implementation change.

Conceptually:

```text
LAW:
  for every valid input x,
  implementation(x) satisfies property P

IMPLEMENTATION:
  arbitrary code written or rewritten by an agent

PROOF:
  machine-checked evidence that the implementation satisfies the law
```

This matters to tg-aoi because our long-term system is explicitly intended to allow agents to:

- import architectures,
- decompose them,
- rewrite them,
- mutate them,
- optimize them,
- discover repeated structures,
- promote new AOIs,
- and search architecture space automatically.

The more freedom an agent has to modify the implementation, the more valuable it becomes to have **invariants outside the implementation that the agent is not free to redefine**.

That is the niche Bend may fill.

---

# Tinygrad and Bend are not really competitors

A useful comparison is:

| Concern | Tinygrad | Bend 2 |
|---|---|---|
| Primary aim | tensor compiler/runtime | verified general programming language |
| Neural-network execution | central | possible, but not its main specialization |
| Canonical tensor lowering | central strength | not the core abstraction |
| Hardware breadth today | broad | narrower |
| NVIDIA GPU | yes | yes |
| AMD GPU | yes | not currently a native target |
| Apple Metal | yes | yes |
| Qualcomm | yes | not currently a native target |
| OpenCL | yes | not currently a native target |
| WebGPU | yes | not currently a native target |
| CPU | yes | yes |
| Future accelerator extensibility | core architectural concern | possible, but not current emphasis |
| Dependent types | no | yes |
| Machine-checked proofs | no | yes |
| Law/specification layer | external to TG | native concept |
| Best role in tg-aoi | execution/lowering substrate | semantic contract / proof layer |

The question is therefore not:

> Should tg-aoi use Tinygrad or Bend?

The better question is:

> At which layer is each language strongest?

The likely answer is:

```text
Bend:     WHAT MUST REMAIN TRUE?
tg-aoi:   WHAT COMPUTATION IS THIS?
Tinygrad: HOW DO WE EXECUTE IT PORTABLY?
CUDA/C++: HOW ELSE CAN IT BE REALIZED OR OPTIMIZED?
```

---

# The canonical object should sit above every implementation

This repository already treats a name as compression rather than opacity.

That idea should be pushed further.

Consider `SOFTMAX`.

We should not ultimately think of SOFTMAX as:

```text
the Tinygrad softmax function
```

or:

```text
the CUDA softmax kernel
```

Instead, SOFTMAX is a **canonical computational object**.

It may have:

```text
specs/SOFTMAX/
    definition.md
    equations.md
    graph.json
    tinygrad.py
    reference.cpp
    cuda.cu
    bend/
        SOFTMAX.bend
        LAWS.bend
        PROOF.bend
    tests/
    benchmarks/
```

The exact directory structure is not yet important. The separation of concerns is.

The object contains several different kinds of truth:

### Mathematical truth

For vector (x),

```text
softmax(x)_i = exp(x_i) / sum_j exp(x_j)
```

### Structural truth

The operation:

- consumes a tensor,
- preserves all non-reduced dimensions,
- normalizes along a declared axis,
- returns the expected dtype/shape contract.

### Numerical truth

Within a declared dtype and tolerance, an implementation agrees with an accepted reference implementation.

### Behavioral law

Examples might include:

- outputs are nonnegative,
- normalized outputs sum approximately to one,
- adding a scalar constant to every element does not change the mathematical result,
- decomposition and recomposition preserve the declared semantics.

### Implementations

The same object may then be realized through:

- Tinygrad,
- C++,
- CUDA,
- PyTorch,
- JAX,
- Triton,
- Bend,
- or future accelerator-specific code.

This turns the Rosetta Stone from a collection of translations into a **semantic dictionary of computation**.

---

# Why this matters for the translator

The translator's job is not merely syntax conversion.

A weak translator says:

```text
this PyTorch syntax maps to this Tinygrad syntax
```

A stronger translator says:

```text
this source subgraph is an instance of canonical object SOFTMAX@v1
```

and then:

```text
SOFTMAX@v1 can be lowered into:
    Tinygrad implementation A
    CUDA implementation B
    C++ implementation C
    Bend implementation D
```

That distinction is foundational.

Once the middle object has stable semantics, translation can become:

```text
SOURCE
   |
recognize
   v
CANONICAL OBJECT / AOI GRAPH
   |
validate meaning
   |
   +------> Tinygrad
   +------> C++
   +------> CUDA
   +------> Bend
   +------> future target
```

The canonical representation becomes the pivot.

This is much more durable than pairwise translators.

Without the canonical layer, supporting (N) frameworks tends toward an (N^2) translation problem.

With a canonical layer:

```text
framework -> canonical IR -> target
```

each ecosystem only needs to understand the common language.

---

# Bend's strongest future role: definitions that survive agent mutation

The research loop envisioned for this repository eventually becomes something like:

```text
IMPORT
  ->
DECOMPOSE
  ->
RECOGNIZE AOIs
  ->
MUTATE
  ->
LOWER
  ->
TRAIN / EXECUTE
  ->
BENCHMARK
  ->
COMPARE
  ->
KEEP / REJECT
  ->
LEARN
  ->
REPEAT
```

Performance metrics can tell us whether a mutation is faster or smaller.

Tests can tell us whether sampled behavior still matches expected outputs.

Neither necessarily tells us that the transformation preserved the thing we meant.

A semantic law can become another gate:

```text
candidate mutation
      |
      +--> type/shape validation
      |
      +--> equivalence/property tests
      |
      +--> Bend proof obligations        [future]
      |
      +--> benchmark
      |
      +--> training/task evaluation
      |
      v
    accept?
```

That is especially attractive once agents themselves begin proposing novel decompositions and optimizations.

The optimizer may be free to search aggressively.

The specification should not move with it.

---

# Proof does not replace numerical validation

There is an important limitation.

Bend currently treats F32 as axiomatic. Floating-point numerical behavior is therefore not something we should assume can simply be formally proven away.

For neural-network work, correctness is often inseparable from:

- floating-point rounding,
- accumulation order,
- approximation choice,
- reduced precision,
- quantization,
- stochastic training behavior,
- hardware-specific kernels.

Therefore the project should **not** imagine a future where Bend proofs replace equivalence tests and benchmarks.

The correct validation stack is likely cumulative:

```text
1. graph/type validity
2. shape/state invariants
3. symbolic laws where formally expressible
4. numerical equivalence within declared tolerance
5. gradient equivalence where required
6. task-level behavioral validation
7. performance measurement
```

Formal proof is another instrument on the bench.

It is not the whole bench.

---

# C++ and CUDA remain valuable

Even if Tinygrad is the preferred executable backend, C++ and CUDA should remain first-class members of the Rosetta dictionary where practical.

They serve a different purpose.

### C++

C++ gives us:

- an implementation close to conventional systems code,
- a relatively architecture-neutral native reference,
- something compiler engineers and model-runtime developers can inspect,
- a bridge toward CPU and custom-runtime implementations.

### CUDA

CUDA gives us:

- direct exposure to the dominant NVIDIA GPU programming model,
- hand-optimized kernel examples,
- a performance ceiling against which generated code can be compared,
- access to a huge existing corpus of optimized neural-network implementations.

### Tinygrad

Tinygrad gives us:

- the compact compiler substrate,
- graph lowering,
- kernel generation,
- hardware portability,
- a codebase small enough for an agent and human to reason about together.

### Bend

Bend may give us:

- machine-checkable semantic contracts,
- proof-carrying implementations,
- a way to distinguish an AOI's meaning from any particular realization.

These are complementary assets.

---

# The first model should be a quarry, not a monument

We do not need to understand every primitive in advance.

The repository can learn its vocabulary by **mining real models**.

The most productive path is likely:

```text
existing working model
      |
      v
import graph
      |
      v
expand into smaller operations
      |
      v
identify recurring structures
      |
      v
name stable structures as AOIs
      |
      v
attach definitions + implementations + tests
      |
      v
repeat across more models
```

The first models are therefore not sacred architectures.

They are **quarries** from which the computational vocabulary is extracted.

Tinygrad's own model implementations and examples are particularly valuable because they provide code already close to the execution substrate we intend to use.

A model such as a Qwen-family Transformer can be decomposed through the visualizer into recognizable layers:

```text
MODEL
  |
  +-- token embedding
  |
  +-- repeated transformer block
       |
       +-- normalization
       +-- attention
       |    |
       |    +-- Q/K/V projections
       |    +-- positional operation
       |    +-- attention score
       |    +-- mask
       |    +-- softmax
       |    +-- weighted value aggregation
       |    +-- output projection
       |
       +-- MLP / gated MLP
       +-- residual paths
  |
  +-- final norm
  +-- output projection
```

Each recognized piece becomes a candidate reusable object in the AOI language.

The visualizer then becomes more than a diagram viewer.

It becomes the place where a human and agent can move continuously between:

```text
whole model
    <-> architecture block
    <-> AOI
    <-> primitive graph
    <-> implementation
    <-> hardware kernel
```

That is the real educational and research value of the project.

---

# Implication for the visualizer

Every canonical object should eventually be inspectable as a card/block with several views:

```text
+------------------------------------------------+
| FLASH_ATTENTION                                |
|------------------------------------------------|
| Canonical graph                                |
| Mathematical definition                       |
| Input/output tensor contract                   |
| Child AOIs                                     |
|------------------------------------------------|
| Implementations                                |
|   [Tinygrad] [C++] [CUDA] [Bend] [...]         |
|------------------------------------------------|
| Validation                                     |
|   equivalence tests                            |
|   laws / proof status                          |
|------------------------------------------------|
| Metrics                                        |
|   latency | VRAM | FLOPs | training metric    |
|------------------------------------------------|
| Provenance                                     |
|   source model / paper / commit / experiment   |
+------------------------------------------------+
```

An agent should be able to open the block, modify one realization or its decomposition, execute the benchmark suite, and compare the result against the unchanged semantic object.

This gives the mutation loop something solid to push against.

---

# Recommended architecture

## Layer 1 — Canonical semantics

Repository-owned definitions of computational objects.

Examples:

```text
SOFTMAX
RMSNORM
ROPE
LINEAR
SWIGLU
ATTENTION
FLASH_ATTENTION
ADAM_STEP
TRANSFORMER_BLOCK
VISION_TRANSFORMER_BLOCK
DETR_DECODER
...
```

These definitions should be independent of Tinygrad, CUDA, PyTorch, or Bend.

---

## Layer 2 — Canonical graph / tg-aoi IR

The explicit reducible representation.

Every high-level AOI must be expandable until it reaches the supported primitive substrate.

This remains the heart of the project.

---

## Layer 3 — Rosetta implementations

Multiple realizations may exist for the same canonical object:

```text
Tinygrad
PyTorch
JAX
C++
CUDA
Bend
Triton
ONNX decomposition
...
```

No implementation owns the meaning of the object.

---

## Layer 4 — Execution

Tinygrad should remain the preferred general execution path because its compiler/runtime architecture already spans a wider hardware set and is intentionally extensible.

Native C++/CUDA paths can be used where they add research value.

---

## Layer 5 — Verification

Today:

- tensor/type checking,
- shape/state contracts,
- graph equivalence,
- numerical equivalence,
- unit/property tests,
- benchmark validation.

Later:

- Bend laws,
- Bend proofs,
- formalized invariants on graph rewrites,
- proof obligations for selected canonical objects.

---

## Layer 6 — Search and recursive improvement

Agents propose modifications to:

- decomposition,
- architecture,
- implementation,
- optimizer,
- kernel strategy,
- training recipe.

The unchanged specification and evaluation harness judge the proposal.

That separation is essential.

```text
SPECIFICATION        <- conservative
      |
SEARCH SPACE         <- aggressive
      |
EXECUTION
      |
MEASUREMENT
      |
EVIDENCE
```

---

# Adoption plan

Bend should be adopted experimentally and **late enough that it strengthens the architecture rather than distracting from it**.

## Phase A — now

Do not make Bend a dependency.

Focus on:

- recursive AOIs,
- canonical graph stability,
- Tinygrad lowering,
- importer breadth,
- numerical equivalence,
- model visualization,
- decomposition of real models,
- C++/CUDA/Tinygrad Rosetta entries where useful.

The project first needs stable objects worth specifying.

## Phase B — semantic contracts

For each mature AOI, add explicit machine-readable semantic metadata independent of Bend:

- input types,
- output types,
- shape relationships,
- state behavior,
- mathematical identity,
- invariants,
- tolerances,
- decomposition rules.

This creates the specification layer whether Bend survives or not.

## Phase C — Bend experiment

Choose **one small, mature AOI**.

Good candidates include:

- a shape transformation,
- a pure integer/indexing primitive,
- a masking rule,
- a simplified normalization object,
- another operation whose important law is formally expressible without depending heavily on F32 semantics.

Implement:

```text
canonical AOI
Tinygrad implementation
reference implementation
Bend definition
Bend law
Bend proof
```

Then measure whether the proof adds practical value to the agent-modification loop.

## Phase D — proof-aware mutation

If Phase C is useful, allow an agent to mutate the Bend or canonical implementation while keeping the law fixed.

The experiment succeeds if the law catches invalid transformations that ordinary local tests could plausibly miss.

## Phase E — selective proof library

Only then promote Bend into a supported optional subsystem.

Formalization should concentrate on high-value invariants, not become ceremony applied to every tensor operation.

---

# What we should avoid

## Do not rewrite the project in Bend

That would confuse execution architecture with specification architecture.

## Do not make Bend the canonical IR

The canonical IR should remain framework-neutral and represent neural computation directly.

## Do not make Tinygrad define semantic truth

Tinygrad is an implementation/execution system. The canonical object must be able to outlive it.

## Do not formalize everything immediately

A proof system can consume enormous engineering effort if the specification target is still moving.

First stabilize the computational vocabulary.

## Do not treat passing a proof as total neural-network correctness

Floating-point semantics, training behavior, dataset dependence, stochasticity, and hardware behavior still require empirical validation.

---

# Strategic position

The strongest architecture is not:

```text
Bend instead of Tinygrad
```

and not:

```text
Tinygrad instead of Bend
```

It is:

```text
                 HUMAN / AGENT INTENT
                         |
                         v
                CANONICAL AOI LANGUAGE
                         |
              semantics / contracts
                         |
             +-----------+-----------+
             |                       |
         Bend laws               tests/math
        [optional]                   |
             +-----------+-----------+
                         |
                   canonical IR
                         |
          +--------------+--------------+
          |              |              |
       Tinygrad          C++           CUDA
    primary portable   reference    tuned native
       backend
          |
          v
  heterogeneous hardware
```

Tinygrad gives this project a compact bridge **downward** from architectures toward machines.

Bend may eventually give it a rigorous bridge **upward** from implementations toward meaning.

Those directions are different.

That is exactly why they may fit together.

---

# North-star design rule

> **The AOI is the object. Implementations are evidence.**

SOFTMAX is not Tinygrad code.

FLASH_ATTENTION is not a CUDA kernel.

ATTENTION is not a PyTorch module.

They are computational objects with structure, semantics, realizations, provenance, and measured behavior.

The long-term purpose of tg-aoi is to make those objects explicit enough that humans and AI agents can inspect them, decompose them, translate them, modify them, execute them, compare them, and eventually reason formally about what must remain true.

Tinygrad is presently the strongest candidate for carrying those objects onto diverse hardware.

Bend is a promising future candidate for stating and checking some of the laws those objects are supposed to obey.

That is not duplication.

It is a separation of powers.

---

## References

- Tinygrad runtimes: https://docs.tinygrad.org/runtime/
- Tinygrad repository: https://github.com/tinygrad/tinygrad
- Bend 2 repository: https://github.com/bendlang/bend
- Bend guide / law-driven development: https://github.com/bendlang/bend/blob/main/guide/GUIDE.md

