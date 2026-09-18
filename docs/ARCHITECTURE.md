# Architecture language

`tg-aoi` treats AI architecture as a reducible language.

| Language analogy | Architecture layer | Examples |
|---|---|---|
| Phonetics | tensor/axis/type semantics | batch, token, feature, dtype, shape |
| Letters | smallest stable operations | ADD, MUL, REDUCE, EXP, INDEX, CAST |
| Words | reusable AOIs | MATMUL, SOFTMAX, RMSNORM, CONV |
| Sentences | useful blocks | ATTENTION, MLP, optimizer step |
| Paragraphs | architecture blocks | TransformerBlock, ViTBlock, DiTBlock |
| Chapters | complete model families | ViT, Llama, RF-DETR |
| Novels | trainable systems | model + heads + objectives + optimizer |
| Library | empirical research corpus | architectures + experiments + results |

## Non-negotiable invariant

Every declared AOI must remain expandable toward the chosen primitive substrate. A name is compression, not opacity.

## Grammar

Graphs are typed. Composition is legal only where tensor types, dimensions, axes, and state transitions are compatible. Future versions should support symbolic dimension unification and richer semantic axis checking.

## Tinygrad first

The canonical IR is framework-neutral, but Tinygrad is the first execution target. This keeps one end-to-end route working before broadening exporters.
