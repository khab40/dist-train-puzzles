# Assignment Overview

The source notebook, `src/Distributed_Training_Puzzles.ipynb`, teaches distributed training algorithms through five implementation puzzles. It is intended for a CPU-only runtime and uses PyTorch Distributed with the Gloo backend.

## Puzzle A: Ring AllReduce

Implement Ring AllReduce as a more bandwidth-balanced alternative to a naive rank-0 gather/sum/broadcast. The target communication volume per rank is `2s(1 - 1/n)` elements, where `s` is tensor size and `n` is world size.

## Puzzle B: Tensor Parallelism

Implement tensor parallel MLP components inspired by Megatron-LM:

- `ColumnParallelLinear`
- `RowParallelLinear`
- `ParallelTransformerMLP`

The tests compare distributed shards against equivalent non-parallel PyTorch operations.

## Puzzle C: Ulysses

Implement Ulysses-style sequence parallel self-attention:

- Differentiable all-to-all communication.
- Sequence-parallel to head-parallel conversion.
- Head-parallel to sequence-parallel conversion.
- Local attention over reshaped shards.

## Puzzle D: Ring Attention

Implement long-context attention using a ring communication pattern:

- Start asynchronous K/V rotation between neighbor ranks.
- Wait for all scheduled communication work.
- Compute exact unmasked self-attention over a context sharded by sequence length.

## Puzzle E: Zero Bubble Pipelining

Implement a ZB-H2-style pipeline schedule under idealized assumptions:

- Forward time equals backward-input time equals backward-weight time.
- Communication time is ignored.
- Number of microbatches is at least twice the number of pipeline stages.

The schedule separates backward input gradients from backward weight gradients so weight work can be delayed off the critical path.

