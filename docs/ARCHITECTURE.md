# Architecture

This repository contains one notebook-based assignment:

- `src/Distributed_Training_Puzzles.ipynb`

The notebook is not an application with a long-running service. It is a set of PyTorch Distributed exercises that write small helper scripts, launch local multi-process workers with `torchrun` or `torch.multiprocessing`, and validate each implementation against embedded tests.

## Runtime Model

The code should run on CPU-only machines by using the PyTorch Distributed Gloo backend. Each puzzle creates a process group, assigns each process a rank, shards tensors or work across ranks, performs communication, and checks the result against a known-correct reference.

```mermaid
flowchart LR
    User[Student opens notebook] --> Notebook[Distributed_Training_Puzzles.ipynb]
    Notebook --> Cells[Implementation cells]
    Notebook --> Tests[Embedded test cells]
    Cells --> Scripts[Generated helper scripts]
    Scripts --> Torchrun[torchrun local workers]
    Tests --> MP[torch.multiprocessing workers]
    Torchrun --> Gloo[PyTorch Distributed Gloo backend]
    MP --> Gloo
    Gloo --> Ranks[Rank 0..N-1 processes]
    Ranks --> Assert[Reference comparisons and assertions]
```

## Puzzle Map

```mermaid
flowchart TD
    A[Puzzle A: Ring AllReduce] --> B[Puzzle B: Tensor Parallelism]
    B --> C[Puzzle C: Ulysses Sequence Parallelism]
    C --> D[Puzzle D: Ring Attention]
    D --> E[Puzzle E: Zero Bubble Pipelining]

    A --> A1[Collective communication from point-to-point ops]
    B --> B1[Shard Transformer MLP weights across tensor-parallel ranks]
    C --> C1[Move between sequence-parallel and head-parallel layouts]
    D --> D1[Rotate K/V blocks around a rank ring]
    E --> E1[Build a pipeline schedule with delayed weight gradients]
```

## Puzzle A: Ring AllReduce

`ring_allreduce(x, pg)` should replace a naive rank-0 gather/sum/broadcast with a ring algorithm. Every rank communicates with only its neighbors, balances communication volume across ranks, and ends with the same reduced tensor.

```mermaid
flowchart LR
    R0((Rank 0)) --> R1((Rank 1))
    R1 --> R2((Rank 2))
    R2 --> R3((Rank 3))
    R3 --> R0

    R1 -. previous .-> R0
    R2 -. previous .-> R1
    R3 -. previous .-> R2
    R0 -. previous .-> R3
```

Expected behavior:

- Split the local tensor into rank-sized chunks.
- Run reduce-scatter around the ring so each rank owns one reduced chunk.
- Run all-gather around the ring so every rank reconstructs the full reduced tensor.
- Match `dist.all_reduce` for tested tensor shapes, dtypes, rank order, and process-group subsets.

## Puzzle B: Tensor Parallel Transformer MLP

The tensor-parallel MLP should behave like a normal Transformer MLP while sharding large weight matrices across ranks.

```mermaid
flowchart LR
    X[Input X] --> CP[ColumnParallelLinear]
    CP --> GELU[GELU]
    GELU --> Drop1[Dropout]
    Drop1 --> RP[RowParallelLinear]
    RP --> Drop2[Dropout]
    Drop2 --> Y[Output Y]

    subgraph TP[Tensor-parallel process group]
        W1A[fc1 shard rank 0]
        W1B[fc1 shard rank 1]
        W2A[fc2 shard rank 0]
        W2B[fc2 shard rank 1]
    end

    CP -. owns .-> W1A
    CP -. owns .-> W1B
    RP -. owns .-> W2A
    RP -. owns .-> W2B
```

Expected behavior:

- `ColumnParallelLinear` stores output-feature shards of the first projection.
- `RowParallelLinear` stores input-feature shards of the second projection.
- `ParallelTransformerMLP` composes both layers with activation and dropout.
- Distributed output and gradients match the equivalent non-parallel `TransformerMLP`.

## Puzzle C: Ulysses Self-Attention

Ulysses should shard long sequences across ranks, use all-to-all communication to redistribute data by attention heads, run local attention, and convert the output back to sequence-parallel layout.

```mermaid
flowchart TD
    SP[Sequence-parallel input shard] --> QKV[Local QKV projections]
    QKV --> S2H[seq_par_to_head_par]
    S2H --> A2A1[DiffAllToAll forward]
    A2A1 --> Local[Local scaled dot-product attention]
    Local --> H2S[head_par_to_seq_par]
    H2S --> A2A2[DiffAllToAll forward]
    A2A2 --> Out[Sequence-parallel output shard]

    Grad[Backward pass] -. uses differentiable all-to-all .-> A2A1
    Grad -. uses differentiable all-to-all .-> A2A2
```

Expected behavior:

- `DiffAllToAll.forward` performs all-to-all redistribution.
- `DiffAllToAll.backward` applies the inverse redistribution for gradients.
- `seq_par_to_head_par` changes layout from sequence-sharded to head-sharded.
- `head_par_to_seq_par` restores sequence-sharded output.
- Local attention over redistributed heads matches full reference attention after reassembly.

## Puzzle D: Ring Attention

Ring Attention should compute exact unmasked self-attention for sequence-sharded context without materializing the full K/V context on every rank at once.

```mermaid
sequenceDiagram
    participant R0 as Rank 0
    participant R1 as Rank 1
    participant R2 as Rank 2

    Note over R0,R2: Each rank starts with local Q, K, V sequence shard
    R0->>R1: send K/V block
    R1->>R2: send K/V block
    R2->>R0: send K/V block
    R1-->>R0: receive previous K/V block
    R2-->>R1: receive previous K/V block
    R0-->>R2: receive previous K/V block
    Note over R0,R2: Compute partial attention contribution
    Note over R0,R2: Repeat rotation until all K/V blocks were seen
```

Expected behavior:

- `start_kv_rotate` schedules asynchronous neighbor sends and receives with `dist.batch_isend_irecv`.
- `wait_all` waits for every communication request.
- `ring_attention` rotates K/V shards around the rank ring.
- Each rank keeps its local Q shard and updates attention output using numerically stable online softmax logic.
- Final output matches `torch.nn.functional.scaled_dot_product_attention` against full K/V context.

## Puzzle E: Zero Bubble Pipeline Schedule

`zb_h2_schedule(n, m)` should return a schedule table for `n` pipeline stages and `m` microbatches. It models forward work, backward-input work, and backward-weight work as equal-duration slots while delaying weight-gradient work off the critical path.

```mermaid
flowchart LR
    subgraph S0[Stage 0]
        S0F0[F0] --> S0F1[F1] --> S0B0[B0] --> S0W0[W0]
    end

    subgraph S1[Stage 1]
        S1I[idle] --> S1F0[F0] --> S1F1[F1] --> S1B0[B0] --> S1W0[W0]
    end

    subgraph S2[Stage 2]
        S2I0[idle] --> S2I1[idle] --> S2F0[F0] --> S2F1[F1] --> S2B0[B0] --> S2W0[W0]
    end

    S0F0 -. activates .-> S1F0
    S1F0 -. activates .-> S2F0
    S2B0 -. grad input .-> S1B0
    S1B0 -. grad input .-> S0B0
    S0W0 -. delayed weight work .-> S1W0
    S1W0 -. delayed weight work .-> S2W0
```

Expected behavior:

- Stage `i` starts with `i` idle slots.
- Every stage schedules `m` forward tokens, `m` backward-input tokens, and `m` backward-weight tokens.
- Backward-input work stays on the critical path.
- Backward-weight work is delayed where possible to reduce pipeline bubbles.
- The generated rows satisfy the notebook's structural and dependency tests.

## Data And Control Boundaries

```mermaid
flowchart TB
    subgraph Notebook[Notebook]
        Impl[Student implementations]
        Ref[Reference PyTorch operations]
        Tests[Assertions]
    end

    subgraph Distributed[Distributed runtime]
        PG[ProcessGroup]
        Mesh[DeviceMesh]
        P2P[isend and irecv]
        Collective[all_reduce and all_to_all]
    end

    subgraph Local[Local machine]
        CPU[CPU tensors]
        Gloo[Gloo transport]
        Workers[Multiple Python workers]
    end

    Impl --> PG
    Impl --> Mesh
    PG --> P2P
    PG --> Collective
    P2P --> Gloo
    Collective --> Gloo
    Gloo --> Workers
    Workers --> CPU
    Ref --> Tests
    Impl --> Tests
```

The architecture intentionally keeps all code inside the notebook. Generated files are temporary execution artifacts, and the source of truth remains `src/Distributed_Training_Puzzles.ipynb`.
