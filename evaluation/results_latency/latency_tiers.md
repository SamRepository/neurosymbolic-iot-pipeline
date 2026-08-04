# Latency by deployment tier (E5 — reviewer item R1-11)

**Two different quantities are reported per row and must not be conflated.**
*Time-to-decision* is the end-to-end latency of one inference call — what strict
edge real-time operation is bound by, since no window can be acted on before its
batch completes. *Amortised per-window* is time-to-decision divided by batch size:
a throughput measure, and the one the sub-5 ms headline refers to.

## CASAS

| Batch | Tier | Time-to-decision (ms) | Amortised /window (ms) | Throughput (win/s) | Sub-5 ms? | ≤100 ms? |
|---|---|---|---|---|---|---|
| 1 | edge (single-window, strict real-time) | 198.66 | 198.66 | 10.05 | no | no |
| 5 | fog (batched) | 113.71 | 22.74 | 48.16 | no | no |
| 10 | fog (batched) | 112.55 | 11.26 | 89.31 | no | no |
| 20 | fog (batched) | 135.92 | 6.80 | 147.79 | no | no |
| 50 | fog (batched) | 201.33 | 4.03 | 249.84 | yes | no |

## SPHERE

| Batch | Tier | Time-to-decision (ms) | Amortised /window (ms) | Throughput (win/s) | Sub-5 ms? | ≤100 ms? |
|---|---|---|---|---|---|---|
| 1 | edge (single-window, strict real-time) | 56.20 | 56.20 | 18.29 | no | yes |
| 5 | fog (batched) | 56.77 | 11.35 | 88.18 | no | yes |
| 10 | fog (batched) | 67.39 | 6.74 | 155.57 | no | yes |
| 20 | fog (batched) | 91.79 | 4.59 | 245.02 | yes | yes |
| 50 | fog (batched) | 101.44 | 2.03 | 497.27 | yes | no |

## What the numbers do and do not support

- Sub-5 ms is an amortised per-window figure, and on CASAS it is reached at exactly one operating point: batch 50 (4.03 ms at batch 50). On SPHERE it is reached at batch 20 and above. At batch 1 the amortised and true latencies coincide and are 198.66 ms (CASAS) and 56.20 ms (SPHERE).

- Batching buys throughput, not responsiveness. For CASAS the end-to-end latency of a batch-50 call is 201.33 ms, marginally *higher* than the 198.66 ms single-window call, because the amortised figure divides a larger total across more windows. No window in a batch can be acted on before the whole batch completes, so the sub-5 ms figure never describes how quickly an individual event is answered.

- The headline throughput figure of ~497 windows/s is SPHERE at batch 50 (497.27/s). The corresponding CASAS figure is 249.84/s - roughly half. If the manuscript quotes 497 windows/s it must name SPHERE, or a reader will attribute it to the CASAS pipeline the ablation is built on.

- Against a 100 ms interactive budget, CASAS meets the bound at batch sizes none and SPHERE at [1, 5, 10, 20]. Single-window CASAS (198.66 ms) does not, so strict edge real-time operation is not demonstrated for the CASAS pipeline on this hardware at any batch size; what is demonstrated is fog-tier batched feasibility.

- At batch 1 the cost is dominated by symbolic reasoning, not by the network: 147.49 ms of 198.66 ms (74 %), against 48.34 ms of neural inference. Any latency optimisation should target the rdflib SPARQL path first.

## ARM-class hardware

No ARM-class board (Raspberry Pi 4 / Jetson) was available for this revision, so no ARM measurement is reported. The manuscript should keep this as an explicit limitation and future-work item rather than extrapolating desktop CPU timings to edge hardware: the pipeline is dominated by an in-memory SPARQL engine whose performance on ARM is not predictable from these numbers.

## Hardware provenance

The stored benchmark records device='cpu' and torch='2.9.1+cpu' but **does not record the CPU model**. The manuscript attributes these timings to an i7-12700K; that attribution rests on the authors' record, not on the artifact, and cannot be verified from it. The machine running this analysis is a Intel64 Family 6 Model 142 Stepping 10, GenuineIntel, i.e. not the machine that produced the measurements — so the numbers were deliberately not regenerated here. Recommend adding CPU model capture to the benchmark's metadata so future runs are self-describing.
