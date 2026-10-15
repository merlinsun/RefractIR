# Track B discovery batch: raw records

The bounded discovery batch for Track B (composed padded tails and their storage
checks) is frozen in
[`../../results/track-b-discovery-batch-manifest.json`](../../results/track-b-discovery-batch-manifest.json)
and reduced to
[`../../results/track-b-discovery-batch.json`](../../results/track-b-discovery-batch.json).
This directory keeps the unfiltered driver output the reduction was made from,
so nothing in the review copy is the only witness of a run.

| File | What it is |
|---|---|
| `track-b-discovery-batch.raw.json.gz` | the driver's own record: 128 proposals with every input-variant admission and both reference replays, 1024 per-run staging records, changed-slot lists, and the observer's per-stage readings |
| `generation-summary.json` | the generation step's own tally (128 drawn, 128 admitted) |
| `control.json` | the intentional in-allocation write control, classified separately |
| `verification.json` | three archived runs re-checked from the bytes on disk and both reference precisions |
| `batch-log.txt` | the detached batch log: environment, imported-module and tool digests, device inventory, and the step timestamps |

The reduction drops per-run duplication only: the changed-slot list of a run is
summarised by its guard/gap verdict, and the emitted predicate lines are
recorded once per proposal because every execution of one program emits the
same kernel. No outcome, digest, or refusal is dropped, and the batch produced
no refusals.

Run provenance: committed-tree export at revision
`47d7ceb79c4e748ba4b3916de5f9f5afa08923e7` (built in the export copy under
`/host/track-b-disc-20261017`, never in `/host/refractir-targeted-20261011`),
Triton `3.9.0+git32f06df4` from `/root/triton-latest/python`, eight
NVIDIA Tesla V100-SXM2-32GB devices.
