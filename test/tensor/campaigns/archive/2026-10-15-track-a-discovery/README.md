# Track A discovery batch: raw records

This directory preserves the records behind
[`../../results/track-a-discovery-batch.json`](../../results/track-a-discovery-batch.json)
and the "Track A discovery batch" section of
[`../../results/2026-10-11-targeted-compiler-behavior.md`](../../results/2026-10-11-targeted-compiler-behavior.md).

`track-a-discovery-raw-records.tar.gz` (SHA-256 in `SHA256SUMS`) contains:

* `work/frozen-manifest.json` - the allocation as it was written at
  2026-10-15T11:57:31Z, before any proposal was generated, digest
  `68880da39fd718e9cb93930a5de4e90bdd19518ba25af028391ed5ca7a128cc3`;
* `work/proposals.json` - all 128 proposals with their region request, generated
  source digest, graph facts, and admission record;
* `work/build-records.json` - the four derived input cases of every admitted
  proposal, with each case's admission kind, reference, and input digests;
* `work/run-records-0.json` … `run-records-7.json` - the shard result files: one
  record per execution, with the runner's status, comparison, campaign validity,
  artifacts, and wall clock;
* `work/driver-state.json` - the per-step state, the control records, and the
  stage timings;
* `work/replay-ledger.jsonl` - every interpreter replay the case builder ran,
  with its command;
* `work/track-a-discovery-batch.json` - the assembled compact record, identical
  to the delivered `../../results/track-a-discovery-batch.json`
  (`sha256 e5665877…`);
* `work/track-a-discovery-batch-full.json` - the full assembled record (16.9 MB,
  `sha256 5ec11767…`) with every per-conversion record, every per-occurrence
  conversion match, and every replay command line of all 512 cases;
* `work/logs/shard-*.log` - the eight shard logs;
* `run_tad.sh`, `smoke-propose-build.log`, `run-report.log`, `report3.log`,
  `report4.log` - the remote invocation and its logs;
* `refractir/test/tensor/campaigns/run_track_a_discovery.py` - the driver
  (`sha256 94a08acb…`), the same file as
  [`../../run_track_a_discovery.py`](../../run_track_a_discovery.py).

The 512 case directories themselves - program, inputs, reference, certificate,
emitted kernel, per-stage IR, and MLIR pass dump (about 1.1 GB) - are not in this
archive. They stay at `/host/track-a-discovery-20261015/work/suite/` on
`root@210.28.133.13:22230`, inside the `great_rhodes` container, together with
the two control cases under `work/controls/`.

The evidence was produced from the exported tree
`/host/track-a-discovery-20261015/refractir`, an export of the committed revision
`47d7ceb79c4e748ba4b3916de5f9f5afa08923e7` made with
`git checkout-index -a --prefix=<dir>/`, built in that copy with `clang++` and
`SOLVER=bitwuzla`; nothing was built in `/host/refractir-targeted-20261011`. The
`symirc` it produced digests to `5915fbda…`, the same binary the earlier Track A
and Track C runs recorded. Triton `3.9.0` at
`32f06df4bf1c30885ff61ee3488836ca81ed65f0` was imported from
`/root/triton-latest/python` on Python 3.12.15 with torch 2.5.1+cu121, on eight
Tesla V100-SXM2-32GB devices with driver 535.274.02.
