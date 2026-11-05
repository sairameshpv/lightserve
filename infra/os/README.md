# infra/os: the GPU machine's operating system (Phase 4)

## The topology report: look before tuning (8g)

Before pinning lightserve to particular cores (8h) or changing OS settings (8i), a map of the GPU
machine and **what it means for us**:

```
python3 infra/os/topology.py --out runs/topology     # on the GPU VM; prints the report too
```

Writes `report.txt` (to read) and `topology.json` (the facts and the suggested layout, for 8h's
scripts). Python 3 standard library only: nothing to install on the VM.

**What it reads**, all from Linux's own files (`/sys`, `/proc`) plus `nvidia-smi`:
- **cores and twins:** the hyper-threads that share one physical core ("0+12"): pinning hands out
  whole physical cores, so the engine never shares one with noise;
- **NUMA nodes** (groups of cores with their own nearby memory) and the GPU's node and nearby cores
  (from its PCI folder; a VM often says "-1" = unknown);
- **the GPU's settings:** persistence mode, current/max clocks, power limit, driver; NVIDIA's own
  map (`nvidia-smi topo -m`) kept as is. If `nvidia-smi` fails, its error message is shown;
- **interrupts:** each device's count so far and which cores may handle it; whether
  `irqbalance` runs (it would undo steering);
- **settings 8i will test:** CPU speed policy ("governor") and sleep states (if exposed), huge
  pages (THP), swappiness, network queue limits, cores set aside at boot (`isolcpus`).

**What it says it means:**
- **NUMA:** 1 node → "NUMA pinning can't help here" (pinning is then only about keeping other work
  off the engine's core); 2+ nodes → keep lightserve on the GPU's node, with the `numactl` command.
- **A suggested layout for 8h**, whole physical cores: **housekeeping** = cpu0's core (interrupts,
  the OS); **engine** = the first free core near the GPU, its twin left idle (lightserve's
  GPU-feeding thread); **server** = the next core (the web server); **rest** = everything else
  (including the noisy neighbour in the experiment).
- **interrupts to move** off the reserved cores, and the irqbalance warning.
- **each 8i A/B with today's value**, or "can't run here" when the VM hides the setting.

Example, on the test's fake machine shaped like the L40S VM (24 vCPUs, 1 NUMA node):

```
WHAT IT MEANS
  NUMA: 1 NUMA node: every core is equally close to memory and the GPU, so NUMA pinning can't help here; ...
  suggested layout for 8h (whole physical cores):
    housekeeping (interrupts, the OS): 0,12
    engine: 1 (its twin 13 left idle)
    server: 2,14
    rest (everything else): 3-11,15-23
  interrupts to move to housekeeping: 24 (nvidia), 25 (virtio0-input.0)
  irqbalance is running: it would move interrupts back onto the reserved cores; stop it for 8h ...
  tuning A/Bs (8i), current values:
    gpu: persistence off, SM clock now 210 MHz (max 2520); A/B: persistence on + clocks locked at max ...
    cpu_power: CPU speed policy (governor) and sleep states not exposed (typical VM): this A/B can't run here
```

## How it was built and checked (without the GPU machine)

`--root` points the script at a folder shaped like `/`, so the tests (`tests/test_topology.py`,
22, run by CI) build **fake machines**: the L40S-shaped VM, a 2-node machine with the GPU on node 1
(the layout lands on node 1's cores), a VM reporting the GPU's node as -1, too few cores for a
layout, missing `nvidia-smi`, and an `nvidia-smi` that errors. `nvidia-smi` output is parsed
from a saved sample. On the Mac it also ran on real Linux files (Docker): it caught one wrong
answer: ARM Linux has no x86 CPU flags, so "virtual machine" is now "unknown" there, not "no".

**Not checked yet:** the real `nvidia-smi` and the real VM; that's the first two minutes of
session VM-2. If a query field were wrong for the driver, the report shows NVIDIA's error.

## CPU isolation and the noisy-neighbour experiment (8h)

lightserve is "launch-bound": the GPU often waits for Python on the CPU to send the next piece of
work, so a busy neighbour on the same machine should hurt it, mostly in the slowest requests.
**Isolation** gives it its own cores from the topology report's layout and keeps the neighbour
and hardware interrupts off them.

**lightserve pins itself** (one process, many threads; only the engine thread must be alone):
`python -m server --server-cpus 2,14 --engine-cpus 1`. The whole process starts on the server
cores (every thread inherits them); the engine thread (the one feeding the GPU) then moves
itself onto its own core. Checked on Linux: the engine thread alone on core 1, every other
thread on the server cores.

**`isolate.py` puts the machine in one state**, fresh each time (both containers recreated), so
it is the A/B tool's per-round setup:

| State | lightserve | neighbour (a CPU burn: one spinning Python per core) | interrupts |
|---|---|---|---|
| quiet | unpinned | none | put back |
| noisy | unpinned | every core **but housekeeping** | put back |
| isolated-quiet | engine core + server cores; engine thread alone; its twin idle | none | moved to housekeeping, irqbalance stopped |
| isolated-noisy | same | only the "rest" cores | moved to housekeeping |
| restore | stopped | stopped | put back, irqbalance restarted if it ran |

- **Housekeeping is never burned**, not even in "noisy": the OS and the load generator (Locust,
  run under `taskset` on those cores) live there in every state. Otherwise the neighbour would slow
  the thing *measuring* lightserve, and the A/B would mix the two up.
- **Interrupts:** the originals are saved to a file on the first move (a second move keeps them);
  every non-isolated state puts them back, so alternating A/B rounds don't leak into each other;
  interrupts Linux refuses to move (kernel-managed) are listed, not fatal.
- **Every isolated state checks itself:** it reads each lightserve thread's allowed cores from
  `/proc` (through `docker exec`) and exits with an error ("PINNING WRONG") unless exactly one thread
  sits alone on the engine core and the rest on the server cores. An error stops the A/B tool, so
  a round is never measured with isolation silently not working.

```
sudo python3 infra/os/isolate.py isolated-noisy --topology runs/topology/topology.json [--dry-run]
```
(`--server-args` takes `=`: `--server-args=--fake-engine`.)

**Mac rehearsal** (Docker's 8-core Linux VM, the fake engine built from this code, interrupts
printed only): every state ran for real: containers on the right cores; "pinning OK: engine thread
alone on core 1; 2 others on 2-3"; the neighbour at 402% CPU on its 4 cores. One short A/B,
noisy vs isolated-noisy, through the A/B tool: every round set up and checked, 292 requests, 0
failed. (The fake engine sleeps instead of computing, so that proves the plumbing, not the
effect.) **One hint:** the isolated side was ~7% slower on typical latency (background measure, 2
short rounds): pinning gives lightserve 3 cores instead of sharing all of them, and its engine and
web-server threads hand Python's global lock (GIL) across cores. Hence a third A/B on the GPU:
quiet vs isolated-quiet (does pinning help or hurt *without* a neighbour?).

## Session VM-2, attempt 1 (2026-11-03, 20 min, stopped early)

- **Topology report, first real run:** Xeon Gold 6338, 24 vCPUs = 12 cores x 2, twins numbered
  **next to each other** (0+1, 2+3, ...; the test machine had 0+12), 1 NUMA node. Layout:
  housekeeping 0-1, engine 2 (twin 3 idle), server 4-5, rest 6-23. 9 disk-queue interrupts are
  kernel-managed and can't be moved (listed, as designed).
- **The GPU image's first real run:** first completion 13.1 s (kernel compile), then 0.062 s;
  pinning OK (engine thread alone on core 2).
- **A/Bs aborted:** under 10 users lightserve crawled at ~1 token/s with the GPU 98% busy, ~20
  requests per 60 s round. Cause: the GPU kernels re-ran their tuning search for almost every new
  size (see `kernels/README.md`, "Tuning once per size group"), and every round's fresh container
  started from scratch. A GPU-bound server can't show CPU interference, and 20 requests can't give
  a p99. Fixed: tuning per size group, and the results kept on disk across containers
  (`isolate.py`'s `tuning_cache`).

## Session VM-2 runbook (< 1 h, needs a go-ahead)

**Prediction, written before:** noisy raises latency p99 by tens of %; isolated-noisy is within a
few % of quiet. **Deciding measure for every A/B: latency p99** (the prediction is about the
slowest requests). Toy 4-layer model with `--max-model-len 512` (the prompts need it): little GPU
work, so CPU interference shows most. Agree a hard stop time (start + 55 min) before starting.

| Min | Step |
|---|---|
| before | CI built `lightserve:gpu-latest` from this code (the `image` workflow green) |
| 0-5 | `nebius compute instance start --id <node 0>` (a spot VM: may need a retry); ssh in; `sudo docker stop vllm-server` (it holds the GPU); pull `ghcr.io/sairameshpv/lightserve:gpu-latest` (~7 GB) and `python:3.11-slim` in the background; a **fresh clone** (`~/lightserve` on node 0 has old uncommitted work); `python3 -m venv ~/ab && ~/ab/bin/pip install locust==2.46.4` |
| 5-7 | `python3 infra/os/topology.py --out runs/topology` |
| 7-11 | **GPU kernel tests in the new image:** `sudo docker run --rm --gpus all --user root --entrypoint sh <image> -c "pip install -q pytest && python -m pytest -q kernels/tests/test_tiled_matmul.py kernels/tests/test_flash_attention.py"`, all pass incl. `test_one_tuning_per_size_group_on_the_gpu`. Then `isolate.py isolated-quiet` + one completion. If anything fails: restore, stop the node, report |
| 11-14 | **Fill the tuning cache** (not measured): 2 min of the A/B load on housekeeping, `taskset -c <housekeeping> ~/ab/bin/locust -f benchmarks/locustfile.py --headless -u 10 -r 10 -t 120s --host http://localhost:8000 --target lightserve`. Must show **hundreds of requests per minute**, and `/var/tmp/lightserve_triton` must hold tuning files; if not, stop and report |
| 14-25 | **A/B 1, quiet vs noisy:** how much the neighbour hurts |
| 25-36 | **A/B 2, noisy vs isolated-noisy:** how much isolation gives back |
| 36-47 | **A/B 3, quiet vs isolated-quiet:** what pinning costs or gives without a neighbour (dropped if the hard stop gets close) |
| 47-50 | `isolate.py restore`; copy `runs/` to the Mac; `nebius compute instance stop` |

Each A/B (3 rounds a side, 30 s warm-up + 60 s, ~11 min), with the load generator on housekeeping.
Run all three as one background script on the node (`sudo nohup`), with **absolute paths**:
under `sudo`, `~` is `/root` (attempt 1 lost a minute to that).
```
ISO="python3 infra/os/isolate.py --topology runs/topology/topology.json"
sudo taskset -c <housekeeping> ~/ab/bin/python benchmarks/ab/ab_run.py --host http://localhost:8000 \
    --out runs/vm2/ab1 --a-setup "$ISO quiet" --b-setup "$ISO noisy" \
    --warmup-s 30 --round-s 60 --primary "latency p99 (s)"
```
If a first round has < 1,000 requests, p99 rests on too few slow requests: noted in the results,
not hidden.

## OS tuning, one setting at a time (8i)

**What it's for:** a careful, repeatable way to test one machine setting at a time. "No
difference" is a good result, as long as it can be trusted. A comparison is only as good as its
two sides (a taste test where the cook forgets the spice on half the "spiced" plates says "no
difference", and it's a lie), so every change is **read back** before a round is measured.

**`tune.py` puts one setting group in "as found" or "tuned":**

| Group | As found on the node | Tuned | Where it lives |
|---|---|---|---|
| GPU clock | free (idles at 210 MHz, max 2,520) | locked at max (`nvidia-smi -lgc 2520,2520`); must read >= 90% of max | the GPU |
| Memory | huge pages "madvise", swappiness 60 | huge pages "always", swappiness 10 | whole machine |
| Network | network-card backlog 1000; connection queue 4096 | 16384; 65535 | backlog: whole machine (`tune.py`); queue: **lightserve's own container** (`isolate.py --somaxconn`) |
| CPU power | hidden by the VM | — | reported as "can't run here" |

- **The originals are saved once**, to `/var/tmp/lightserve_tune_state.json`, before the first
  change; "as found" writes them back (nothing saved = already as found); `restore` puts every group
  back and forgets them. A node stop/start also resets all of these (none is permanent).
- **Exit codes:** 1 = a change didn't take ("DIDN'T TAKE", with what the machine reads), 2 = the
  group can't run here. As the A/B tool's per-round setup, either stops the run before a round is
  measured on the wrong setting.
- **The connection queue is per container (found on the Mac):** on Docker's Linux a container
  reads its own default 4096 whatever the machine's value is, and the network-card backlog isn't
  visible inside one at all. So the queue is set when lightserve's container starts (`--sysctl`)
  and read back inside it; setting it on the machine would have tested nothing.
- **Not tuned:** huge-page "defrag" (its "always" is known to cause pauses, and would mix two
  changes into one test); the open-file limit (Docker already gives containers a very high one);
  persistence mode (already on on the node).

```
python3 infra/os/tune.py check                        # what's here, today's values
sudo python3 infra/os/tune.py memory tuned [--dry-run]
sudo python3 infra/os/tune.py restore
```

**How it was checked (Mac):** tests on a fake machine folder and a fake `nvidia-smi`: each change
written and read back, a change the machine ignores caught, originals kept across rounds,
alternating rounds + restore leave every value as found, missing settings refused. On Docker's real
Linux: `check` and every dry run behaved, nothing changed; lightserve's container started with
`--somaxconn 65535` read 65535 inside while the machine read 4096. **Not run yet:** `isolate.py`
end to end with `--somaxconn` (its tuning-folder step needs root; the node has it), the real
`nvidia-smi -lgc`.

### Predictions and the win rule (written before any run)

| A/B | Prediction | Deciding measure |
|---|---|---|
| GPU clock | steadier and slightly faster per token: the GPU no longer ramps its clock up between bursts | time/token p50 (ms) |
| Memory | no difference: 96 GB machine, tiny model, nothing gets swapped | latency p99 (s) |
| Network | no difference: 10 users never fill a queue of 4,096, and the web server (uvicorn) asks for only 2,048 | latency p99 (s) |

**Win rule:** a setting wins only if the A/B tool's verdict on its deciding measure is "better";
"not sure" is no win. **Only a winner gets setup code** (a start-up file for the VM, a Kubernetes
setting), in a later step.

### Session VM-3 runbook (< 1 h, needs a go-ahead; best after the VM-2 retry)

Same node, same image and fresh clone as VM-2 (its tuning folder `/var/tmp/lightserve_triton`
survives a node stop). lightserve in the plain `quiet` state (unpinned, no neighbour), so only the
one setting differs; if VM-2 shows pinning clearly helps, use `isolated-quiet` on both sides
instead. Hard stop: start + 55 min.

| Min | Step |
|---|---|
| 0-5 | start node 0; `sudo docker stop vllm-server`; `git pull` in the clone |
| 5-6 | `sudo python3 infra/os/tune.py check`: GPU, memory, network available; CPU power can't run here |
| 6-9 | `isolate.py quiet` + 2 min of the A/B load to fill the tuning cache (not measured): hundreds of requests a minute, or stop and report |
| 9-20 | **A/B 1, GPU clock** |
| 20-31 | **A/B 2, memory** |
| 31-42 | **A/B 3, network** |
| 42-46 | `tune.py restore`, `isolate.py restore`, `tune.py check` (all as found); copy `runs/` to the Mac; stop the node |

Run as one background script (`sudo nohup`), **absolute paths** (under `sudo`, `~` is `/root`):
```
TUNE="python3 /home/<user>/ls_vm2/infra/os/tune.py"
ISO="python3 /home/<user>/ls_vm2/infra/os/isolate.py --topology runs/topology/topology.json"
AB="taskset -c <housekeeping> /home/<user>/ab/bin/python benchmarks/ab/ab_run.py --host http://localhost:8000 --warmup-s 30 --round-s 60"
sudo $AB --out runs/vm3/gpu --primary "time/token p50 (ms)" \
    --a-setup "$TUNE gpu as-found && $ISO quiet" --b-setup "$TUNE gpu tuned && $ISO quiet"
sudo $AB --out runs/vm3/memory --primary "latency p99 (s)" \
    --a-setup "$TUNE memory as-found && $ISO quiet" --b-setup "$TUNE memory tuned && $ISO quiet"
sudo $AB --out runs/vm3/network --primary "latency p99 (s)" \
    --a-setup "$TUNE network as-found && $ISO quiet" --b-setup "$TUNE network tuned && $ISO quiet --somaxconn 65535"
```
lightserve restarts every round, so memory and network changes apply from a fresh start.

### Results

| A/B | Verdict on the deciding measure | Change | Prediction right? | Wins? |
|---|---|---|---|---|
| GPU clock | not run yet | | | |
| Memory | not run yet | | | |
| Network | not run yet | | | |
