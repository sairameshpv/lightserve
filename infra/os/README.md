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

## Session VM-2 retry results (2026-11-07, 02:27-03:07 UTC, ~40 min, node stopped)

**In one line:** a CPU-burning neighbour made lightserve's slowest requests ~17% slower; pinning
lightserve to its own cores with the neighbour kept off them won back about two thirds of that
(with a caveat: the isolated neighbour was also smaller, below).

**Before measuring (the go/no-go checks):**
- **The kernel fix holds on the real L40S:** all 55 kernel tests pass in the image, including both
  "one tuning per size group" GPU tests (never run on a GPU before).
- **Pinning:** engine thread alone on core 2, the 5 others on 4-5; interrupts moved, then restored.
- **Speed under load:** first completion 13.8 s (kernels compiling), then 335 requests in the 2-min
  cache fill, **~167 a minute** (attempt 1: ~20). At the low end of "hundreds a minute", but the
  server is no longer stuck tuning; the tuning cache held 11 MB afterwards.

**A/B 1, quiet vs noisy** (6 rounds, ABBAAB, 30 s warm-up + 60 s, 0 failures):

| measure | quiet | noisy | change (95% range) | verdict |
|---|---|---|---|---|
| **latency p99 (deciding)** | 5.50 s | 6.42 s | **+16.7%** (+11.6 .. +21.8) | worse |
| time/token p50 | 24.7 ms | 28.1 ms | +13.8% (+9.8 .. +17.9) | worse |
| tokens/s | 402 | 352 | -12.6% (-15.7 .. -9.4) | worse |

Every noisy round was slower than every quiet round. Prediction "p99 up by tens of %": right, at
the low end.

**A/B 2, noisy vs isolated-noisy** (same set-up, 0 failures):

| measure | noisy | isolated-noisy | change (95% range) | verdict |
|---|---|---|---|---|
| **latency p99 (deciding)** | 6.43 s | 5.84 s | **-9.3%** (-16.3 .. -2.3) | better |
| time/token p50 | 28.5 ms | 25.7 ms | -9.8% (-15.6 .. -4.1) | better |
| tokens/s | 348 | 386 | +10.7% (+3.5 .. +17.9) | better |

latency p50 "not sure" (one isolated round was unusually fast: 3.13 s). Prediction "isolated-noisy
within a few % of quiet": a little optimistic. Across the two A/Bs (separate runs, so a rough
reading) quiet 5.50 s, noisy 6.42/6.43 s, isolated-noisy 5.84 s: **~6% above quiet**, i.e. about
two thirds of the damage taken back.

**Caveats, not hidden:**
- **A/B 2 changed two things at once:** `noisy` burns 22 cores (all but housekeeping), `isolated-noisy`
  only the 18 "rest" cores (`isolate.py`), so the neighbour was ~18% smaller on the isolated side.
  "Two thirds won back" is pinning *and* a smaller neighbour together ("carve-out isolation"); A/B 3
  was dropped, so the two can't be separated. Next time: burn the same 18 cores in both states.
- **~200 requests per round** (169-209), under the runbook's 1,000: each round's p99 is set by its
  ~2 slowest requests. The verdicts still agree with the steadier time/token and tokens/s measures.
- **A/B 3 (quiet vs isolated-quiet) dropped for time**, so "does pinning help or hurt with no
  neighbour" (the Mac hinted ~7% slower) is still open. Lost to: 3.5 min (a zsh mistake: `$SSH`
  isn't word-split, so the wait-for-ssh loop never ran ssh), and the node's disk being full.
- **Node disk:** the image pull failed at 6.4 GB free; only the outdated lightserve image was
  removed (the vLLM, OpenRLHF and NVIDIA PyTorch images kept). 97% full again after the pull: the
  next session must check `df` first.
- Attempt 1's results folder on the node is root-owned and was left alone; these went to
  `runs/vm2r/` there and `runs/vm2_retry/` on the Mac (untracked, like the rest of `runs/`).

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
| Network | network-card backlog 1000 | 16384 | whole machine |
| CPU power | hidden by the VM | — | reported as "can't run here" |

- **The originals are saved once**, to `/var/tmp/lightserve_tune_state.json`, before the first
  change; "as found" writes them back (nothing saved = already as found); `restore` puts every group
  back and forgets them. A node stop/start also resets all of these (none is permanent).
- **Exit codes:** 1 = a change didn't take ("DIDN'T TAKE", with what the machine reads), 2 = the
  group can't run here. As the A/B tool's per-round setup, either stops the run before a round is
  measured on the wrong setting.
- **The connection queue is per container (found on the Mac):** on Docker's Linux a container
  reads its own default 4096 whatever the machine's value is, and the network-card backlog isn't
  visible inside one at all. So the queue can only be set when lightserve's container starts
  (`isolate.py --somaxconn N`: Docker's `--sysctl`, read back inside the container).
- **...and it isn't tested at all (found by the independent review):** Linux gives a server the
  *smaller* of what it asks for and the queue setting. lightserve's web server (uvicorn) asks for
  2048, below both 4096 and 65535, so its queue is 2048 on both sides: a "no difference" would be
  guaranteed, not measured. Testing it would first need a server option for a bigger request.
  `--somaxconn` stays in `isolate.py` (tested) for that day; VM-3 doesn't use it.
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
alternating rounds + restore leave every value as found, missing settings refused; after the
review: `restore` tries every group past a failure and keeps the originals, `nvidia-smi` failures
are messages, a GPU status field the driver lacks doesn't block the group, leftover originals warn.
On Docker's real Linux: `check` and every dry run behaved, nothing changed (swap read right: 1023
MB). For the connection-queue finding, **the `docker run` command `isolate.py --somaxconn 65535
--dry-run` prints was run by hand** (not `isolate.py` itself): inside that container 65535, the
machine 4096 at the same moment. **Not run yet:** `isolate.py` end to end with `--somaxconn` (its
tuning-folder step needs root; the node has it; VM-3 doesn't use it), the real `nvidia-smi -lgc`
and `-rgc`, the GPU status field names, a GPU A/B under load.

### Predictions and the win rule (written before any run)

| A/B | Prediction | Deciding measure |
|---|---|---|
| GPU clock | steadier and slightly faster per token: the GPU no longer ramps its clock up between bursts | time/token p50 (ms) |
| Memory | no difference: 96 GB machine, tiny model, nothing gets swapped. If `tune.py check` says "no swap" (likely on a cloud VM; the topology report didn't record it), swappiness has nothing to act on and this A/B is huge pages alone | latency p99 (s) |
| Network (backlog) | no difference: 10 users' traffic never fills the incoming-packet queue of 1,000 per core | latency p99 (s) |

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
| 5-6 | `sudo python3 infra/os/tune.py check`: GPU, memory, network available; CPU power can't run here; no swap noted or not; if it prints a WARNING about saved originals (an earlier session didn't finish), `tune.py restore` first |
| 6-9 | `isolate.py quiet` + 2 min of the A/B load to fill the tuning cache (not measured): hundreds of requests a minute, or stop and report |
| 9-20 | **A/B 1, GPU clock** |
| 20-31 | **A/B 2, memory** |
| 31-42 | **A/B 3, network** |
| 42-46 | the script's exit step: `tune.py restore`, `isolate.py restore`, `tune.py check`, compared with the session's first `check` (must print "AS FOUND"; GPU lines left out: idle clock and power wander on their own, the reset is proven by `-rgc` succeeding); copy `runs/` to the Mac; stop the node |

Run as one background script (`sudo nohup bash vm3.sh`; everything inside already runs as root, so no `sudo` in it). It starts with a `cd` into the clone,
so every relative path (`runs/...`, `benchmarks/...`) means the same thing; the rest are absolute
(under `sudo`, `~` is `/root`). `set -e` stops at the first failure (a "DIDN'T TAKE" or "can't run
here" stops that A/B, and the later ones must not run on a machine left half-tuned), and the
`trap` puts everything back on the way out, whether the script finished or stopped:
```
set -e
cd /home/<user>/ls_vm2
TUNE="python3 /home/<user>/ls_vm2/infra/os/tune.py"
ISO="python3 /home/<user>/ls_vm2/infra/os/isolate.py --topology runs/topology/topology.json"
AB="taskset -c <housekeeping> /home/<user>/ab/bin/python benchmarks/ab/ab_run.py --host http://localhost:8000 --warmup-s 30 --round-s 60"
mkdir -p runs/vm3
$TUNE check | tee runs/vm3/check_start.txt       # as found, before anything changes
nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,clocks_event_reasons.sw_power_cap,clocks_event_reasons.hw_slowdown,clocks_event_reasons.sw_thermal_slowdown \
    --format=csv -l 5 > runs/vm3/gpu_samples.csv &  # the GPU under load, every 5 s
GPULOG=$!
trap 'kill $GPULOG; $TUNE restore; $ISO restore; $TUNE check | tee runs/vm3/check_end.txt
      if diff <(grep -v "^gpu" runs/vm3/check_start.txt) <(grep -v "^gpu" runs/vm3/check_end.txt); then
          echo "AS FOUND: memory and network match the start"; else echo "NOT AS FOUND: see the diff above"; fi' EXIT
$AB --out runs/vm3/gpu --primary "time/token p50 (ms)" \
    --a-setup "$TUNE gpu as-found && $ISO quiet" --b-setup "$TUNE gpu tuned && $ISO quiet"
$TUNE restore && $TUNE check
$AB --out runs/vm3/memory --primary "latency p99 (s)" \
    --a-setup "$TUNE memory as-found && $ISO quiet" --b-setup "$TUNE memory tuned && $ISO quiet"
$TUNE restore && $TUNE check
$AB --out runs/vm3/network --primary "latency p99 (s)" \
    --a-setup "$TUNE network as-found && $ISO quiet" --b-setup "$TUNE network tuned && $ISO quiet"
$TUNE restore && $TUNE check
```
lightserve restarts every round, so memory and network changes apply from a fresh start.
**A `restore` after every A/B:** the rounds go A B B A A B, so each A/B ends on a tuned round;
without it the GPU clock would stay locked through the memory and network A/Bs, and memory tuned
through the network one (found by the independent review). The `check` after it logs the as-found
values each A/B starts from.
**`gpu_samples.csv`, the GPU under load:** `tune.py`'s lock check reads the clock on an idle GPU,
which proves the lock was accepted, not that it held while serving. In the GPU A/B's tuned rounds
the clock should sit near 2,520 MHz; if the power-cap or slowdown columns say "Active" often, the
lock only partly held, and the result says so. (The field names are from NVIDIA's docs for this
driver generation; checked for real only on the node: `tune.py check` shows "status unavailable"
if one is wrong, and the GPU group still runs.)

### Results (session VM-3, 2026-11-06, 14:24-14:59 UTC, ~35 min, node stopped)

**In one line:** none of the three settings made a measurable difference, so nothing wins and no
setup code is written. A trustworthy "no difference" was the goal: every change was read back before
each round, the GPU log shows what the GPU did under load, and the machine was proven back as found.

| A/B | Verdict on the deciding measure | Change (95% range) | Prediction right? | Wins? |
|---|---|---|---|---|
| GPU clock locked at max | time/token p50: **not sure** | +0.7% (-2.6 .. +4.1) | **no** (predicted steadier and slightly faster) | no |
| Memory: huge pages "always" (no swap, so huge pages alone) | latency p99: **not sure** | +2.7% (-5.0 .. +10.4) | consistent (no difference seen; only effects over ~10% ruled out) | no |
| Network backlog 16384 | latency p99: **not sure** | +1.2% (-5.9 .. +8.3) | consistent (no difference seen; only effects over ~8% ruled out) | no |

6 rounds each (ABBAAB, 30 s warm-up + 60 s), ~205 requests a round, 0 failures. Every other measure
was "not sure" too.

- **Why the GPU lock changed nothing:** in all 343 samples of `gpu_samples.csv`, **both sides**, the
  SM clock sat at 2,500-2,599 MHz, with no power-cap, hardware or heat slowdown. Under this steady load
  the free clock already boosts to its maximum, so locking it has nothing to add. It might matter for
  light, bursty traffic (idle, the clock drops to 210 MHz): not tested.
- **`tune.py check` on the real node:** memory, network and GPU available; CPU power "can't run here";
  **no swap** (so swappiness had nothing to act on, as the prediction warned); NVIDIA's status fields
  (power, power cap, slowdowns) work on driver 570.
- **Back as found:** the exit step printed "AS FOUND: memory and network match the start"; the saved
  originals file was gone (a clean restore); the GPU idled at 210 MHz again (the reset worked).
- **Before measuring:** the tuning cache survived the node stop (first completion 2.75 s, VM-2: 13.8
  s); ~197 requests a minute in the 2-min fill. The as-found times match VM-2's quiet rounds (p99 ~5.4
  s vs 5.5 s), a cross-session sanity check.
- **Caveat, as in VM-2:** ~205 requests a round, under the runbook's 1,000: each round's p99 is set by
  its ~2 slowest requests. The steadier time/token and tokens/s measures agree ("not sure", within a
  few %).
- **Left as it was:** plain `quiet` on both sides (VM-2 showed pinning helps *with* a neighbour, not
  whether it helps without one). Raw data on the Mac in `runs/vm3/` (untracked, like the rest of `runs/`).
