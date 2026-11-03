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
