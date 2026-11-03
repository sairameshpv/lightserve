"""Tests for infra/os/topology.py on fake machines: small folders shaped like Linux's own /sys and
/proc, so the GPU machine's report can be checked without the GPU machine."""
from pathlib import Path

from infra.os.topology import advise, gather, gather_gpus, parse_cpulist, parse_gpus, render, sysfs_pci_address

INTERRUPTS = """\
            CPU0       CPU1       CPU2       CPU3
   0:         45          0          0          0   IO-APIC   2-edge      timer
  24:     123456          0          0          0   PCI-MSI 524288-edge      nvidia
  25:          0       9999          0          0   PCI-MSI 1048576-edge      virtio0-input.0
 NMI:          0          0          0          0   Non-maskable interrupts
"""


def write(root: Path, rel: str, text: str):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def make_machine(root: Path, n_cpus=24, nodes=None, smt=True, governor=None, idle=None,
                 cmdline="BOOT_IMAGE=/vmlinuz root=/dev/vda1", virtual=True, irqbalance=True):
    """A fake machine. smt: cpu i and i + n/2 are twins (Linux's usual numbering on x86)."""
    nodes = nodes or {0: f"0-{n_cpus - 1}"}
    write(root, "sys/devices/system/cpu/online", f"0-{n_cpus - 1}\n")
    half = n_cpus // 2
    for c in range(n_cpus):
        twins = f"{c % half},{c % half + half}" if smt else str(c)
        write(root, f"sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list", twins + "\n")
        if governor:
            write(root, f"sys/devices/system/cpu/cpu{c}/cpufreq/scaling_governor", governor + "\n")
        for i, name in enumerate(idle or []):
            write(root, f"sys/devices/system/cpu/cpu{c}/cpuidle/state{i}/name", name + "\n")
    for n, cpus in nodes.items():
        write(root, f"sys/devices/system/node/node{n}/cpulist", cpus + "\n")
    write(root, "sys/kernel/mm/transparent_hugepage/enabled", "always [madvise] never\n")
    write(root, "sys/kernel/mm/transparent_hugepage/defrag", "always defer defer+madvise [madvise] never\n")
    write(root, "proc/sys/vm/swappiness", "60\n")
    write(root, "proc/sys/net/core/somaxconn", "4096\n")
    write(root, "proc/sys/net/core/netdev_max_backlog", "1000\n")
    write(root, "proc/cmdline", cmdline + "\n")
    flags = "fpu sse avx2" + (" hypervisor" if virtual else "")
    write(root, "proc/cpuinfo", f"processor\t: 0\nmodel name\t: AMD EPYC 9354 32-Core Processor\nflags\t\t: {flags}\n")
    write(root, "proc/interrupts", INTERRUPTS)
    write(root, "proc/irq/0/smp_affinity_list", "0\n")
    write(root, "proc/irq/24/smp_affinity_list", f"0-{n_cpus - 1}\n")
    write(root, "proc/irq/25/smp_affinity_list", "1\n")
    write(root, "proc/1/comm", "systemd\n")
    if irqbalance:
        write(root, "proc/812/comm", "irqbalance\n")


def test_parse_cpulist():
    assert parse_cpulist("0-3,8-11") == [0, 1, 2, 3, 8, 9, 10, 11]
    assert parse_cpulist("5") == [5]
    assert parse_cpulist("") == []
    assert parse_cpulist("0,2\n") == [0, 2]


def test_one_node_vm_like_the_l40s(tmp_path):
    make_machine(tmp_path)
    f = gather(tmp_path)
    assert f["cpus"] == list(range(24))
    assert f["cores"][0] == [0, 12] and len(f["cores"]) == 12
    assert f["numa"] == {0: list(range(24))}
    assert f["virtual"] is True and "EPYC" in f["cpu_model"]
    assert (f["governor"], f["idle_states"]) == (None, None)  # VMs often hide these
    assert f["thp"] == {"enabled": "madvise", "defrag": "madvise"}
    assert (f["swappiness"], f["somaxconn"], f["netdev_max_backlog"]) == (60, 4096, 1000)
    assert f["isolated"] == [] and f["irqbalance"] is True


def test_two_nodes_and_exposed_power_settings(tmp_path):
    make_machine(tmp_path, n_cpus=8, nodes={0: "0-1,4-5", 1: "2-3,6-7"}, governor="schedutil",
                 idle=["POLL", "C1", "C6"], virtual=False, irqbalance=False)
    f = gather(tmp_path)
    assert f["numa"] == {0: [0, 1, 4, 5], 1: [2, 3, 6, 7]}
    assert f["governor"] == "schedutil" and f["idle_states"] == ["POLL", "C1", "C6"]
    assert f["virtual"] is False and f["irqbalance"] is False


def test_interrupts_with_their_devices_and_cores(tmp_path):
    make_machine(tmp_path, n_cpus=4)
    irqs = {i["irq"]: i for i in gather(tmp_path)["irqs"]}
    assert set(irqs) == {0, 24, 25}  # NMI etc. aren't steerable: left out
    assert irqs[24]["device"] == "nvidia" and irqs[24]["count"] == 123456 and irqs[24]["cpus"] == [0, 1, 2, 3]
    assert irqs[25]["device"] == "virtio0-input.0" and irqs[25]["cpus"] == [1]


def test_isolcpus_from_the_kernel_command_line(tmp_path):
    make_machine(tmp_path, cmdline="ro isolcpus=managed_irq,domain,2-5,8 nohz_full=2-5")
    assert gather(tmp_path)["isolated"] == [2, 3, 4, 5, 8]


def test_missing_numa_folder_means_unknown(tmp_path):
    make_machine(tmp_path)
    import shutil
    shutil.rmtree(tmp_path / "sys/devices/system/node")
    assert gather(tmp_path)["numa"] == {}


def test_virtual_is_unknown_when_cpuinfo_has_no_flags(tmp_path):
    """ARM Linux (e.g. Docker on a Mac) lists "Features", not x86 "flags": say unknown, not False."""
    make_machine(tmp_path)
    write(tmp_path, "proc/cpuinfo", "processor\t: 0\nFeatures\t: fp asimd\n")
    assert gather(tmp_path)["virtual"] is None



# --- the GPU part ---

# nvidia-smi --query-gpu=<QUERY> --format=csv,noheader,nounits, as an L40S VM prints it
GPU_CSV = "0, NVIDIA L40S, 00000000:8D:00.0, Disabled, 210, 2520, 2520, 350.00, 350.00, 570.124.06\n"
TOPO = "\tGPU0\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID\nGPU0\t X \t0-23\t0\t\tN/A\n"


def test_parse_gpus():
    [g] = parse_gpus(GPU_CSV)
    assert g["name"] == "NVIDIA L40S" and g["pci"] == "00000000:8D:00.0"
    assert g["persistence"] is False
    assert (g["sm_clock_mhz"], g["max_sm_clock_mhz"], g["app_clock_mhz"]) == (210, 2520, 2520)
    assert (g["power_limit_w"], g["max_power_limit_w"], g["driver"]) == (350.0, 350.0, "570.124.06")


def test_parse_gpus_not_available_values_are_none():
    [g] = parse_gpus("0, NVIDIA L40S, 00000000:8D:00.0, Enabled, [N/A], 2520, [N/A], 350.00, [N/A], 570.124.06")
    assert g["persistence"] is True and g["sm_clock_mhz"] is None and g["max_power_limit_w"] is None


def test_sysfs_pci_address():
    """nvidia-smi pads the domain to 8 digits in capitals; /sys uses 4, lower case."""
    assert sysfs_pci_address("00000000:8D:00.0") == "0000:8d:00.0"


def fake_run(outputs):
    def run(cmd):
        if outputs is None:
            raise FileNotFoundError(cmd[0])
        return outputs["topo" if "topo" in cmd else "query"]
    return run


def test_gpu_with_its_numa_node_and_nearby_cores(tmp_path):
    make_machine(tmp_path, n_cpus=8, nodes={0: "0-1,4-5", 1: "2-3,6-7"})
    write(tmp_path, "sys/bus/pci/devices/0000:8d:00.0/numa_node", "1\n")
    write(tmp_path, "sys/bus/pci/devices/0000:8d:00.0/local_cpulist", "2-3,6-7\n")
    g = gather_gpus(tmp_path, fake_run({"query": GPU_CSV, "topo": TOPO}))
    assert g["available"] is True and g["topo"] == TOPO
    [gpu] = g["gpus"]
    assert (gpu["numa_node"], gpu["local_cpus"]) == (1, [2, 3, 6, 7])


def test_vm_reports_numa_node_minus_one_as_unknown(tmp_path):
    make_machine(tmp_path)
    write(tmp_path, "sys/bus/pci/devices/0000:8d:00.0/numa_node", "-1\n")
    write(tmp_path, "sys/bus/pci/devices/0000:8d:00.0/local_cpulist", "0-23\n")
    [gpu] = gather_gpus(tmp_path, fake_run({"query": GPU_CSV, "topo": TOPO}))["gpus"]
    assert gpu["numa_node"] is None and gpu["local_cpus"] == list(range(24))


def test_no_nvidia_smi(tmp_path):
    make_machine(tmp_path)
    assert gather_gpus(tmp_path, fake_run(None)) == {"available": False, "gpus": [], "topo": None,
                                                     "error": "nvidia-smi not found"}


def test_nvidia_smi_error_is_shown_not_hidden(tmp_path):
    """A wrong query field makes nvidia-smi exit non-zero: the report must show why, not just "no GPU"."""
    import subprocess

    def run(cmd):
        raise subprocess.CalledProcessError(2, cmd, stderr='Field "clocks.max.sm" is not a valid field to query.')
    g = gather_gpus(tmp_path, run)
    assert g["available"] is False and "not a valid field" in g["error"]



# --- what it means ---

def l40s_vm(tmp_path):
    make_machine(tmp_path, n_cpus=24, irqbalance=True)
    write(tmp_path, "sys/bus/pci/devices/0000:8d:00.0/numa_node", "-1\n")
    write(tmp_path, "sys/bus/pci/devices/0000:8d:00.0/local_cpulist", "0-23\n")
    return gather(tmp_path), gather_gpus(tmp_path, fake_run({"query": GPU_CSV, "topo": TOPO}))


def test_one_node_vm_numa_pinning_cant_help(tmp_path):
    a = advise(*l40s_vm(tmp_path))
    assert a["numa_nodes"] == 1 and "can't help" in a["numa"]


def test_layout_on_the_one_node_vm(tmp_path):
    """Twins are (c, c+12): whole physical cores are handed out, never half of one."""
    lay = advise(*l40s_vm(tmp_path))["layout"]
    assert lay["housekeeping"] == [0, 12]
    assert (lay["engine"], lay["engine_twin_idle"]) == (1, [13])
    assert lay["server"] == [2, 14]
    assert lay["rest"] == [c for c in range(24) if c not in (0, 12, 1, 13, 2, 14)]


def test_interrupts_on_reserved_cores_must_move_and_irqbalance_would_undo_it(tmp_path):
    a = advise(*l40s_vm(tmp_path))
    assert [(i["irq"], i["device"]) for i in a["irqs_to_move"]] == [(24, "nvidia"), (25, "virtio0-input.0")]
    assert "irqbalance" in a["irqbalance"]  # it runs: steering would be undone


def test_two_nodes_layout_goes_on_the_gpus_node(tmp_path):
    make_machine(tmp_path, n_cpus=8, nodes={0: "0-1,4-5", 1: "2-3,6-7"}, irqbalance=False)
    write(tmp_path, "sys/bus/pci/devices/0000:8d:00.0/numa_node", "1\n")
    write(tmp_path, "sys/bus/pci/devices/0000:8d:00.0/local_cpulist", "2-3,6-7\n")
    a = advise(gather(tmp_path), gather_gpus(tmp_path, fake_run({"query": GPU_CSV, "topo": TOPO})))
    assert a["numa_nodes"] == 2 and "node 1" in a["numa"]
    lay = a["layout"]
    assert (lay["housekeeping"], lay["engine"], lay["engine_twin_idle"], lay["server"], lay["rest"]) == \
        ([0, 4], 2, [6], [3, 7], [1, 5])
    assert a["irqbalance"] is None


def test_too_few_cores_for_a_layout(tmp_path):
    make_machine(tmp_path, n_cpus=2, smt=False)
    a = advise(gather(tmp_path), {"available": False, "gpus": [], "topo": None, "error": "nvidia-smi not found"})
    assert a["layout"] is None and "too few" in a["layout_why"]


def test_what_the_tuning_ab_tests_can_change(tmp_path):
    t = advise(*l40s_vm(tmp_path))["tuning"]
    assert "off" in t["gpu"] and "2520" in t["gpu"]            # persistence off; lock clocks at max
    assert "not exposed" in t["cpu_power"]                     # VM hides governor + sleep states
    assert "madvise" in t["memory"] and "60" in t["memory"]    # THP mode, swappiness
    assert "4096" in t["network"] and "1000" in t["network"]



def test_report_has_the_map_the_raw_gpu_map_and_what_it_means(tmp_path):
    facts, gpus = l40s_vm(tmp_path)
    text = render(facts, gpus, advise(facts, gpus))
    for needle in ("AMD EPYC", "24 CPUs", "NVIDIA L40S", "CPU Affinity", "can't help",
                   "engine: 1", "irqbalance is running", "persistence off", "WHAT IT MEANS"):
        assert needle in text, needle


def test_report_without_gpu_tools_shows_the_reason(tmp_path):
    make_machine(tmp_path)
    facts, gpus = gather(tmp_path), gather_gpus(tmp_path, fake_run(None))
    assert "nvidia-smi not found" in render(facts, gpus, advise(facts, gpus))
