"""Compare cache policies on unchanged real router requests (no inference).

CSV rows: tensor offset, expert bytes, expert count, selected count, batch rows,
then selected IDs. Simulation reports useful transfer bytes, not physical I/O
or token/s. Model gates and actual timing must still qualify any chosen policy.
"""
import argparse
import collections
import csv
import heapq
import json


def read_trace(path):
    with open(path, newline="") as file:
        source = iter(csv.reader(file))
        for row in source:
            base, size, experts, selected, tokens = map(int, row[:5])
            ids = [list(map(int, row[5:]))]
            for _ in range(tokens - 1):
                next_row = next(source)
                assert list(map(int, next_row[:5])) == [base, size, experts, selected, tokens]
                ids.append(list(map(int, next_row[5:])))
            assert all(len(x) == selected for x in ids)
            yield base, size, experts, selected, tokens, ids


def simulate(requests, policy, gpu_budget, ram_budget):
    history, frequency = {}, {}
    caches = [{}, {}]
    reserved = [0, 0]
    serial = 0
    stats = collections.Counter()
    widths = collections.Counter()
    available = [{}, {}]

    def observe(base, size, experts, selected, tokens, ids):
        queue = history.setdefault(base, collections.deque(maxlen=128))
        queue.extend(ids)
        recent = list(queue)
        c32 = collections.Counter(e for row in recent[-32:] for e in row if e >= 0)
        c64 = collections.Counter(e for row in recent[-64:] for e in row if e >= 0)
        if policy in ("adaptive", "adaptive-tv"):
            if policy == "adaptive-tv":
                d = sum(abs(c32[e] - (c64[e] - c32[e])) for e in range(experts))
                total = 64 * selected
                width = 32 if len(recent) < 64 or d > total / 2 else 64 if d > total / 4 else 128
            else:
                active = sum(c64[e] > 0 for e in range(experts))
                discrepancy = sum((2 * c32[e] - c64[e]) ** 2 / c64[e] for e in range(experts) if c64[e])
                df = max(active - 1, 1)
                excess = max(discrepancy - df, 0)
                width = 32 if len(recent) < 64 or excess ** 2 > 72 * df else 64 if excess ** 2 > 18 * df else 128
        else:
            width = 32 if policy == "lru" else int(policy)
        widths[width] += 1
        count = collections.Counter(e for row in recent[-width:] for e in row if e >= 0)
        divisor = min(width, len(recent))
        for e in range(experts):
            frequency[base + e * size] = 0 if policy == "lru" else (count[e] << 20) // divisor

    def admit(tier, off, size, pinned):
        cache = caches[tier]
        budget = ram_budget if tier else gpu_budget
        capacity = size + (12288 if tier else 0)
        if reserved[tier] + capacity <= budget:
            cache[off] = [capacity, serial]
            reserved[tier] += capacity
            return True
        heaps = available[tier]
        fitting = [cap for cap, heap in heaps.items() if cap >= capacity and heap]
        if not fitting:
            return False
        chosen = min(fitting, key=lambda cap: heaps[cap][0])
        entry = heaps[chosen][0]
        victim = entry[-1]
        if not tier and frequency.get(off, 0) < frequency.get(victim, 0):
            return False
        heapq.heappop(heaps[chosen])
        cap, _ = cache.pop(victim)
        cache[off] = [cap, serial]
        stats[("ram" if tier else "gpu") + "_evictions"] += 1
        # A GPU victim still has an unchanged SSD source. If its RAM copy was
        # evicted, actual runtime demotes it; this simulator counts separately.
        if not tier and victim not in caches[1]:
            stats["demotion_candidates"] += 1
        return True

    for base, size, experts, selected, tokens, ids in requests:
        serial += 1
        observe(base, size, experts, selected, tokens, ids)
        offsets = list(dict.fromkeys(base + e * size for row in ids for e in row if 0 <= e < experts))
        pinned = set(offsets)
        # Requests in one tensor batch share an unchanged score and all current
        # offsets are pinned. Build capacity heaps once, rather than scan the
        # whole cache for each miss. New entries cannot become current victims.
        for tier in range(2):
            available[tier] = {}
            for old, (cap, last) in caches[tier].items():
                if old in pinned:
                    continue
                preferred = 0 if tier and old in caches[0] else 1
                available[tier].setdefault(cap, []).append((preferred, frequency.get(old, 0), last, old))
            for heap in available[tier].values():
                heapq.heapify(heap)
        for off in offsets:
            if off in caches[0]:
                caches[0][off][1] = serial
                stats["gpu_hits"] += 1
                continue
            stats["gpu_misses"] += 1
            stats["h2d_bytes"] += size
            if off in caches[1]:
                caches[1][off][1] = serial
                stats["ram_hits"] += 1
            else:
                stats["disk_reads"] += 1
                stats["disk_bytes"] += size
                if admit(1, off, size, pinned):
                    caches[1][off][1] = serial
            if tokens <= 8:
                admit(0, off, size, pinned)
    return dict(stats, gpu_reserved=reserved[0], ram_reserved=reserved[1], windows=dict(widths))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace")
    parser.add_argument("--gpu-mb", type=int, default=8192)
    parser.add_argument("--ram-mb", type=int, default=16384)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    requests = list(read_trace(args.trace))
    result = {policy: simulate(requests, policy, args.gpu_mb * 1048576, args.ram_mb * 1048576)
              for policy in ("lru", "32", "64", "128", "adaptive-tv", "adaptive")}
    report = dict(requests=len(requests), trace=args.trace, gpu_mb=args.gpu_mb, ram_mb=args.ram_mb,
                  limitation="Replay approximation: excludes kernel time, disk alignment and asynchronous demotions; validate winner end-to-end.", results=result)
    with open(args.output, "w") as file:
        json.dump(report, file, indent=2)
    for policy, values in result.items():
        print(policy, "H2D MiB", round(values.get("h2d_bytes", 0) / 1048576, 2),
              "disk MiB", round(values.get("disk_bytes", 0) / 1048576, 2), "windows", values["windows"], flush=True)


if __name__ == "__main__":
    main()
