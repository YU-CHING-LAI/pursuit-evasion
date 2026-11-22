import time
import tracemalloc
import numpy as np
import matplotlib.pyplot as plt
from env import PursuitEvasionGrid
from solver import value_iteration
import csv, os

sizes = [10, 15, 20, 25, 30, 35, 40]

# scale the wall wrt n
def make_env(n): 
    mid = n // 2
    wall = [(mid, c) for c in range(n // 3)]
    return PursuitEvasionGrid(n=n, obstacles=wall, seed=0)

# test value iteration
results = []
for n in sizes:

    # time
    times = []
    for _ in range(3):
        env = make_env(n)
        t0 = time.perf_counter()
        V0, V1, iters = value_iteration(env) 
        times.append(time.perf_counter() - t0)
    elapsed = np.median(times)

    # memory
    env = make_env(n)
    tracemalloc.start()
    value_iteration(env)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    free = np.flatnonzero(~env.blocked) 
    sub = V0[np.ix_(free, free)] # both pursuer and evader are not on obstacle

    results.append({
        "n": n,
        "states": 2 * (n * n) ** 2,
        "iterations": iters,
        "time_s": float(elapsed),
        "peak_mb": peak / 1e6,
        "max_steps": int(sub[np.isfinite(sub)].max()), # exclude infinite
    })
    print(results[-1])

# plots
ns = [r["n"] for r in results] # turn all size n to the lables on axis
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(ns, [r["time_s"] for r in results], "o-")
axes[0].set_xlabel("grid size n")
axes[0].set_ylabel("time (s)")
axes[0].set_xscale("log")
axes[0].set_yscale("log")

times = np.array([r["time_s"] for r in results])
ns_arr = np.array(ns)
ref_time = times[-1] * (ns_arr / ns_arr[-1]) ** 5     
axes[0].loglog(ns_arr, ref_time, "--", label=r"$O(n^5)$")
axes[0].legend()

axes[1].plot(ns, [r["peak_mb"] for r in results], "o-")
axes[1].set_xlabel("grid size n")
axes[1].set_ylabel("peak memory (MB)")
axes[1].set_xscale("log")
axes[1].set_yscale("log")

mems = np.array([r["peak_mb"] for r in results])
ref_mem = mems[-1] * (ns_arr / ns_arr[-1]) ** 4     
axes[1].loglog(ns_arr, ref_mem, "--", label=r"$O(n^4)$")
axes[1].legend()

for ax in axes:
    ax.set_xticks(ns)
    ax.set_xticklabels(ns)
    ax.minorticks_off()

fig.tight_layout()
fig.savefig("outputs/scaling.png", dpi=150)

# calculate slopes
mask = ns_arr >= 20
t_slope = np.polyfit(np.log(ns_arr[mask]), np.log(times[mask]), 1)[0]
m_slope = np.polyfit(np.log(ns_arr), np.log(mems), 1)[0]
print(f"time slope (n>=20): {t_slope:.2f}")
print(f"memory slope: {m_slope:.2f}")

# save result as csv
with open("outputs/scaling.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
