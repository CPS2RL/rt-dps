# delta_eta_sweep_attack_plot_ms.py
# Fix: U=0.5, n_tasks=10; sweep delta_eta in ms:
#   [0,20,40,60,80,100,120,140,160,180,200]
# Compare Vanilla RM vs ε-RM (paper-style bounded Laplace with b = 2*J*Δη/ε).
# Output: CSV summary + PDF/PNG plot.

from __future__ import annotations
from dataclasses import dataclass
import csv
import math
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import matplotlib.pyplot as plt

# =========================
# Fixed settings
# =========================
U_TOTAL = 0.5
N_TASKS = 10
N_TRIALS = 200

DELTA_ETA_LIST_MS = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]

# Period range (ms)
PERIOD_MIN_MS = 10
PERIOD_MAX_MS = 200

HORIZON_MULT = 200

# ε-scheduler params
EPSILON = 10.0
J = 16

# Bounded Laplace interval [T_perp, T_top] in ms
T_PERP_MS = 10
T_TOP_MS = 200

# Filter tasksets that are schedulable under Vanilla RM (D=T)
FILTER_SCHEDULABLE = True
MAX_GEN_ATTEMPTS = 8000

# Output
OUT_DIR = Path("delta_eta_sweep_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Time tolerance for float comparisons
# =========================
TIME_EPS = 1e-9

# =========================
# Helpers
# =========================
def pct(num: int, den: int) -> float:
    return (100.0 * num / den) if den > 0 else 0.0

def time_equal(a: float, b: float, eps: float = TIME_EPS) -> bool:
    return abs(a - b) <= eps

# =========================
# RM response-time analysis (D=T)
# =========================
def rm_rta_schedulable(tasks_sorted_by_T: List["TaskBase"]) -> bool:
    for i, ti in enumerate(tasks_sorted_by_T):
        D = ti.T_ms  # D=T
        R = ti.C_ms
        hp = tasks_sorted_by_T[:i]

        for _ in range(10_000):
            interference = 0.0
            for tj in hp:
                interference += math.ceil(R / tj.T_ms) * tj.C_ms

            R_next = ti.C_ms + interference

            if time_equal(R_next, R):
                break
            if R_next > D + TIME_EPS:
                return False
            R = R_next

        if R > D + TIME_EPS:
            return False

    return True

# =========================
# UUniFast utilization generation
# =========================
def uunifast(n: int, U_total: float, rng: random.Random) -> List[float]:
    if not (0.0 < U_total < 1.0):
        raise ValueError("U_total must be in (0,1)")
    utils = []
    sum_u = U_total
    for i in range(1, n):
        next_sum = sum_u * (rng.random() ** (1.0 / (n - i)))
        utils.append(sum_u - next_sum)
        sum_u = next_sum
    utils.append(sum_u)
    return utils

# =========================
# Laplace sampler + bounded resampling
# =========================
def laplace_sample_0_b(rng: random.Random, b: float) -> float:
    """Sample Laplace(0,b) via inverse CDF."""
    u = rng.random() - 0.5
    return -b * math.copysign(1.0, u) * math.log(1.0 - 2.0 * abs(u))

def bounded_laplace_interarrival_ms(
    rng: random.Random,
    mu_ms: float,
    epsilon_i: float,
    J_i: int,
    delta_eta_i_ms: float,
    t_perp_ms: float,
    t_top_ms: float,
    *,
    max_tries: int = 50_000,
) -> float:
    """
    Paper-style bounded Laplace:
      b = (2 * J_i * Δη_i) / ε_i
      sample cand ~ Laplace(mu, b), resample until cand in [t_perp, t_top]
    """
    if epsilon_i <= 0:
        return min(max(mu_ms, t_perp_ms), t_top_ms)

    b_ms = (2.0 * float(J_i) * float(delta_eta_i_ms)) / float(epsilon_i)
    if b_ms <= 0:
        return min(max(mu_ms, t_perp_ms), t_top_ms)

    for _ in range(max_tries):
        noise = laplace_sample_0_b(rng, b_ms)
        cand = float(mu_ms + noise)
        if t_perp_ms <= cand <= t_top_ms:
            return cand

    return min(max(mu_ms, t_perp_ms), t_top_ms)

# =========================
# Task / Job / Segment
# =========================
@dataclass(frozen=True)
class TaskBase:
    """Task parameters that do NOT change across delta_eta sweep."""
    name: str
    C_ms: float
    T_ms: float

@dataclass(frozen=True)
class TaskEps:
    """Task parameters for epsilon scheduler (delta_eta changes)."""
    name: str
    C_ms: float
    T_ms: float
    epsilon_i: float
    J_i: int
    delta_eta_i_ms: float
    T_perp_ms: float
    T_top_ms: float
    offset_ms: float = 0.0

    def D(self) -> float:
        return self.T_ms  # D=T

@dataclass
class Job:
    task: TaskEps
    release_ms: float
    abs_deadline_ms: float
    remaining_ms: float
    job_id: int

@dataclass
class Segment:
    start_ms: float
    end_ms: float
    task_name: str
    job_id: int

def rm_pick(ready: List[Job]) -> Job:
    return min(ready, key=lambda j: (j.task.T_ms, j.abs_deadline_ms, j.release_ms, j.job_id))

# =========================
# Taskset generation (fixed across delta_eta sweep)
# =========================
def generate_taskset_base(rng: random.Random) -> List[TaskBase]:
    periods_ms = sorted(rng.sample(range(PERIOD_MIN_MS, PERIOD_MAX_MS + 1), N_TASKS))
    utils = uunifast(N_TASKS, U_TOTAL, rng)

    tasks: List[TaskBase] = []
    for i, (Tms, u) in enumerate(zip(periods_ms, utils), start=1):
        T_ms = float(Tms)

        # Preserve sub-ms execution times by rounding to 0.001 ms (1 microsecond)
        C_ms = round(u * T_ms, 3)
        C_ms = max(0.001, min(C_ms, T_ms - 0.001))

        tasks.append(TaskBase(name=f"tau{i}", C_ms=C_ms, T_ms=T_ms))

    return tasks  # sorted by T

def generate_one_schedulable_taskset(base_seed: int, trial_idx: int) -> Tuple[List[TaskBase], bool]:
    rng = random.Random(base_seed + 10_000 * trial_idx)
    last = None
    for _ in range(MAX_GEN_ATTEMPTS):
        last = generate_taskset_base(rng)
        if not FILTER_SCHEDULABLE or rm_rta_schedulable(last):
            return last, True
    return last, False

def pick_random_pair(task_names: List[str], rng: random.Random) -> Tuple[str, str]:
    a, v = rng.sample(task_names, 2)
    return a, v

def make_eps_tasks(base_tasks: List[TaskBase], delta_eta_ms: int) -> List[TaskEps]:
    return [
        TaskEps(
            name=t.name,
            C_ms=t.C_ms,
            T_ms=t.T_ms,
            epsilon_i=EPSILON,
            J_i=J,
            delta_eta_i_ms=float(delta_eta_ms),
            T_perp_ms=float(T_PERP_MS),
            T_top_ms=float(T_TOP_MS),
        )
        for t in base_tasks
    ]

# =========================
# Counting patterns + denominator
# =========================
@dataclass
class Totals:
    anterior: int = 0
    posterior: int = 0
    pincer: int = 0
    attacker_segments: int = 0
    attacker_jobs: int = 0

def simulate_count_patterns(
    tasks: List[TaskEps],
    horizon_ms: float,
    randomized: bool,
    sim_seed: int,
    attacker: str,
    victim: str,
) -> Totals:
    rng = random.Random(sim_seed)
    next_release: Dict[str, float] = {t.name: t.offset_ms for t in tasks}
    job_seq: Dict[str, int] = {t.name: 0 for t in tasks}
    ready: List[Job] = []

    totals = Totals()
    prev2: Optional[Segment] = None
    prev1: Optional[Segment] = None
    last: Optional[Segment] = None

    def reset_chain():
        nonlocal prev2, prev1, last
        prev2 = prev1 = last = None

    def release_at(time_ms: float):
        for t in tasks:
            if time_equal(next_release[t.name], time_ms):
                jid = job_seq[t.name]
                job_seq[t.name] += 1

                if t.name == attacker:
                    totals.attacker_jobs += 1

                ready.append(Job(
                    task=t,
                    release_ms=time_ms,
                    abs_deadline_ms=time_ms + t.D(),
                    remaining_ms=t.C_ms,
                    job_id=jid
                ))

                if randomized:
                    ia = bounded_laplace_interarrival_ms(
                        rng=rng,
                        mu_ms=t.T_ms,
                        epsilon_i=t.epsilon_i,
                        J_i=t.J_i,
                        delta_eta_i_ms=t.delta_eta_i_ms,
                        t_perp_ms=t.T_perp_ms,
                        t_top_ms=t.T_top_ms,
                    )
                    next_release[t.name] = time_ms + ia
                else:
                    next_release[t.name] = time_ms + t.T_ms

    def push_segment(seg: Segment):
        nonlocal prev2, prev1, last

        # merge contiguous same (task, job)
        if (
            last is not None
            and time_equal(seg.start_ms, last.end_ms)
            and seg.task_name == last.task_name
            and seg.job_id == last.job_id
        ):
            last.end_ms = seg.end_ms
            return

        prev2, prev1 = prev1, last
        last = seg

        if seg.task_name == attacker:
            totals.attacker_segments += 1

        if prev1 is not None and time_equal(prev1.end_ms, seg.start_ms):
            if prev1.task_name == attacker and seg.task_name == victim:
                totals.anterior += 1
            if prev1.task_name == victim and seg.task_name == attacker:
                totals.posterior += 1

        if prev2 is not None and prev1 is not None:
            if (
                prev2.task_name == attacker
                and prev1.task_name == victim
                and seg.task_name == attacker
                and prev2.job_id == seg.job_id
                and time_equal(prev2.end_ms, prev1.start_ms)
                and time_equal(prev1.end_ms, seg.start_ms)
            ):
                totals.pincer += 1

    # initial releases
    t_now = 0.0
    release_at(0.0)

    while t_now < horizon_ms - TIME_EPS:
        if not ready:
            t_next = min(next_release.values())
            if t_next >= horizon_ms - TIME_EPS:
                break
            reset_chain()
            t_now = t_next
            release_at(t_now)
            continue

        cur = rm_pick(ready)

        finish_time = t_now + cur.remaining_ms
        next_arrival = min(next_release.values())
        t_event = min(finish_time, next_arrival, horizon_ms)

        ran = t_event - t_now
        if ran > TIME_EPS:
            push_segment(Segment(
                start_ms=t_now,
                end_ms=t_event,
                task_name=cur.task.name,
                job_id=cur.job_id
            ))
            cur.remaining_ms -= ran
            t_now = t_event

        if time_equal(t_now, next_arrival):
            release_at(t_now)

        if cur.remaining_ms <= TIME_EPS:
            ready.remove(cur)

    return totals

# =========================
# Main sweep
# =========================
def main():
    # Choose a fresh seed each run; print it so you can reproduce later if needed
    base_seed = random.SystemRandom().randint(0, 2**63 - 1)
    print(f"base_seed = {base_seed}")

    # 1) Generate fixed 200 tasksets + fixed attacker/victim per taskset
    tasksets: List[List[TaskBase]] = []
    pairs: List[Tuple[str, str]] = []

    pair_rng = random.Random(base_seed + 999)

    used = 0
    retries = 0
    while used < N_TRIALS:
        ts, ok = generate_one_schedulable_taskset(base_seed + retries * 1_000_000, used)
        if not ok:
            retries += 1
            continue
        tasksets.append(ts)
        names = [t.name for t in ts]
        pairs.append(pick_random_pair(names, pair_rng))
        used += 1

    print(f"Generated {len(tasksets)} tasksets. (schedulable_filter={FILTER_SCHEDULABLE}, retries={retries})")

    # 2) Vanilla baseline once (independent of delta_eta)
    totals_van = Totals()
    for i in range(N_TRIALS):
        base_tasks = tasksets[i]
        attacker, victim = pairs[i]

        # Vanilla uses same task struct; delta_eta doesn't matter.
        tasks_v = make_eps_tasks(base_tasks, delta_eta_ms=190)
        Tmax_ms = max(t.T_ms for t in base_tasks)
        horizon_ms = HORIZON_MULT * Tmax_ms

        c = simulate_count_patterns(
            tasks=tasks_v,
            horizon_ms=horizon_ms,
            randomized=False,
            sim_seed=base_seed + 100_000 + i,
            attacker=attacker,
            victim=victim,
        )
        totals_van.anterior += c.anterior
        totals_van.posterior += c.posterior
        totals_van.pincer += c.pincer
        totals_van.attacker_segments += c.attacker_segments
        totals_van.attacker_jobs += c.attacker_jobs

    van_A = pct(totals_van.anterior, totals_van.attacker_segments)
    van_P = pct(totals_van.posterior, totals_van.attacker_segments)
    van_Pin = pct(totals_van.pincer, totals_van.attacker_segments)

    print("\nVanilla baseline (percent of attacker segments):")
    print(f"  Anterior={van_A:.3f}%, Posterior={van_P:.3f}%, Pincer={van_Pin:.3f}%")

    # 3) Sweep delta_eta
    rows: List[Dict[str, object]] = []
    x_vals = []
    eps_A = []
    eps_P = []
    eps_Pin = []

    for delta_eta_ms in DELTA_ETA_LIST_MS:
        totals_eps = Totals()

        for i in range(N_TRIALS):
            base_tasks = tasksets[i]
            attacker, victim = pairs[i]

            tasks_e = make_eps_tasks(base_tasks, delta_eta_ms=delta_eta_ms)
            Tmax_ms = max(t.T_ms for t in base_tasks)
            horizon_ms = HORIZON_MULT * Tmax_ms

            c = simulate_count_patterns(
                tasks=tasks_e,
                horizon_ms=horizon_ms,
                randomized=True,
                sim_seed=base_seed + 200_000 + 10_000 * delta_eta_ms + i,
                attacker=attacker,
                victim=victim,
            )
            totals_eps.anterior += c.anterior
            totals_eps.posterior += c.posterior
            totals_eps.pincer += c.pincer
            totals_eps.attacker_segments += c.attacker_segments
            totals_eps.attacker_jobs += c.attacker_jobs

        A_pct = pct(totals_eps.anterior, totals_eps.attacker_segments)
        P_pct = pct(totals_eps.posterior, totals_eps.attacker_segments)
        Pin_pct = pct(totals_eps.pincer, totals_eps.attacker_segments)

        rows.append({
            "delta_eta_ms": delta_eta_ms,
            "U_total": U_TOTAL,
            "n_tasks": N_TASKS,
            "trials": N_TRIALS,
            "epsilon": EPSILON,
            "J": J,
            "bounds_ms": f"[{T_PERP_MS},{T_TOP_MS}]",
            "horizon_mult": HORIZON_MULT,
            "vanilla_anterior_pct": van_A,
            "vanilla_posterior_pct": van_P,
            "vanilla_pincer_pct": van_Pin,
            "eps_attacker_segments": totals_eps.attacker_segments,
            "eps_attacker_jobs": totals_eps.attacker_jobs,
            "eps_anterior": totals_eps.anterior,
            "eps_posterior": totals_eps.posterior,
            "eps_pincer": totals_eps.pincer,
            "eps_anterior_pct": A_pct,
            "eps_posterior_pct": P_pct,
            "eps_pincer_pct": Pin_pct,
        })

        x_vals.append(delta_eta_ms)
        eps_A.append(A_pct)
        eps_P.append(P_pct)
        eps_Pin.append(Pin_pct)

        print(f"delta_eta={delta_eta_ms:3d} ms:  ε-RM %  A={A_pct:.3f}  P={P_pct:.3f}  Pin={Pin_pct:.3f}")

    # 4) Save CSV
    csv_path = OUT_DIR / "summary_delta_eta.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # 5) Plot
    fig = plt.figure(figsize=(7.2, 4.2), dpi=200)
    ax = fig.add_subplot(111)

    ax.plot(x_vals, eps_A, marker="o", label="ε-RM Anterior (%)")
    ax.plot(x_vals, eps_P, marker="o", label="ε-RM Posterior (%)")
    ax.plot(x_vals, eps_Pin, marker="o", label="ε-RM Pincer (%)")

    # Vanilla as horizontal reference lines
    ax.axhline(van_A, linestyle="--", label="Vanilla Anterior (%)")
    ax.axhline(van_P, linestyle="--", label="Vanilla Posterior (%)")
    ax.axhline(van_Pin, linestyle="--", label="Vanilla Pincer (%)")

    ax.set_xlabel(r"$\Delta\eta$ (ms)")
    ax.set_ylabel("Detected instances / attacker segments (%)")
    ax.set_xticks(x_vals)
    ax.grid(True, which="both", axis="both", linestyle=":", linewidth=0.7)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    pdf_path = OUT_DIR / "delta_eta_sweep_plot.pdf"
    png_path = OUT_DIR / "delta_eta_sweep_plot.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)

    print("\nSaved:")
    print(f"  {csv_path}")
    print(f"  {pdf_path}")
    print(f"  {png_path}")

if __name__ == "__main__":
    main()