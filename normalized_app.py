import math
import random
import secrets
from dataclasses import dataclass
from collections import Counter
from typing import List, Dict, Tuple, Optional, Iterable
import pandas as pd

# =========================
# Utilities
# =========================
def lcm(a: int, b: int) -> int:
    return abs(a // math.gcd(a, b) * b)

def lcm_list(xs: Iterable[int]) -> int:
    hp = 1
    for x in xs:
        hp = lcm(hp, int(x))
    return int(hp)

def laplace_noise(rng: random.Random, b: float) -> float:
    u = rng.random() - 0.5
    if u == 0.0:
        return 0.0
    return -b * math.copysign(1.0, u) * math.log(1.0 - 2.0 * abs(u))

def bounded_laplace_interarrival_ms(
    rng: random.Random,
    mu_ms: int,
    epsilon: float,
    J: float,
    delta_eta_ms: int,
    t_perp_ms: int,
    t_top_ms: int,
) -> int:
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    b = (2.0 * float(J) * float(delta_eta_ms)) / float(epsilon)
    x = float(mu_ms) + laplace_noise(rng, b)
    ia = int(round(x))
    ia = max(1, ia)
    ia = max(t_perp_ms, min(ia, t_top_ms))
    return ia

# =========================
# UUniFast
# =========================
def uunifast(n_tasks: int, U_total: float, rng: random.Random) -> List[float]:
    """
    Generate n_tasks utilizations summing to U_total using UUniFast.
    """
    utils: List[float] = []
    sum_u = float(U_total)

    for i in range(1, n_tasks):
        next_sum_u = sum_u * (rng.random() ** (1.0 / (n_tasks - i)))
        utils.append(sum_u - next_sum_u)
        sum_u = next_sum_u

    utils.append(sum_u)
    return utils

# =========================
# Task / Job / Segment
# =========================
@dataclass(frozen=True)
class Task:
    name: str
    C_ms: int
    T_ms: int
    D_ms: Optional[int] = None
    offset_ms: int = 0
    epsilon_i: float = 100.0
    J_i: float = 16.0
    delta_eta_ms: int = 100
    T_perp_ms: Optional[int] = None
    T_top_ms: Optional[int] = None

    def D(self) -> int:
        return self.D_ms if self.D_ms is not None else self.T_ms

@dataclass
class Job:
    task: Task
    release_ms: int
    abs_deadline_ms: int
    remaining_ms: int
    job_id: int

@dataclass
class Segment:
    start_ms: int
    end_ms: int
    task_name: str
    job_id: int

@dataclass
class Totals:
    anterior: int = 0
    posterior: int = 0
    pincer: int = 0
    attacker_segments: int = 0
    attacker_jobs: int = 0
    victim_jobs: int = 0          # NEW (for N = Jv + Jt)

@dataclass
class TraceResult:
    hp_used_ms: int
    hp_true_ms: int
    hp_was_capped: bool
    num_windows: int
    unique_traces: int
    all_windows_match: bool
    trace_freq: Counter
    totals: Totals

# =========================
# RM priority selection
# =========================
def rm_pick(ready: List[Job]) -> Job:
    return min(ready, key=lambda j: (j.task.T_ms, j.release_ms, j.task.name, j.job_id))

# =========================
# Taskset generation (UUniFast)
# =========================
def generate_taskset(
    *,
    n_tasks: int,
    U_total: float,
    seed: int,
    period_pool_ms: Optional[List[int]] = None,
    period_min_ms: int = 50,
    period_max_ms: int = 150,
    epsilon: float = 1.0,
    J: float = 16.0,
    delta_eta_ms: int = 100,
) -> List[Task]:
    rng = random.Random(seed)

    # 1) Pick periods
    if period_pool_ms is None:
        Ts = [rng.randint(period_min_ms, period_max_ms) for _ in range(n_tasks)]
    else:
        Ts = [rng.choice(period_pool_ms) for _ in range(n_tasks)]

    # 2) Generate utilizations via UUniFast
    utils = uunifast(n_tasks, U_total, rng)
    rng.shuffle(utils)

    # 3) Build tasks
    tasks: List[Task] = []
    for i, (T, u) in enumerate(zip(Ts, utils), start=1):
        C = max(1, int(math.ceil(u * T)))
        if C >= T:
            C = max(1, T - 1)

        t_perp = max(1, T - delta_eta_ms)
        t_top = T + delta_eta_ms

        tasks.append(Task(
            name=f"tau_{i}",
            C_ms=int(C),
            T_ms=int(T),
            D_ms=int(T),
            offset_ms=0,
            epsilon_i=float(epsilon),
            J_i=float(J),
            delta_eta_ms=int(delta_eta_ms),
            T_perp_ms=int(t_perp),
            T_top_ms=int(t_top),
        ))

    return tasks

# =========================
# Simulation
# =========================
def simulate_trace_windows_and_patterns(
    tasks: List[Task],
    *,
    randomized: bool,
    seed: int,
    attacker: str,
    victim: str,
    num_hp_windows: int = 200,
    hp_cap_ms: Optional[int] = 2000,
    signature_mode: str = "order_only",
    include_idle_in_trace: bool = True,
) -> TraceResult:
    rng = random.Random(seed)

    hp_true_ms = lcm_list([t.T_ms for t in tasks])
    hp_used_ms = hp_true_ms
    hp_was_capped = False
    if hp_cap_ms is not None and hp_true_ms > hp_cap_ms:
        hp_used_ms = int(hp_cap_ms)
        hp_was_capped = True

    horizon_ms = hp_used_ms * num_hp_windows
    win_raw: List[List[Tuple[str, int]]] = [[] for _ in range(num_hp_windows)]

    def add_to_windows(start_ms: int, end_ms: int, task_name: str) -> None:
        if end_ms <= start_ms:
            return
        start_ms = max(0, start_ms)
        end_ms = min(horizon_ms, end_ms)
        cur = start_ms
        while cur < end_ms:
            w = cur // hp_used_ms
            if w >= num_hp_windows:
                break
            w_end = min(end_ms, (w + 1) * hp_used_ms)
            dur = int(w_end - cur)
            if dur > 0:
                win_raw[int(w)].append((task_name, dur))
            cur = w_end

    next_release: Dict[str, int] = {t.name: t.offset_ms for t in tasks}
    job_seq: Dict[str, int] = {t.name: 0 for t in tasks}
    ready: List[Job] = []
    totals = Totals()

    prev2: Optional[Segment] = None
    prev1: Optional[Segment] = None
    last: Optional[Segment] = None

    def reset_chain() -> None:
        nonlocal prev2, prev1, last
        prev2 = prev1 = last = None

    def release_at(time_ms: int) -> None:
        for t in tasks:
            if next_release[t.name] == time_ms:
                jid = job_seq[t.name]
                job_seq[t.name] += 1

                if t.name == attacker:
                    totals.attacker_jobs += 1
                if t.name == victim:
                    totals.victim_jobs += 1   # NEW

                ready.append(Job(
                    task=t,
                    release_ms=time_ms,
                    abs_deadline_ms=time_ms + t.D(),
                    remaining_ms=t.C_ms,
                    job_id=jid
                ))

                if randomized:
                    t_perp = t.T_perp_ms if t.T_perp_ms is not None else max(1, t.T_ms - t.delta_eta_ms)
                    t_top = t.T_top_ms if t.T_top_ms is not None else (t.T_ms + t.delta_eta_ms)
                    ia = bounded_laplace_interarrival_ms(
                        rng=rng,
                        mu_ms=t.T_ms,
                        epsilon=t.epsilon_i,
                        J=t.J_i,
                        delta_eta_ms=t.delta_eta_ms,
                        t_perp_ms=int(t_perp),
                        t_top_ms=int(t_top),
                    )
                    next_release[t.name] = time_ms + ia
                else:
                    next_release[t.name] = time_ms + t.T_ms

    def push_segment(seg: Segment) -> None:
        nonlocal prev2, prev1, last

        add_to_windows(seg.start_ms, seg.end_ms, seg.task_name)

        if last is not None and seg.start_ms == last.end_ms and seg.task_name == last.task_name and seg.job_id == last.job_id:
            last.end_ms = seg.end_ms
            return

        prev2, prev1 = prev1, last
        last = seg

        if seg.task_name == attacker:
            totals.attacker_segments += 1

        if prev1 is not None and prev1.end_ms == seg.start_ms:
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
                and prev2.end_ms == prev1.start_ms
                and prev1.end_ms == seg.start_ms
            ):
                totals.pincer += 1

    t_now = 0
    release_at(0)

    while t_now < horizon_ms:
        if not ready:
            t_next = min(next_release.values())
            if t_next >= horizon_ms:
                if include_idle_in_trace and t_now < horizon_ms:
                    add_to_windows(t_now, horizon_ms, "IDLE")
                break

            if include_idle_in_trace:
                add_to_windows(t_now, t_next, "IDLE")

            reset_chain()
            t_now = t_next
            release_at(t_now)
            continue

        cur = rm_pick(ready)
        finish_time = t_now + cur.remaining_ms
        next_arrival = min(next_release.values())
        t_event = min(finish_time, next_arrival, horizon_ms)

        ran = t_event - t_now
        if ran > 0:
            push_segment(Segment(t_now, t_event, cur.task.name, cur.job_id))
            cur.remaining_ms -= ran
            t_now = t_event

        if t_now == next_arrival:
            release_at(t_now)

        if cur.remaining_ms == 0:
            ready.remove(cur)

    signatures: List[Tuple] = []
    for w in range(num_hp_windows):
        merged: List[List] = []
        for name, dur in win_raw[w]:
            if dur <= 0:
                continue
            if merged and merged[-1][0] == name:
                merged[-1][1] += dur
            else:
                merged.append([name, dur])

        if signature_mode == "order_only":
            signatures.append(tuple(x[0] for x in merged))
        elif signature_mode == "order+dur":
            signatures.append(tuple((x[0], int(x[1])) for x in merged))
        else:
            raise ValueError("signature_mode must be 'order_only' or 'order+dur'")

    freq = Counter(signatures)
    unique = len(freq)

    return TraceResult(
        hp_used_ms=hp_used_ms,
        hp_true_ms=hp_true_ms,
        hp_was_capped=hp_was_capped,
        num_windows=num_hp_windows,
        unique_traces=unique,
        all_windows_match=(unique == 1),
        trace_freq=freq,
        totals=totals,
    )

# =========================
# One run
# =========================
def one_run(
    *,
    n_tasks: int = 15,
    U_total: float = 0.5,
    num_hp_windows: int = 200,
    hp_cap_ms: Optional[int] = 2000,
    period_pool_ms: Optional[List[int]] = (50,60,70,80,90,100,110,120,130,140,150),
    epsilon: float = 1.0,
    J: float = 16.0,
    delta_eta_ms: int = 100,
    seed: Optional[int] = None,
    max_regen_attempts: int = 1000,
):
    if seed is None:
        seed = secrets.randbits(64)

    pair_rng = random.Random(seed ^ 0x5A5A5A5A5A5A5A5A)

    tasks = None
    attacker = None
    victim = None
    attempt = 0

    while attempt < max_regen_attempts:
        cur_seed = seed ^ (0xA5A5A5A5A5A5A5A5 + attempt)

        tasks = generate_taskset(
            n_tasks=n_tasks,
            U_total=U_total,
            seed=cur_seed,
            period_pool_ms=list(period_pool_ms) if period_pool_ms is not None else None,
            epsilon=epsilon,
            J=J,
            delta_eta_ms=delta_eta_ms,
        )

        names = [t.name for t in tasks]
        attacker, victim = pair_rng.sample(names, 2)

        attempt += 1

    if tasks is None or attacker is None or victim is None:
        raise ValueError("Could not generate a taskset with attacker higher priority than victim.")

    attacker_T = next(t.T_ms for t in tasks if t.name == attacker)
    victim_T = next(t.T_ms for t in tasks if t.name == victim)
    actual_U = sum(t.C_ms / t.T_ms for t in tasks)

    rm_res = simulate_trace_windows_and_patterns(
        tasks,
        randomized=False,
        seed=seed ^ 0x1111111111111111,
        attacker=attacker,
        victim=victim,
        num_hp_windows=num_hp_windows,
        hp_cap_ms=hp_cap_ms,
        signature_mode="order_only",
        include_idle_in_trace=True,
    )

    eps_res = simulate_trace_windows_and_patterns(
        tasks,
        randomized=True,
        seed=seed ^ 0x2222222222222222,
        attacker=attacker,
        victim=victim,
        num_hp_windows=num_hp_windows,
        hp_cap_ms=hp_cap_ms,
        signature_mode="order_only",
        include_idle_in_trace=True,
    )

    # NEW METRICS: N and normalized anterior ratios
    N_RM = int(rm_res.totals.attacker_jobs + rm_res.totals.victim_jobs)
    N_DPS = int(eps_res.totals.attacker_jobs + eps_res.totals.victim_jobs)

    RM_Anterior_ratio = (rm_res.totals.anterior / N_RM) if N_RM > 0 else 0.0
    DPS_Anterior_ratio = (eps_res.totals.anterior / N_DPS) if N_DPS > 0 else 0.0

    return {
        "seed": int(seed),
        "attacker": attacker,
        "victim": victim,
        "attacker_T_ms": int(attacker_T),
        "victim_T_ms": int(victim_T),
        "attacker_HP_than_victim": int(attacker_T < victim_T),
        "actual_U": float(actual_U),

        "RM_Anterior": int(rm_res.totals.anterior),
        "RM_Posterior": int(rm_res.totals.posterior),
        "RM_Pincer": int(rm_res.totals.pincer),
        "RM_Unique_Traces": int(rm_res.unique_traces),

        "DPS_Anterior": int(eps_res.totals.anterior),
        "DPS_Posterior": int(eps_res.totals.posterior),
        "DPS_Pincer": int(eps_res.totals.pincer),
        "DPS_Unique_Traces": int(eps_res.unique_traces),

        "RM_N_jobs": int(N_RM),                         # NEW
        "DPS_N_jobs": int(N_DPS),                       # NEW
        "RM_Anterior_ratio": float(RM_Anterior_ratio),  # NEW
        "DPS_Anterior_ratio": float(DPS_Anterior_ratio),# NEW

        "HP_true_ms": int(rm_res.hp_true_ms),
        "HP_used_ms": int(rm_res.hp_used_ms),
        "HP_capped": bool(rm_res.hp_was_capped),
    }

# =========================
# Main experiment:
# U = 0.1 to 0.8, 200 runs each
# =========================
if __name__ == "__main__":
    U_VALUES = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
    RUNS_PER_U = 200

    all_rows = []
    summary_rows = []

    for U in U_VALUES:
        rows_u = []

        for run_id in range(1, RUNS_PER_U + 1):
            r = one_run(
                U_total=U,
                n_tasks=15,
                num_hp_windows=200,
                hp_cap_ms=2000,
                period_pool_ms=[50,60,70,80,90,100,110,120,130,140,150],
                epsilon=10.0,
                J=16.0,
                delta_eta_ms=190,
            )
            r["run"] = run_id
            r["U_total"] = U
            rows_u.append(r)
            all_rows.append(r)

        df_u = pd.DataFrame(rows_u)

        # NEW: only consider attack cases for ratio averaging (but keep all rows stored)
        attack_mask = (df_u["RM_Anterior"] > 0) | (df_u["DPS_Anterior"] > 0)
        df_u_attack = df_u.loc[attack_mask]

        summary_rows.append({
            "U_total": U,
            "runs": RUNS_PER_U,

            "RM_Anterior_incidence_percent": 100.0 * (df_u["RM_Anterior"] > 0).mean(),
            "RM_Posterior_incidence_percent": 100.0 * (df_u["RM_Posterior"] > 0).mean(),
            "RM_Pincer_incidence_percent": 100.0 * (df_u["RM_Pincer"] > 0).mean(),

            "DPS_Anterior_incidence_percent": 100.0 * (df_u["DPS_Anterior"] > 0).mean(),
            "DPS_Posterior_incidence_percent": 100.0 * (df_u["DPS_Posterior"] > 0).mean(),
            "DPS_Pincer_incidence_percent": 100.0 * (df_u["DPS_Pincer"] > 0).mean(),

            "RM_Anterior_mean_count": df_u["RM_Anterior"].mean(),
            "RM_Posterior_mean_count": df_u["RM_Posterior"].mean(),
            "RM_Pincer_mean_count": df_u["RM_Pincer"].mean(),

            "DPS_Anterior_mean_count": df_u["DPS_Anterior"].mean(),
            "DPS_Posterior_mean_count": df_u["DPS_Posterior"].mean(),
            "DPS_Pincer_mean_count": df_u["DPS_Pincer"].mean(),

            # NEW METRIC RESULTS (avg X over attack cases only)
            "attack_cases_used_for_ratio": int(attack_mask.sum()),
            "RM_Anterior_ratio_mean_attack_cases": float(df_u_attack["RM_Anterior_ratio"].mean()) if len(df_u_attack) else 0.0,
            "DPS_Anterior_ratio_mean_attack_cases": float(df_u_attack["DPS_Anterior_ratio"].mean()) if len(df_u_attack) else 0.0,

            "RM_Unique_Traces_mean": df_u["RM_Unique_Traces"].mean(),
            "DPS_Unique_Traces_mean": df_u["DPS_Unique_Traces"].mean(),

            "actual_U_mean": df_u["actual_U"].mean(),
        })

    df_all = pd.DataFrame(all_rows)
    df_summary = pd.DataFrame(summary_rows)

    print("\n==================== Summary by Utilization ====================")
    print(df_summary.to_string(index=False))

    out_xlsx = "new_exp_n15.xlsx"

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df_all.to_excel(writer, sheet_name="all_runs", index=False)
        df_summary.to_excel(writer, sheet_name="summary_by_U", index=False)
