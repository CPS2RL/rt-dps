import math
import random
import secrets
from dataclasses import dataclass
from typing import List, Optional, Iterable
import pandas as pd


# =========================================================
# Utilities
# =========================================================
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


# =========================================================
# UUniFast
# =========================================================
def uunifast(n_tasks: int, U_total: float, rng: random.Random) -> List[float]:
    """
    Generate n_tasks utilizations that sum to U_total using UUniFast.
    """
    utils: List[float] = []
    sum_u = float(U_total)

    for i in range(1, n_tasks):
        next_sum_u = sum_u * (rng.random() ** (1.0 / (n_tasks - i)))
        utils.append(sum_u - next_sum_u)
        sum_u = next_sum_u

    utils.append(sum_u)
    return utils


# =========================================================
# Data structures
# =========================================================
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
class StabilityResult:
    failed: bool
    Rb_ms: float
    Rw_ms: float
    Jitter_ms: float
    lhs: float
    rhs: float


# =========================================================
# RM priority
# =========================================================
def rm_pick(ready: List[Job]) -> Job:
    return min(ready, key=lambda j: (j.task.T_ms, j.release_ms, j.task.name, j.job_id))


# =========================================================
# Taskset generation
# =========================================================
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

    if period_pool_ms is None:
        Ts = [rng.randint(period_min_ms, period_max_ms) for _ in range(n_tasks)]
    else:
        Ts = [rng.choice(period_pool_ms) for _ in range(n_tasks)]

    # Real UUniFast split
    utils = uunifast(n_tasks, U_total, rng)
    rng.shuffle(utils)  # optional: avoid bias from generation order

    tasks: List[Task] = []
    for i, (T, u) in enumerate(zip(Ts, utils), start=1):
        C = max(1, int(math.ceil(u * T)))
        if C >= T:
            C = max(1, T - 1)

        t_perp = max(1, T - delta_eta_ms)
        t_top = T + delta_eta_ms

        tasks.append(
            Task(
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
            )
        )
    return tasks


# =========================================================
# Simulate target response times
# =========================================================
def simulate_target_response_times(
    tasks: List[Task],
    *,
    target_task_name: str,
    randomized: bool,
    seed: int,
    num_hp_windows: int = 200,
    hp_cap_ms: Optional[int] = 2000,
) -> List[int]:
    rng = random.Random(seed)

    hp_true_ms = lcm_list([t.T_ms for t in tasks])
    hp_used_ms = hp_true_ms
    if hp_cap_ms is not None and hp_true_ms > hp_cap_ms:
        hp_used_ms = int(hp_cap_ms)

    horizon_ms = hp_used_ms * num_hp_windows

    next_release = {t.name: t.offset_ms for t in tasks}
    job_seq = {t.name: 0 for t in tasks}
    ready: List[Job] = []

    target_response_times: List[int] = []

    def release_at(time_ms: int) -> None:
        for t in tasks:
            if next_release[t.name] == time_ms:
                jid = job_seq[t.name]
                job_seq[t.name] += 1

                ready.append(
                    Job(
                        task=t,
                        release_ms=time_ms,
                        abs_deadline_ms=time_ms + t.D(),
                        remaining_ms=t.C_ms,
                        job_id=jid,
                    )
                )

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

    t_now = 0
    release_at(0)

    while t_now < horizon_ms:
        if not ready:
            t_next = min(next_release.values())
            if t_next >= horizon_ms:
                break
            t_now = t_next
            release_at(t_now)
            continue

        cur = rm_pick(ready)
        finish_time = t_now + cur.remaining_ms
        next_arrival = min(next_release.values())
        t_event = min(finish_time, next_arrival, horizon_ms)

        ran = t_event - t_now
        if ran > 0:
            cur.remaining_ms -= ran
            t_now = t_event

        if t_now == next_arrival:
            release_at(t_now)

        if cur.remaining_ms == 0:
            if cur.task.name == target_task_name:
                rt = t_now - cur.release_ms
                target_response_times.append(rt)
            ready.remove(cur)

    return target_response_times


# =========================================================
# Stability rule
# =========================================================
def evaluate_stability_from_response_times(
    response_times: List[int],
    *,
    alpha: float,
    beta: float,
) -> StabilityResult:
    if len(response_times) == 0:
        return StabilityResult(
            failed=True,
            Rb_ms=float("nan"),
            Rw_ms=float("nan"),
            Jitter_ms=float("nan"),
            lhs=float("nan"),
            rhs=beta,
        )

    rb = float(min(response_times))
    rw = float(max(response_times))
    jitter = rw - rb
    lhs = rb + alpha * jitter
    failed = lhs > beta

    return StabilityResult(
        failed=failed,
        Rb_ms=rb,
        Rw_ms=rw,
        Jitter_ms=jitter,
        lhs=lhs,
        rhs=beta,
    )


# =========================================================
# One taskset trial
# =========================================================
def one_taskset_trial(
    *,
    n_tasks: int = 15,
    U_total: float = 0.5,
    period_pool_ms: Optional[List[int]] = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150),
    epsilon: float = 1.0,
    J: float = 16.0,
    delta_eta_ms: int = 100,
    alpha: float = 1.0,
    beta: float = 20.83,
    num_hp_windows: int = 200,
    hp_cap_ms: Optional[int] = 2000,
    seed: Optional[int] = None,
):
    if seed is None:
        seed = secrets.randbits(64)

    tasks = generate_taskset(
        n_tasks=n_tasks,
        U_total=U_total,
        seed=seed ^ 0xA5A5A5A5A5A5A5A5,
        period_pool_ms=list(period_pool_ms) if period_pool_ms is not None else None,
        epsilon=epsilon,
        J=J,
        delta_eta_ms=delta_eta_ms,
    )

    names = [t.name for t in tasks]
    pick_rng = random.Random(seed ^ 0x5A5A5A5A5A5A5A5A)
    target_task = pick_rng.choice(names)

    rm_rts = simulate_target_response_times(
        tasks,
        target_task_name=target_task,
        randomized=False,
        seed=seed ^ 0x1111111111111111,
        num_hp_windows=num_hp_windows,
        hp_cap_ms=hp_cap_ms,
    )
    rm_res = evaluate_stability_from_response_times(
        rm_rts,
        alpha=alpha,
        beta=beta,
    )

    eps_rts = simulate_target_response_times(
        tasks,
        target_task_name=target_task,
        randomized=True,
        seed=seed ^ 0x2222222222222222,
        num_hp_windows=num_hp_windows,
        hp_cap_ms=hp_cap_ms,
    )
    eps_res = evaluate_stability_from_response_times(
        eps_rts,
        alpha=alpha,
        beta=beta,
    )

    vanilla_fail = int(rm_res.failed)
    eps_fail = int(eps_res.failed)

    return {
        "seed": int(seed),
        "U": float(U_total),
        "target_task": target_task,

        "Vanilla_failed": vanilla_fail,
        "EPS_failed": eps_fail,

        "Vanilla_only": int(vanilla_fail == 1 and eps_fail == 0),
        "EPS_only": int(vanilla_fail == 0 and eps_fail == 1),
        "Both": int(vanilla_fail == 1 and eps_fail == 1),
        "Neither": int(vanilla_fail == 0 and eps_fail == 0),

        "Vanilla_Rb_ms": rm_res.Rb_ms,
        "Vanilla_Rw_ms": rm_res.Rw_ms,
        "Vanilla_Jitter_ms": rm_res.Jitter_ms,
        "Vanilla_LHS": rm_res.lhs,
        "Vanilla_RHS": rm_res.rhs,

        "EPS_Rb_ms": eps_res.Rb_ms,
        "EPS_Rw_ms": eps_res.Rw_ms,
        "EPS_Jitter_ms": eps_res.Jitter_ms,
        "EPS_LHS": eps_res.lhs,
        "EPS_RHS": eps_res.rhs,
    }


# =========================================================
# Sweep over utilizations
# =========================================================
def run_utilization_sweep(
    *,
    U_values,
    n_runs_per_u: int = 200,
    n_tasks: int = 15,
    period_pool_ms: Optional[List[int]] = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150),
    epsilon: float = 10.0,
    J: float = 16.0,
    delta_eta_ms: int = 190,
    alpha: float = 1.20,
    beta: float = 20.83,
    num_hp_windows: int = 200,
    hp_cap_ms: Optional[int] = 2000,
):
    all_rows = []
    summary_rows = []

    global_run_id = 1

    for U_total in U_values:
        rows_u = []

        for local_run_id in range(1, n_runs_per_u + 1):
            res = one_taskset_trial(
                n_tasks=n_tasks,
                U_total=U_total,
                period_pool_ms=period_pool_ms,
                epsilon=epsilon,
                J=J,
                delta_eta_ms=delta_eta_ms,
                alpha=alpha,
                beta=beta,
                num_hp_windows=num_hp_windows,
                hp_cap_ms=hp_cap_ms,
            )
            res["run_global"] = global_run_id
            res["run_within_U"] = local_run_id
            global_run_id += 1

            rows_u.append(res)
            all_rows.append(res)

        df_u = pd.DataFrame(rows_u)

        summary_rows.append({
            "U": float(U_total),
            "n_runs": int(n_runs_per_u),

            "Vanilla_total_violations": int(df_u["Vanilla_failed"].sum()),
            "EPS_total_violations": int(df_u["EPS_failed"].sum()),

            "Vanilla_incidence_percent": 100.0 * df_u["Vanilla_failed"].mean(),
            "EPS_incidence_percent": 100.0 * df_u["EPS_failed"].mean(),

            "Vanilla_only_count": int(df_u["Vanilla_only"].sum()),
            "EPS_only_count": int(df_u["EPS_only"].sum()),
            "Both_count": int(df_u["Both"].sum()),
            "Neither_count": int(df_u["Neither"].sum()),

            "Vanilla_only_percent": 100.0 * df_u["Vanilla_only"].mean(),
            "EPS_only_percent": 100.0 * df_u["EPS_only"].mean(),
            "Both_percent": 100.0 * df_u["Both"].mean(),
            "Neither_percent": 100.0 * df_u["Neither"].mean(),

            "Vanilla_avg_Rb_ms": df_u["Vanilla_Rb_ms"].mean(),
            "Vanilla_avg_Rw_ms": df_u["Vanilla_Rw_ms"].mean(),
            "Vanilla_avg_Jitter_ms": df_u["Vanilla_Jitter_ms"].mean(),

            "EPS_avg_Rb_ms": df_u["EPS_Rb_ms"].mean(),
            "EPS_avg_Rw_ms": df_u["EPS_Rw_ms"].mean(),
            "EPS_avg_Jitter_ms": df_u["EPS_Jitter_ms"].mean(),
        })

    df_all = pd.DataFrame(all_rows)
    df_summary = pd.DataFrame(summary_rows)

    grand_total = pd.DataFrame([{
        "total_tasksets": int(len(df_all)),
        "Vanilla_total_violations": int(df_all["Vanilla_failed"].sum()),
        "EPS_total_violations": int(df_all["EPS_failed"].sum()),

        "Vanilla_incidence_percent": 100.0 * df_all["Vanilla_failed"].mean(),
        "EPS_incidence_percent": 100.0 * df_all["EPS_failed"].mean(),

        "Vanilla_only_count": int(df_all["Vanilla_only"].sum()),
        "EPS_only_count": int(df_all["EPS_only"].sum()),
        "Both_count": int(df_all["Both"].sum()),
        "Neither_count": int(df_all["Neither"].sum()),

        "Vanilla_only_percent": 100.0 * df_all["Vanilla_only"].mean(),
        "EPS_only_percent": 100.0 * df_all["EPS_only"].mean(),
        "Both_percent": 100.0 * df_all["Both"].mean(),
        "Neither_percent": 100.0 * df_all["Neither"].mean(),
    }])

    return df_all, df_summary, grand_total


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    U_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    df_all, df_summary, df_total = run_utilization_sweep(
        U_values=U_values,
        n_runs_per_u=200,
        n_tasks=15,
        period_pool_ms=[50,60,70,80,90,100,110,120,130,140,150],
        epsilon=10.0,
        J=16.0,
        delta_eta_ms=190,
        alpha=1.20,
        beta=20.83,
        num_hp_windows=200,
        hp_cap_ms=2000,
    )

    print("\n==================== Per-U Summary ====================")
    print(df_summary.to_string(index=False))

    print("\n==================== Grand Total ====================")
    print(df_total.to_string(index=False))

    out_xlsx = "stability_violation_n15.xlsx"
    # out_runs_csv = "stability_violation_all_runs.csv"
    # out_summary_csv = "stability_violation_summary_by_U.csv"
    # out_total_csv = "stability_violation_grand_total.csv"

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df_all.to_excel(writer, sheet_name="all_runs", index=False)
        df_summary.to_excel(writer, sheet_name="summary_by_U", index=False)
        df_total.to_excel(writer, sheet_name="grand_total", index=False)

    # df_all.to_csv(out_runs_csv, index=False)
    # df_summary.to_csv(out_summary_csv, index=False)
    # df_total.to_csv(out_total_csv, index=False)

    print("\nSaved files:")
    print(f"  {out_xlsx}")
    # print(f"  {out_runs_csv}")
    # print(f"  {out_summary_csv}")
    # print(f"  {out_total_csv}")