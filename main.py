import json
import math
import random
import os
import sys
from datetime import datetime, timezone
from collections import defaultdict
import bisect

# Deterministic
random.seed(12345)

# Input/Output setup
INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

def load_data():
    with open(os.path.join(INPUT_DIR, "task.json"), "r") as f:
        task = json.load(f)
    
    marks = []
    with open(os.path.join(INPUT_DIR, "history.ndjson"), "r") as f:
        for line in f:
            marks.append(json.loads(line))
            
    return task, marks

def get_log_size(lamports):
    return math.log10(max(1, lamports)) - 9

def get_iso_time(iso_str):
    return datetime.fromisoformat(iso_str.replace("Z", "+00:00"))

def run_simulation():
    task, marks = load_data()
    
    history_until = get_iso_time(task["history_until"])
    global_last_mark_time = get_iso_time(marks[-1]["at"])
    
    # 1. Trajectories from ALL marks
    all_trajectories = {}
    for mark in marks:
        t = get_iso_time(mark["at"])
        for r in mark["rows"]:
            mint = r["mint"]
            log_size = get_log_size(int(r["market_cap_lamports"]))
            if mint not in all_trajectories:
                all_trajectories[mint] = {
                    "points": [], 
                    "created_at": get_iso_time(r["created_at"]), 
                    "last_venue": r["venue"],
                    "last_log_size": log_size
                }
            all_trajectories[mint]["points"].append((t, log_size))
            all_trajectories[mint]["last_venue"] = r["venue"]
            all_trajectories[mint]["last_log_size"] = log_size

    # Candidate SET: union of last 3 marks' mints (top 250 by most-recent log-size)
    last_3_mints = set()
    for mark in marks[-3:]:
        for r in mark["rows"]:
            last_3_mints.add(r["mint"])
    
    top_candidates = sorted(list(last_3_mints), key=lambda m: (-all_trajectories[m]["last_log_size"], m))[:250]
    
    # Pre-calculate slopes and drifts for candidates
    candidate_mints = {}
    for mint in top_candidates:
        data = all_trajectories[mint]
        points = data["points"] # Already sorted by time since we process marks in order
        
        # Slope calculation over full last 1.0 hour (points within 1.0h of global_last_mark_time)
        recent_points = [p for p in points if (global_last_mark_time - p[0]).total_seconds() <= 3600]
        
        if len(recent_points) >= 2:
            n = len(recent_points)
            x_vals = [(p[0] - global_last_mark_time).total_seconds() / 3600 for p in recent_points]
            y_vals = [p[1] for p in recent_points]
            
            x_sum = sum(x_vals)
            y_sum = sum(y_vals)
            xy_sum = sum(x * y for x, y in zip(x_vals, y_vals))
            x2_sum = sum(x * x for x in x_vals)
            
            denom = (n * x2_sum - x_sum**2)
            slope = (n * xy_sum - x_sum * y_sum) / denom if denom != 0 else 0.0
            slope = max(-0.5, min(0.5, slope))
        else:
            slope = 0.0
        
        age = (global_last_mark_time - data["created_at"]).total_seconds() / 3600
        
        candidate_mints[mint] = {
            "slope": slope,
            "d_inf": -0.05 if data["last_venue"] == "pool" else -0.11,
            "extra_sd": 0.25 if age < 1.0 else 0.0,
            "last_log_size": data["last_log_size"]
        }
    
    # 2. Monte Carlo
    answer_reads = [get_iso_time(r) for r in task["reads"]]
    dts = [(r - history_until).total_seconds() / 3600 for r in answer_reads]
    max_dt = max(dts)
    
    # Newcomer reserve: U(dt) = min(0.70, 0.45 + 0.35 * log10(1 + dt))
    u_vals = [min(0.70, 0.45 + 0.35 * math.log10(1 + dt)) for dt in dts]
    u_avg = sum(u_vals) / len(u_vals)
    
    # Last k caps for newcomer normal
    last_mark_rows = marks[-1]["rows"]
    def get_last_k_cap(k):
        # k is 1-indexed
        if len(last_mark_rows) >= k:
            return get_log_size(int(last_mark_rows[k-1]["market_cap_lamports"]))
        return 3.0

    step_h = 1.0 if max_dt > 2 else max(0.05, max_dt / 6.0)
    n_steps = math.ceil(max_dt / step_h)
    
    recorded_steps = []
    for dt in dts:
        step = min(n_steps - 1, int(dt // step_h))
        recorded_steps.append(step)
    
    place_counts = [defaultdict(int) for _ in range(5)]
    size_samples = [[] for _ in range(5)]
    
    rng = random.Random(12345)
    
    for _ in range(6000):
        candidate_x = {m: candidate_mints[m]["last_log_size"] for m in top_candidates}
        
        read_idx = 0
        for step in range(n_steps):
            t_mid = (step + 0.5) * step_h
            
            for m in top_candidates:
                c = candidate_mints[m]
                sd_i = math.sqrt((0.70**2 + c["extra_sd"]**2) * step_h)
                change = (c["slope"] * math.exp(-t_mid / 1.5) + c["d_inf"]) * step_h + rng.gauss(0, sd_i)
                candidate_x[m] += change
            
            while read_idx < len(recorded_steps) and step == recorded_steps[read_idx]:
                ranked = sorted(candidate_x.items(), key=lambda x: (-x[1], x[0]))
                for k in range(5):
                    mint, x = ranked[k]
                    place_counts[k][mint] += 1
                    size_samples[k].append(max(0.0, min(5.5, x)))
                read_idx += 1
    # 3. Probabilities, Top5, and Sizes
    total_draws = 6000 * len(recorded_steps)
    
    # top5: score(mint) = sum over k of (place_counts[k][mint] / total_draws) * (6 - k)
    mint_scores = defaultdict(float)
    for k in range(5):
        for mint, count in place_counts[k].items():
            mint_scores[mint] += (count / total_draws) * (6 - (k + 1))
    
    top5 = sorted(mint_scores.keys(), key=lambda m: (-mint_scores[m], m))[:5]
    
    # Probabilities
    prob_rows = []
    for k in range(5):
        raw_probs = {m: count / total_draws for m, count in place_counts[k].items()}
        sorted_mints = sorted(raw_probs.keys(), key=lambda m: (-raw_probs[m], m))
        
        place_rows = []
        named_mints = set()
        for mint in sorted_mints:
            p = round(raw_probs[mint] * (1 - u_avg), 6)
            if p >= 0.001 and len(place_rows) < 100:
                place_rows.append({"place": k + 1, "mint": mint, "p": p})
                named_mints.add(mint)
        
        # Zero-fill: top 100 of last mark
        if len(place_rows) < 100:
            for r in last_mark_rows[:100]:
                mint = r["mint"]
                if mint not in named_mints and len(place_rows) < 100:
                    place_rows.append({"place": k + 1, "mint": mint, "p": 0.0})
                    named_mints.add(mint)
        
        prob_rows.extend(place_rows)
        
    # Sizes
    mc_w = (1 - u_avg) * math.exp(-max(max_dt - 2.0, 0.0) / 3.0)
    newcomer_w = 1 - mc_w
    
    def phi(z):
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))
    
    size_rows = []
    edges = [2.00 + 0.05 * i for i in range(81)] # 2.00 to 6.00
    
    for k in range(5):
        counts = [0] * 82
        for s in size_samples[k]:
            if s < 2.00:
                counts[0] += 1
            elif s >= 6.00:
                counts[81] += 1
            else:
                b = int((s - 2.00) / 0.05) + 1
                counts[min(81, b)] += 1
        
        n_samples = len(size_samples[k])
        shares = [ (c / n_samples) * mc_w for c in counts ]
        
        # Newcomer normal
        mu = get_last_k_cap(k + 1) + 0.10
        sd = 0.40 + 0.035 * max_dt
        
        # Bin 0: (-inf, 2.00)
        shares[0] += (phi((2.00 - mu) / sd)) * newcomer_w
        # Bins 1..80: [lo, hi)
        for b in range(1, 81):
            lo = edges[b-1]
            hi = edges[b]
            shares[b] += (phi((hi - mu) / sd) - phi((lo - mu) / sd)) * newcomer_w
        # Bin 81: [6.00, +inf)
        shares[81] += (1 - phi((6.00 - mu) / sd)) * newcomer_w
        
        # Normalize
        total_share = sum(shares)
        if total_share > 0:
            shares = [s / total_share for s in shares]
            
        for b, s in enumerate(shares):
            if s >= 0.001:
                size_rows.append({"place": k + 1, "bin": b, "p": round(s, 6)})
                
    output = {
        "probabilities": prob_rows,
        "top5": top5,
        "sizes": size_rows
    }
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "forecast.json"), "w") as f:
        json.dump(output, f)
        f.write("\n")
    print("Forecast generated.")

if __name__ == "__main__":
    run_simulation()
