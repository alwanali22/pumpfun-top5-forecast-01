import json
import math
import random
import os
from datetime import datetime

def parse_iso(iso_str):
    return datetime.fromisoformat(iso_str.replace('Z', '+00:00'))

def log_size(mcap_str):
    return math.log10(int(mcap_str)) - 9

def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(input_dir, "task.json")) as f:
        task = json.load(f)
    
    with open(os.path.join(input_dir, "history.ndjson")) as f:
        marks = [json.loads(line) for line in f if json.loads(line).get("ok") and json.loads(line).get("rows")]
    
    if not marks:
        raise Exception("No usable marks")

    cut_time = parse_iso(task["history_until"])
    t0 = parse_iso(marks[0]["at"])
    
    # Trajectories
    trajectories = {}
    for mark in marks:
        t = (parse_iso(mark["at"]) - t0).total_seconds() / 3600
        for row in mark["rows"]:
            mint = row["mint"]
            if mint not in trajectories:
                trajectories[mint] = []
            trajectories[mint].append((t, log_size(row["market_cap_lamports"]), parse_iso(row["created_at"]), row.get("venue", "pool")))

    last_mark = marks[-1]
    last_time = (parse_iso(last_mark["at"]) - t0).total_seconds() / 3600
    
    # Candidates
    candidate_mints = set()
    for mark in marks[-3:]:
        for row in mark["rows"]:
            candidate_mints.add(row["mint"])
    
    candidates = []
    for mint in candidate_mints:
        traj = trajectories[mint]
        # Most recent
        x0 = traj[-1][1]
        
        # Slope
        recent_points = [p for p in traj if last_time - p[0] <= 1.0]
        if len(recent_points) >= 2:
            # OLS
            n = len(recent_points)
            sum_t = sum(p[0] for p in recent_points)
            sum_x = sum(p[1] for p in recent_points)
            sum_tx = sum(p[0] * p[1] for p in recent_points)
            sum_t2 = sum(p[0]**2 for p in recent_points)
            slope = (n * sum_tx - sum_t * sum_x) / (n * sum_t2 - sum_t**2)
        elif len(traj) >= 2:
            slope = (traj[-1][1] - traj[-2][1]) / (traj[-1][0] - traj[-2][0])
        else:
            slope = 0
        slope = max(-0.5, min(0.5, slope))
        
        # d_inf
        last_venue = traj[-1][3]
        d_inf = -0.05 if last_venue == "pool" else -0.11
        
        # extra_sd
        created_at = traj[0][2]
        extra_sd = 0.15 if (parse_iso(last_mark["at"]) - created_at).total_seconds() / 3600 < 1.0 else 0.0
        
        candidates.append({"mint": mint, "x0": x0, "slope": slope, "d_inf": d_inf, "extra_sd": extra_sd})
    
    # Sort candidates by x0
    candidates.sort(key=lambda c: c["x0"], reverse=True)
    candidates = candidates[:250]
    
    # Monte Carlo
    rng = random.Random(12345)
    dts = sorted([(parse_iso(r) - cut_time).total_seconds() / 3600 for r in task["reads"]])
    max_dt = dts[-1]
    step_h = 1.0 if max_dt > 2 else max(0.05, max_dt / 6.0)
    n_steps = math.ceil(max_dt / step_h)
    read_steps = [min(n_steps - 1, math.floor(dt / step_h)) for dt in dts]
    
    n_paths = 6000
    place_counts = {k: {} for k in range(1, 6)}
    size_samples = {k: [] for k in range(1, 6)}
    
    for _ in range(n_paths):
        # Current state of candidates
        current = {c["mint"]: c["x0"] for c in candidates}
        for s in range(n_steps):
            t_mid = (s + 0.5) * step_h
            for c in candidates:
                mint = c["mint"]
                # Update: new_x = x + (slope * exp(-t_mid / 1.5) + d_inf) * step_h + gauss(0, sd)
                sd = math.sqrt(0.70**2 * step_h + c["extra_sd"]**2 * step_h)
                drift = (c["slope"] * math.exp(-t_mid / 1.5) + c["d_inf"]) * step_h
                noise = rng.gauss(0, sd)
                current[mint] += drift + noise
            
            if s in read_steps:
                # Rank
                sorted_mints = sorted(current.keys(), key=lambda m: current[m], reverse=True)
                for k in range(1, 6):
                    mint = sorted_mints[k-1]
                    place_counts[k][mint] = place_counts[k].get(mint, 0) + 1
                    size_samples[k].append(current[mint])
    
    sorted_last_rows = sorted(last_mark["rows"], key=lambda r: r.get("rank", 0))
    
    # Probabilities
    total_draws = n_paths * len(set(read_steps))
    probabilities = []
    
    def U(dt): return min(0.70, 0.45 + 0.35 * math.log10(1 + dt))
    u_avg = sum(U(dt) for dt in dts) / len(dts)
    
    for k in range(1, 6):
        raw = {m: place_counts[k].get(m, 0) / total_draws for m in place_counts[k]}
        sorted_mints = sorted(raw.keys(), key=lambda m: raw[m], reverse=True)
        
        # Named rows
        named = []
        for m in sorted_mints:
            p = raw[m] * (1 - u_avg)
            if p >= 0.001:
                named.append({"mint": m, "place": k, "p": round(p, 6)})
        
        # Cap at 100
        named = sorted(named, key=lambda x: x["p"], reverse=True)[:100]
        probabilities.extend(named)
        
        # Zero-fill
        chart_likely = [r["mint"] for r in sorted_last_rows][:80]
        count = len([r for r in probabilities if r["place"] == k])
        for m in chart_likely:
            if count >= 100: break
            if m not in [r["mint"] for r in probabilities if r["place"] == k] and raw.get(m, 0) < 0.001:
                probabilities.append({"mint": m, "place": k, "p": 0.0})
                count += 1
                
    # Top 5
    scores = {}
    for k in range(1, 6):
        for m, count in place_counts[k].items():
            scores[m] = scores.get(m, 0) + (count / total_draws) * (6 - k)
    
    top5_mints = sorted(scores.keys(), key=lambda m: scores[m], reverse=True)[:5]
    
    # Sizes
    sizes = []
    for k in range(1, 6):
        # Histogram
        hist = [0] * 82
        for val in size_samples[k]:
            if val < 2.00: bin = 0
            elif val >= 6.00: bin = 81
            else: bin = int((val - 2.00) / 0.05) + 1
            hist[bin] += 1
        
        # Weights
        bin_shares = [(h / len(size_samples[k])) * (1 - u_avg) for h in hist]
        
        # Newcomer component
        # mu_n = (log-size of the coin at rank k in last_rows) + 0.10, sd_n = 0.70
        last_row = sorted_last_rows[k-1] if len(sorted_last_rows) >= k else None
        mu_n = (log_size(last_row["market_cap_lamports"]) + 0.10) if last_row else 2.0
        sd_n = 0.70
        
        def Phi(x, mu, sd): return 0.5 * (1 + math.erf((x - mu) / (sd * math.sqrt(2))))
        
        for b in range(82):
            if b == 0:
                prob = Phi(2.00, mu_n, sd_n)
            elif b == 81:
                prob = 1 - Phi(6.00, mu_n, sd_n)
            else:
                prob = Phi(2.00 + 0.05 * b, mu_n, sd_n) - Phi(2.00 + 0.05 * (b - 1), mu_n, sd_n)
            
            bin_shares[b] += prob * u_avg
            
        # Normalize
        total_share = sum(bin_shares)
        for b in range(82):
            share = bin_shares[b] / total_share
            if share >= 0.001:
                sizes.append({"place": k, "bin": b, "p": round(share, 6)})
                
    # Final write
    output = {"probabilities": probabilities, "top5": top5_mints, "sizes": sizes}
    with open(os.path.join(output_dir, task["output"]), "w") as f:
        f.write(json.dumps(output) + "\n")

if __name__ == "__main__":
    main()
