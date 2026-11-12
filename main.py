import json
import math
import os
import random
import sys
import bisect
from datetime import datetime

def parse_iso(s):
    return datetime.fromisoformat(s.replace('Z', '+00:00'))

def load_data(input_dir):
    with open(os.path.join(input_dir, 'task.json'), 'r') as f:
        task = json.load(f)
    
    history = []
    with open(os.path.join(input_dir, 'history.ndjson'), 'r') as f:
        for line in f:
            mark = json.loads(line)
            if not mark.get('ok') or 'rows' not in mark:
                continue
            history.append(mark)
    
    history.sort(key=lambda m: parse_iso(m['at']))
    return task, history

def run_simulation():
    input_dir = os.environ.get('MAKEFUN_INPUT_DIR', '/input')
    output_dir = os.environ.get('MAKEFUN_OUTPUT_DIR', '/output')
    
    task, history = load_data(input_dir)
    history_until = parse_iso(task['history_until'])
    
    # Trajectories
    mints = {} # mint -> list of (t_hours, logsize, venue), created_at
    first_at = parse_iso(history[0]['at'])
    for mark in history:
        at = parse_iso(mark['at'])
        t_hours = (at - first_at).total_seconds() / 3600
        for row in mark['rows']:
            mint = row['mint']
            if mint not in mints:
                mints[mint] = {'points': [], 'created_at': parse_iso(row['created_at'])}
            logsize = math.log10(int(row['market_cap_lamports'])) - 9
            mints[mint]['points'].append((t_hours, logsize, row['venue']))
            
    # Candidates
    last_marks = history[-3:]
    candidates = {}
    for mark in last_marks:
        for row in mark['rows']:
            if row['mint'] not in candidates:
                candidates[row['mint']] = row
    
    sorted_candidates = []
    for mint, row in candidates.items():
        sorted_candidates.append((mint, math.log10(int(row['market_cap_lamports'])) - 9))
    sorted_candidates.sort(key=lambda x: x[1], reverse=True)
    top_candidates = sorted_candidates[:250]
    
    # Per candidate: x0, slope, d_inf, extra_sd
    candidate_params = {}
    last_mark_time = parse_iso(history[-1]['at'])
    for mint, _ in top_candidates:
        points = mints[mint]['points']
        x0 = points[-1][1]
        
        # Robust slope
        recent_points = [p for p in points if (last_mark_time - parse_iso(next(m['at'] for m in history if (parse_iso(m['at']) - parse_iso(history[0]['at'])).total_seconds()/3600 >= p[0] - 0.0001 ))).total_seconds() <= 3600]
        if len(recent_points) >= 2:
            n = len(recent_points)
            sum_x = sum(p[0] for p in recent_points)
            sum_y = sum(p[1] for p in recent_points)
            sum_xy = sum(p[0]*p[1] for p in recent_points)
            sum_xx = sum(p[0]*p[0] for p in recent_points)
            slope = (n*sum_xy - sum_x*sum_y) / (n*sum_xx - sum_x*sum_x)
        else:
            slope = 0
        slope = max(-0.5, min(0.5, slope))
        
        d_inf = -0.05 if points[-1][2] == "pool" else -0.11
        age = (last_mark_time - mints[mint]['created_at']).total_seconds() / 3600
        extra_sd = 0.25 if age < 1.0 else 0.0
        
        candidate_params[mint] = {'x0': x0, 'slope': slope, 'd_inf': d_inf, 'extra_sd': extra_sd}
    
    # Monte Carlo
    random.seed(12345)
    paths = 6000
    dts = []
    for read_iso in task['reads']:
        dt = (parse_iso(read_iso) - history_until).total_seconds() / 3600
        if dt > 0:
            dts.append(dt)
    dts.sort()
    
    max_dt = dts[-1] if dts else 0
    step_h = 1.0 if max_dt > 2 else max(0.05, max_dt/6)
    n_steps = math.ceil(max_dt / step_h)
    record_steps = set()
    for dt in dts:
        record_steps.add(min(n_steps-1, int(dt // step_h)))
    record_steps = sorted(list(record_steps))
    
    place_counts = {k: {mint: 0 for mint in candidate_params} for k in range(1, 6)}
    size_samples = {k: [] for k in range(1, 6)}
    
    for _ in range(paths):
        xs = {mint: candidate_params[mint]['x0'] for mint in candidate_params}
        for step in range(n_steps):
            t_mid = (step + 0.5) * step_h
            for mint, params in candidate_params.items():
                sd_i = math.sqrt((1.0**2 + params['extra_sd']**2) * step_h)
                xs[mint] += (params['slope'] * math.exp(-t_mid / 1.5) + params['d_inf']) * step_h + random.gauss(0, sd_i)
            
            if step in record_steps:
                sorted_mints = sorted(xs.items(), key=lambda x: x[1], reverse=True)
                for k in range(1, 6):
                    if k-1 < len(sorted_mints):
                        mint, x = sorted_mints[k-1]
                        place_counts[k][mint] += 1
                        size_samples[k].append(min(5.0, max(0.0, x)))

    # Post-simulation
    total_draws = paths * len(record_steps)
    U = []
    for dt in dts:
        U.append(min(0.70, 0.45 + 0.35 * math.log10(1 + dt)))
    u_avg = sum(U) / len(U) if U else 0.70
    
    # Probabilities
    output_probs = []
    for k in range(1, 6):
        raw = {mint: place_counts[k][mint] / total_draws for mint in candidate_params}
        sorted_raw = sorted(raw.items(), key=lambda x: x[1], reverse=True)
        rows_k = 0
        for mint, r in sorted_raw:
            if rows_k >= 100: break
            p = r * (1 - u_avg)
            if p < 0.001: break
            output_probs.append({"mint": mint, "place": k, "p": round(p, 6)})
            rows_k += 1
        last_rows = sorted(last_marks[-1]['rows'], key=lambda x: int(x['rank']))
        for i in range(min(100, len(last_rows))):
            if rows_k >= 100: break  # contract: at most 100 rows per place (named + zero-fill)
            mint = last_rows[i]['mint']
            if mint not in raw or raw[mint] < 0.001:
                # Add check to see if this mint is already named
                if not any(item['mint'] == mint and item['place'] == k for item in output_probs):
                    output_probs.append({"mint": mint, "place": k, "p": 0.0})
                    rows_k += 1
                    
    # Top 5
    score = {}
    for mint in candidate_params:
        score[mint] = sum((place_counts[k][mint] / total_draws) * (6 - k) for k in range(1, 6))
    top5_mints = sorted(score.keys(), key=lambda m: score[m], reverse=True)[:5]
    
    # Sizes
    edges = [2.00 + i * 0.05 for i in range(81)]
    # I need to implement a standard normal CDF (Phi).
    def phi(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
        
    output_sizes = []
    for k in range(1, 6):
        # Histogram
        bins = [0] * 82
        for s in size_samples[k]:
            b = bisect.bisect_right(edges, s)
            bins[b] += 1
        
        # Share from MC
        shares = [b / len(size_samples[k]) * (1 - u_avg) for b in bins]
        
        # Newcomer
        # Wait, the spec says "last place-k cap" which is logsize of k-th row of last mark.
        last_mark_rows = sorted(last_marks[-1]['rows'], key=lambda x: int(x['rank']))
        if len(last_mark_rows) >= k:
            cap_k = math.log10(int(last_mark_rows[k-1]['market_cap_lamports'])) - 9
        else:
            cap_k = 3.0
        
        # mu = median of size_samples[k] + 0.30, clamped to [cap_k - 0.5, cap_k + 0.75]
        sorted_samples = sorted(size_samples[k])
        median_sample = sorted_samples[len(sorted_samples)//2]
        mu = max(cap_k - 0.5, min(cap_k + 0.75, median_sample + 0.30))
        sd = 0.35
        
        # Add newcomer shares
        # bin 0: < 2.00. bin 81: >= 6.00.
        # edges = [2.00, 2.05, ..., 6.00]
        # Phi((hi - mu)/sd) - Phi((lo - mu)/sd)
        
        newcomer_shares = [0.0] * 82
        for b in range(82):
            if b == 0:
                lo, hi = -float('inf'), edges[0]
            elif b == 81:
                lo, hi = edges[-1], float('inf')
            else:
                lo, hi = edges[b-1], edges[b]
            
            # Phi is standard normal CDF
            prob = phi((hi - mu)/sd) - phi((lo - mu)/sd)
            newcomer_shares[b] = prob * u_avg
            
        # Add and normalize
        total_shares = [shares[b] + newcomer_shares[b] for b in range(82)]
        norm_sum = sum(total_shares)
        
        for b, share in enumerate(total_shares):
            s = share / norm_sum
            if s >= 0.001:
                output_sizes.append({"place": k, "bin": b, "p": round(s, 6)})

    # Output
    result = {
        "probabilities": output_probs,
        "top5": top5_mints,
        "sizes": output_sizes
    }
    
    with open(os.path.join(output_dir, 'forecast.json'), 'w') as f:
        json.dump(result, f)
        f.write('\n')

if __name__ == "__main__":
    run_simulation()
