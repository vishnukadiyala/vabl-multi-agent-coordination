"""Comprehensive cross-data scan to find what actually sticks out."""
import json, numpy as np, glob, os

def rewards(path): return json.load(open(path))['rewards']
def f50(path):
    r = rewards(path); return sum(r[-50:]) / min(50, len(r))
def best(path): return max(rewards(path))
def drop(path): return best(path) - f50(path)

def agg(paths):
    if not paths: return None
    vals = [f50(p) for p in paths]
    b = [best(p) for p in paths]
    d = [bi-fi for bi, fi in zip(b, vals)]
    n = len(vals)
    return dict(n=n,
                f50_mean=np.mean(vals), f50_std=np.std(vals, ddof=1) if n>1 else 0,
                best_mean=np.mean(b),   best_std=np.std(b, ddof=1) if n>1 else 0,
                drop_mean=np.mean(d),   drop_std=np.std(d, ddof=1) if n>1 else 0,
                f50=vals, best=b, drop=d)

def pp(name, a):
    if a is None:
        print(f"{name:38s}  (no files)")
        return
    print(f"{name:38s} n={a['n']:>2d}  F50={a['f50_mean']:7.2f}+/-{a['f50_std']:5.2f}  Best={a['best_mean']:7.2f}+/-{a['best_std']:5.2f}  drop={a['drop_mean']:6.2f}+/-{a['drop_std']:5.2f}")


# ===== AA =====
print("="*110)
print("OVERCOOKED ASYMMETRIC ADVANTAGES (10M)")
print("="*110)
ph = json.load(open('results/canonical_phase2.json'))['configs']
# Build aggregate from canonical (no best_per_seed) + merge extra seeds
for cfg in ['A_full','A_neither','A_no_attn','A_no_aux','B_anneal','B_stopgrad','B_anneal_stopgrad']:
    vals = ph[cfg]['final50_per_seed']
    best_mean = ph[cfg].get('best_mean', 0)
    best_std = ph[cfg].get('best_std', 0)
    print(f"{'VABL_'+cfg:38s} n={len(vals):>2d}  F50={np.mean(vals):7.2f}+/-{np.std(vals,ddof=1):5.2f}  Best={best_mean:7.2f}+/-{best_std:5.2f}  (drop n/a without per-seed best)")

# n=10 merges
extra_full = sorted(glob.glob('results/extra_seeds_aa/extra_A_full_seed*.json'))
if extra_full:
    f_extra = agg(extra_full)
    full_combined = ph['A_full']['final50_per_seed'] + f_extra['f50']
    print(f"\n>>> A_full combined n=10 (phase2 0-4 + extra 5-9)")
    print(f"     Final50 mean={np.mean(full_combined):7.2f}  std={np.std(full_combined,ddof=1):5.2f}")
    print(f"     per-seed: {[round(x,2) for x in full_combined]}")

extra_neither = sorted(glob.glob('results/extra_seeds_aa/extra_A_neither_seed*.json'))
if extra_neither:
    f_extra = agg(extra_neither)
    neither_combined = ph['A_neither']['final50_per_seed'] + f_extra['f50']
    print(f"\n>>> A_neither combined n=10")
    print(f"     Final50 mean={np.mean(neither_combined):7.2f}  std={np.std(neither_combined,ddof=1):5.2f}")
    print(f"     per-seed: {[round(x,2) for x in neither_combined]}")

# Separate encoder
sep_paths = sorted(glob.glob('results/separate_encoder/sep_encoder_full_seed*.json'))
print()
pp("separate_encoder (all)", agg(sep_paths))

# Baselines
print()
print("--- Baselines on AA ---")
for m in ['aerial','mappo','tarmac','commnet']:
    files = sorted(glob.glob(f'results/remote_pull/phase2_baselines/baseline_{m}_asymmetric_advantages_seed*.json'))
    pp(m, agg(files))

# AERIAL fix-path
print()
print("--- AERIAL fix-path (NEW) ---")
pp("AERIAL+aux (Full)", agg(sorted(glob.glob('results/aerial_fix_path/aerial_aux_full_seed*.json'))))
pp("AERIAL+aux+stopgrad", agg(sorted(glob.glob('results/aerial_fix_path/aerial_aux_stopgrad_seed*.json'))))

# ===== CRAMPED =====
print()
print("="*110)
print("OVERCOOKED CRAMPED ROOM (10M)")
print("="*110)
cr = json.load(open('results/canonical_phase2_cramped.json'))['configs']
for cfg in ['full','neither','no_attn','no_aux']:
    if cfg in cr:
        vals = cr[cfg]['final50_per_seed']
        best_mean = cr[cfg].get('best_mean', 0)
        print(f"{'cramped_'+cfg:38s} n={len(vals):>2d}  F50={np.mean(vals):7.2f}+/-{np.std(vals,ddof=1):5.2f}  Best={best_mean:7.2f}")

print()
extra_cr_full = sorted(glob.glob('results/extra_seeds_cramped/cramped_full_seed*.json'))
extra_cr_noattn = sorted(glob.glob('results/extra_seeds_cramped/cramped_no_attn_seed*.json'))
pp("cramped_full (NEW seeds 5-9)", agg(extra_cr_full))
pp("cramped_no_attn (NEW seeds 5-9)", agg(extra_cr_noattn))

# Sample efficiency on Cramped (new)
print()
print("--- Cramped sample-efficiency (NEW) ---")
for ep in [1250, 6250, 12500, 25000]:
    for cfg in ['full','no_attn','no_aux','neither']:
        files = sorted(glob.glob(f'results/sample_efficiency_cramped/se_cr_{cfg}_{ep}ep_seed*.json'))
        pp(f"cramped_{cfg}_{ep}ep", agg(files))
    print()

# ===== SMAX =====
print("="*110)
print("SMAX 3v3 (50K eps)")
print("="*110)
smax = json.load(open('results/canonical_smax.json'))
for cfg in ['full','no_attn','no_aux','neither']:
    if cfg in smax:
        vals = smax[cfg]['final_per_seed']
        print(f"{'smax_'+cfg:38s} n={len(vals):>2d}  F50={np.mean(vals):7.2f}+/-{np.std(vals,ddof=1):5.2f}  Best={smax[cfg].get('best_mean',0):.2f}")
try:
    sfx = json.load(open('results/canonical_smax_fixes.json'))['configs']
    for cfg in sfx:
        vals = sfx[cfg].get('final_per_seed') or sfx[cfg].get('final50_per_seed') or []
        if vals:
            print(f"{'smax_'+cfg:38s} n={len(vals):>2d}  F50={np.mean(vals):7.2f}+/-{np.std(vals,ddof=1):5.2f}")
except Exception as e:
    print(f"(smax fixes: {e})")

# ===== MPE =====
print()
print("="*110)
print("MPE simple_spread (100K eps)")
print("="*110)
try:
    mpe = json.load(open('results/canonical_mpe.json'))
    for cfg in ['full','no_attn','no_aux','neither']:
        if cfg in mpe:
            vals = mpe[cfg].get('final_per_seed') or mpe[cfg].get('final50_per_seed') or []
            if vals:
                print(f"{'mpe_'+cfg:38s} n={len(vals):>2d}  F50={np.mean(vals):7.2f}+/-{np.std(vals,ddof=1):5.2f}")
except Exception as e:
    print(f"(mpe: {e})")

# ===== LAMBDA =====
print()
print("="*110)
print("LAMBDA SENSITIVITY")
print("="*110)
for name in ['lambda0001_10M','lambda001_10M','lambda001_50M','lambda005_50M']:
    files = sorted(glob.glob(f'results/remote_pull/lambda_sensitivity/{name}_seed*.json'))
    pp(name, agg(files))
