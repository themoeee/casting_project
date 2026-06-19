import numpy as np
import pandas as pd
from scipy.stats import qmc, norm
import itertools
from scipy.spatial.distance import cdist

def generate_optimized_htc_setups(config, initial_config, n_groups):
    """
    Selects Hard-To-Change (HTC) setups using Maximin in index-space 
    and optimizes the transition path using Nearest Neighbor.
    """
    htc_vars = {k: v for k, v in config.items() if v.get('htc')}
    if not htc_vars: 
        return [{}] * n_groups
    
    names = list(htc_vars.keys())
    physical_values, grids = [], []
    
    # 1. Create Index-Space Grid
    for name in names:
        cfg = htc_vars[name]
        lvls = np.arange(cfg['min'], cfg['max'] + cfg['step'] * 0.1, cfg['step'])
        physical_values.append(lvls)
        grids.append(np.arange(len(lvls)))
        
    univ_phys = np.array(list(itertools.product(*physical_values)))
    univ_idx = np.array(list(itertools.product(*grids)))
    
    # 2. Anchor to initial config
    init_vals = np.array([initial_config.get(n, htc_vars[n]['min']) for n in names])
    curr_idx = np.argmin(cdist([init_vals], univ_phys))
    
    # 3. Index-Space Maximin Selection
    selected_indices = [curr_idx]
    remaining = list(range(len(univ_idx)))
    remaining.remove(curr_idx)
    
    while len(selected_indices) < n_groups and remaining:
        dists = cdist(univ_idx[remaining], univ_idx[selected_indices]).min(axis=1)
        selected_indices.append(remaining.pop(np.argmax(dists)))
    
    # 4. Path Optimization (Nearest Neighbor)
    path = [selected_indices[0]]
    to_visit = selected_indices[1:]
    while to_visit:
        dists = cdist(univ_idx[[path[-1]]], univ_idx[to_visit])[0]
        path.append(to_visit.pop(np.argmin(dists)))
        
    return [dict(zip(names, univ_phys[p])) for p in path]


def sample_doe(sampling_config, initial_config, n_groups, samples_per_group, default_sigma_level=3, seed=42):
    """
    Core Split-Plot generator using direct discrete sampling for free variables.
    """
    # 1. Generate optimized HTC setups
    htc_setups = generate_optimized_htc_setups(sampling_config, initial_config, n_groups)
    
    # 2. Identify Free Variables
    free_vars = {k: v for k, v in sampling_config.items() if not v.get('htc')}
    free_names = list(free_vars.keys())
    
    # 3. Initialize Sobol Sampler
    total_runs = n_groups * samples_per_group
    sampler = qmc.Sobol(d=len(free_names), seed=seed) if free_names else None
    sobol_samples = sampler.random(n=total_runs) if sampler else np.zeros((total_runs, 0))
    
    all_data = []
    
    # 4. Discrete Sampling Logic
    for g_idx, htc_vals in enumerate(htc_setups):
        for s_idx in range(samples_per_group):
            run_idx = g_idx * samples_per_group + s_idx
            row = {'Group_ID': g_idx + 1, **htc_vals}
            
            for i, name in enumerate(free_names):
                cfg = free_vars[name]
                u = sobol_samples[run_idx, i]
                
                if cfg['type'] == 'uniform':
                    # Flat probability across all discrete levels
                    prec = cfg.get('step', 1)
                    levels = np.arange(cfg['min'], cfg['max'] + prec * 0.1, prec)
                    level_idx = int(u * len(levels))
                    row[name] = levels[level_idx]
                    
                elif cfg['type'] == 'log-uniform':
                    # Flat probability across defined logarithmic levels
                    num_levels = cfg.get('num_steps', 10)
                    levels = np.logspace(np.log10(cfg['min']), np.log10(cfg['max']), num_levels)
                    prec = cfg.get('step', 1)
                    
                    level_idx = int(u * len(levels))
                    val = levels[level_idx]
                    if prec > 0:
                        val = np.round(val / prec) * prec
                    row[name] = val
                    
                elif cfg['type'] == 'normal':
                    # Probability Binning for discrete levels
                    prec = cfg.get('step', 1)
                    levels = np.arange(cfg['min'], cfg['max'] + prec * 0.1, prec)
                    
                    # Determine Standard Deviation
                    mu = cfg['mean']
                    if 'std' in cfg:
                        sigma = cfg['std']
                    else:
                        sigma_lvl = cfg.get('sigma_level', default_sigma_level)
                        # Derive sigma so bounds touch the targeted sigma level
                        dist_to_edge = max(abs(cfg['max'] - mu), abs(mu - cfg['min']))
                        sigma = dist_to_edge / sigma_lvl
                    
                    # Calculate CDF at the edges of each discrete "bin"
                    edges = np.append(levels - prec/2, levels[-1] + prec/2)
                    probs = norm.cdf(edges, loc=mu, scale=sigma)
                    probs = np.diff(probs)
                    probs /= probs.sum() # Normalize to ensure it sums exactly to 1
                    
                    # Map Sobol [0,1) to the cumulative probability bins
                    cum_probs = np.cumsum(probs)
                    level_idx = np.searchsorted(cum_probs, u)
                    level_idx = min(level_idx, len(levels) - 1) # Safety bound
                    row[name] = levels[level_idx]
            
            all_data.append(row)
            
    return pd.DataFrame(all_data)


def extend_doe(doe: pd.DataFrame, formulae: dict, columns2drop: list = []) -> pd.DataFrame:
    """Applies dependent variable formulas to the existing DataFrame."""
    # assign() automatically passes the DataFrame to each callable and returns a new DataFrame.
    new_df = doe.assign(**formulae)
    
    # Drop the specified columns (errors='ignore' prevents crashes if a column doesn't exist)
    new_df = new_df.drop(columns=columns2drop, errors='ignore')
    
    return new_df


# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    
    # 1. Independent Variables Config (Direct mapping to discrete levels)
    sampling_params = {
        # HTC Variables (Grid generated via 'step')
        "hf1": {"min": 140, "max": 180, "step": 20, "htc": True},
        "hm2": {"min": 110, "max": 150, "step": 20, "htc": True},
        
        # Free Variables (Grid generated via 'step' or 'levels')
        "vcrit":    {"type": "uniform", "min": 0.3, "max": 0.55, "step": 0.05},
        "vI":       {"type": "normal", "mean": 2.8, "min": 2.0, "max": 3.6, "step": 0.2, "sigma_level": 3},
        "pvac":     {"type": "log-uniform", "min": 10, "max": 1000, "num_steps": 5, "step": 1},
        "sol_time": {"type": "uniform", "min": 3, "max": 6, "step": 1}
    }

    # 2. Initial Machine State
    initial_machine_state = {
        "hf1": 160,
        "hm2": 130
    }

    # 3. Dependent Variables Config (Extended later)
    dependent_formulas = {
        "hf2": lambda df: df["hf1"] + 20,
        "hm3": lambda df: df["hm2"] - 10,
        "v3":  lambda df: ((df["vI"] - df["vcrit"]) * 0.5 + df["vcrit"]).round(2)
    }

    # 4. Execution Pipeline
    print("Generating Discrete Split-Plot DOE...")
    indep_doe = sample_doe(
        sampling_config=sampling_params, 
        initial_config=initial_machine_state, 
        n_groups=5, 
        samples_per_group=20, # 100 total experiments
        seed=42
    )
    
    print("Applying Dependent Formulas...")
    final_doe = extend_doe(indep_doe, dependent_formulas)
    
    # Shift index to start at 1 for readability
    final_doe.index += 1
    
    # Show first 10 runs
    print("\nFinal DOE Preview:")
    print(final_doe.head(10))
    