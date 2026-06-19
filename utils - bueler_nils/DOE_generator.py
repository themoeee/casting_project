import numpy as np
import pandas as pd
from itertools import product
from scipy.spatial.distance import cdist
from scipy.stats import qmc
from shot_curve_util import fillingCurvePoints
import matplotlib.pyplot as plt
import seaborn as sns

def generate_split_plot_doe(n_groups, samples_per_group, config, initial_config, seed=None):
    """
    Generates a Split-Plot Design of Experiments using Maximin for 
    Hard-To-Change (HTC) variables and Sobol sequences for free variables.
    """
    if seed is not None:
        np.random.seed(seed)
        
    total_samples = n_groups * samples_per_group
    
    # --- Step 1: Parse Configuration ---
    htc_vars, free_vars, const_vars, dep_vars = {}, {}, {}, {}
    
    for name, params in config.items():
        if params.get('type') == 'dependent':
            dep_vars[name] = params
        elif params.get('type') == 'constant':
            const_vars[name] = params
        # Note the default to False if 'htc' is not specified
        elif params.get('htc', False) == True: 
            htc_vars[name] = params
        else:
            free_vars[name] = params

    # Helper: Generate discrete vector for a parameter
    def get_discrete_vector(params):
        if params['type'] == 'discrete_uniform':
            # Add a small epsilon to ensure the max value is included if it falls exactly on a step
            return np.arange(params['min'], params['max'] + params['step'] * 0.1, params['step'])
        elif params['type'] == 'discrete_log':
            return np.logspace(np.log10(params['min']), np.log10(params['max']), params['num_steps']).round()
        raise ValueError(f"Unknown type {params['type']}")

    # --- Step 2: HTC Generation & Max-Min Sampling ---
    htc_names = list(htc_vars.keys())
    
    if htc_names:
        htc_vectors = [get_discrete_vector(htc_vars[name]) for name in htc_names]
        all_htc_combos = np.array(list(product(*htc_vectors)))
        
        if n_groups > len(all_htc_combos):
            raise ValueError(f"Requested {n_groups} groups, but only {len(all_htc_combos)} discrete HTC combinations exist.")
        
        init_vec = np.array([initial_config[name] for name in htc_names])
        distances_to_init = cdist([init_vec], all_htc_combos)[0]
        init_idx = np.argmin(distances_to_init)
        
        if distances_to_init[init_idx] > 1e-5:
            print(f"Warning: initial_config not exactly in discrete grid. Snapping to nearest: {all_htc_combos[init_idx]}")
        
        # Max-Min Selection
        selected_htc = [all_htc_combos[init_idx]]
        remaining_pool = np.delete(all_htc_combos, init_idx, axis=0)
        
        for _ in range(1, n_groups):
            dist_matrix = cdist(remaining_pool, selected_htc)
            min_dists = dist_matrix.min(axis=1)
            best_candidate_idx = np.argmax(min_dists)
            selected_htc.append(remaining_pool[best_candidate_idx])
            remaining_pool = np.delete(remaining_pool, best_candidate_idx, axis=0)
            
        selected_htc = np.array(selected_htc)
        
        # Step 3: Nearest Neighbor Sorting (The "Path")
        path_indices = [0]
        unvisited = list(range(1, n_groups))
        
        while unvisited:
            current_point = selected_htc[path_indices[-1]].reshape(1, -1)
            candidates = selected_htc[unvisited]
            dists = cdist(current_point, candidates)[0]
            nearest_idx = np.argmin(dists)
            path_indices.append(unvisited[nearest_idx])
            unvisited.pop(nearest_idx)
            
        sorted_htc = selected_htc[path_indices]
    else:
        # Fallback if no HTC variables are provided
        sorted_htc = np.empty((n_groups, 0))

    # --- Step 4: Free Variables Sobol Mapping ---
    free_names = list(free_vars.keys())
    
    # Using Sobol instead of LHS for superior space-filling on discrete grids
    # Note: Sobol is optimal when total_samples is a power of 2, but works well regardless.
    if not (total_samples & (total_samples - 1) == 0):
        print(f"Note: Sobol sequences are most perfectly balanced when total samples ({total_samples}) is a power of 2.")
        
    sobol_sampler = qmc.Sobol(d=len(free_names), seed=seed)
    sobol_matrix = sobol_sampler.random(n=total_samples)
    
    df = pd.DataFrame()
    
    # Tile the sorted HTC parameters
    if htc_names:
        tiled_htc = np.repeat(sorted_htc, samples_per_group, axis=0)
        for i, name in enumerate(htc_names):
            df[name] = tiled_htc[:, i]
        
    # Map Sobol [0, 1) matrix to discrete Free Variables
    for i, name in enumerate(free_names):
        vec = get_discrete_vector(free_vars[name])
        idx = np.floor(sobol_matrix[:, i] * len(vec)).astype(int)
        idx = np.clip(idx, 0, len(vec) - 1) # Safety clip
        df[name] = vec[idx]
        
    # Add Constants
    for name, params in const_vars.items():
        df[name] = params['value']
        
    # --- Step 5: Compute Dependent Variables ---
    for name, params in dep_vars.items():
        df[name] = params['formula'](df)
        
    df.insert(0, 'Group_ID', np.repeat(np.arange(1, n_groups + 1), samples_per_group))
        
    return df

# ==========================================
# Example Usage:
# ==========================================
if __name__ == "__main__":

    filling_constants = fillingCurvePoints(f_liq_sol=0.98, rho_solid=2.7e3)

    def t_kr(vc: float, delay: float = -0.65, first_phase_type: str = "Buhler", filling_obj: fillingCurvePoints = filling_constants) -> float:
        if "buhler" in first_phase_type.lower() or "buehler" in first_phase_type.lower():
            # (v_crit, s2, s3, v3, s4, v4, sbrake, vbrake, s_ffin)
            _, _, _, _, _, _, t_kr = filling_obj._buhler(filling_obj.s, filling_obj.d_M, filling_obj.fr, filling_obj.s_m_100, vc, v_initial=filling_obj.v_init)
            return t_kr - delay
        else:
            raise NotImplementedError()

    
    # 1. Define the configuration dictionary
    parameter_config = {
        # Slow Variables (Temperatures)
        "hf1":  {"type": "discrete_uniform", "min": 140, "max": 180, "step": 20, "htc": True},
        "hm2": {"type": "discrete_uniform", "min": 110, "max": 150, "step": 20, "htc": True},
        "metal": {"type": "discrete_uniform", "min": 675, "max": 715, "step": 20, "htc": True},
        
        # Free Variables
        "vcrit":    {"type": "discrete_uniform", "min": 0.3, "max": 0.55, "step": 0.05},
        "vI":    {"type": "discrete_uniform", "min": 2, "max": 3.6, "step": 0.1},
        # ---------------------------------------------------------------------------------------
        # to be discussed
        "sm100_s4":    {"type": "discrete_uniform", "min": 10, "max": 110, "step": 10},
        "v3_vI_vcrit":    {"type": "discrete_uniform", "min": 0.1, "max": 0.9, "step": 0.05},
        "s34_ratio":    {"type": "discrete_uniform", "min": 0.1, "max": 0.9, "step": 0.05},
        "sbreak_shift":    {"type": "discrete_uniform", "min": -5, "max": 25, "step": 10},
        "vbrake":    {"type": "discrete_uniform", "min": 0.2, "max": 1.5, "step": 0.1},
        # ---------------------------------------------------------------------------------------
        "p0":     {"type": "discrete_uniform", "min": 250, "max": 500, "step": 50},
        "pvac":   {"type": "discrete_log", "min": 10, "max": 1000, "num_steps": 7},
        "sol_time": {"type": "discrete_uniform", "min": 3, "max": 6, "step": 1},
        
        # Constants
        "piston": {"type": "constant", "value": 60},
        
        # Dependent Variables (using lambda functions applied to the dataframe)
        "hf2": {"type": "dependent", "formula": lambda df: df["hf1"] + 20},
        "hf3": {"type": "dependent", "formula": lambda df: df["hf1"]},
        "f3": {"type": "dependent", "formula": lambda df: df["hf1"] - 40},
        "f4": {"type": "dependent", "formula": lambda df: df["hf1"] - 40},
        "hm3": {"type": "dependent", "formula": lambda df: df["hm2"] - 10},
        "hm4": {"type": "dependent", "formula": lambda df: df["hm2"] + 20},
        "m4": {"type": "dependent", "formula": lambda df: df["hm2"] - 50},
        "s4": {"type": "dependent", "formula": lambda df: (df["sm100_s4"] + filling_constants.s_m_100*1e3).round()},
        "s3": {"type": "dependent", "formula": lambda df: (df["sm100_s4"]*df["s34_ratio"] + filling_constants.s_m_100*1e3).round()},
        "v3": {"type": "dependent", "formula": lambda df: ((df["vI"] - df["vcrit"])*df["v3_vI_vcrit"] + df["vcrit"]).round(2)},
        "s_brake": {"type": "dependent", "formula": lambda df: (df["sbreak_shift"] + filling_constants.s_ffin*1e3).round()},
        # "s5": {"type": "dependent", "formula": lambda df: (df["s_brake"] - 50).round()},
        # "s6": {"type": "dependent", "formula": lambda df: (df["s5"] + 13).round()},
        # "v6": {"type": "dependent", "formula": lambda df: (7/8*df["vI"] + df["vbrake"]/8).round(2)},
        # "s7": {"type": "dependent", "formula": lambda df: (df["s_brake"] - 13).round()},
        # "v7": {"type": "dependent", "formula": lambda df: (7/8*df["vbrake"] + df["vI"]/8).round(2)},
        "t_trigger": {"type": "dependent", "formula": lambda df: df["vcrit"].apply(t_kr).round(3)},
    }

    # 2. Define the exact starting configuration of the machine
    initial_machine_state = {
        "hf1": 160,
        "hm2": 130,
        "metal": 695
    }

    # 3. Generate the DOE
    # Example: 5 temperature setups (groups), 10 variations each -> Total 50 tests
    doe_dataframe = generate_split_plot_doe(
        n_groups=8, 
        samples_per_group=16, 
        config=parameter_config, 
        initial_config=initial_machine_state,
        seed=42 # for reproducibility
    )
    
    # visual check
    print(doe_dataframe.head(20))

    # dropping useless information used only for sampling
    columns2print = doe_dataframe.columns.drop(["sm100_s4", "v3_vI_vcrit", "s34_ratio", "sbreak_shift"])

    # hardcoded reordering -> change it with change in parameter dict
    # columns2print = columns2print[[0, 4, 19, 20, 18, 5]].to_list() + columns2print[22:-1].to_list() + columns2print[[21, 6, -1]].to_list() + \
    #     columns2print[7:10].to_list() + columns2print[1:4].to_list() + columns2print[10:18].to_list()
    columns2print = columns2print[[0, 4, 19, 20, 18, 5]].to_list() + columns2print[[21, 6, -1]].to_list() + \
        columns2print[7:10].to_list() + columns2print[1:4].to_list() + columns2print[10:18].to_list()

    # 1-based indexing
    doe_dataframe.index += 1

    # plot
    # sns.pairplot(doe_dataframe[columns2print[:6] + columns2print[11:13] + columns2print[14:20]], hue="Group_ID")
    sns.pairplot(doe_dataframe[columns2print[:8] + columns2print[9:15]], hue="Group_ID")
    plt.show()

    # print to file
    doe_dataframe.to_csv("test.csv", columns=columns2print, index_label="Parameter_set")