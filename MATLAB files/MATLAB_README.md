# MATLAB yield-curve fitting utilities

## Purpose and scope

This folder contains MATLAB utilities for representing and fitting material yield/hardening curves. The main workflow fits a **Whip-Bezier** curve to an already prepared stress-strain dataset. A combined **Swift-Voce** hardening law is included as a reference curve and can also be used to generate synthetic fitting data.

These files do **not** import raw DIC images, synchronize force and displacement measurements, calculate specimen geometry, or convert raw tensile-test measurements into stress and strain. Before using this code, the tensile/DIC pipeline must provide a numeric two-column curve:

```text
equivalent plastic strain    true stress [MPa]
```

The variable names in the code are `eps` and `sig_ref`. The implementations treat `eps` as equivalent plastic strain (`eqps`), with yielding located at `eps = 0`. If the available DIC data contains engineering strain/stress or total true strain, it must first be converted consistently. Elastic modulus, the 0.2% proof-stress procedure, UTS, necking, and fracture detection are outside the scope of this folder.

## Directory overview

```text
MATLAB files/
|-- fit/          Fit a Whip-Bezier curve and evaluate Swift-Voce curves
`-- yield_curve/  Reconstruct/export a sampled curve from Whip-Bezier parameters
```

The files `compute_bezier_point.m`, `define_whip_bezier_control_points.m`, `divide_interval.m`, and `quadratic_bezier_solve_for_middle_point.m` occur in both directories. They are duplicate local copies used to make each directory independently usable. Do not put both directories on the MATLAB path while modifying only one copy: MATLAB path order may silently select the other version.

## Quick start: fit experimental data

1. Prepare an `N x 2` text, CSV, or spreadsheet dataset. Column 1 must be equivalent plastic strain and column 2 true stress in MPa. Remove headers/NaNs as necessary and use monotonically increasing strain values.
2. Open `fit/least_sq_main_flex.m` and replace the hard-coded `exp_file` path with the path to the prepared data.
3. Choose:
   - `scale_e`: strain at the end of the Bezier part; it should cover the fitted strain range unless intentional linear extrapolation is desired.
   - `scale_s`: stress normalization, in the same unit as `sig_ref` (normally MPa).
   - `nb_total`: number of main Bezier points. The fitted vector then has `nb_total + 2` entries.
   - `params2optim`: number of leading parameters that remain free. The script fixes all later parameters to `0.75` in optimizer coordinates.
4. In MATLAB, change to the `fit` directory (or add that directory to the path) and run:

```matlab
least_sq_main_flex
```

This requires Optimization Toolbox because it uses `fmincon`. The main workspace results are:

- `parameters_fit`: fitted Whip-Bezier parameters in **optimizer coordinates**.
- `objective_value`: unweighted sum of squared stress residuals.
- `yc`: fitted stress values at the input strains (`[strain, stress]`).
- `cp_main`, `cp_secondary`: physical Bezier control points.
- `optim_data`: global iteration history populated by the output callback.

To save a process-analysis target table, explicitly export the identifiers, scales, and fitted vector, for example:

```matlab
yield_stress_MPa = parameters_fit(1) * scale_s;
bezier_params_signed = [parameters_fit(1); 2*parameters_fit(2:end)-1];

result = table(sample_id, yield_stress_MPa, scale_e, scale_s, ...
    nb_total, objective_value, {parameters_fit(:).'}, ...
    {bezier_params_signed(:).'}, ...
    'VariableNames', {'sample_id','yield_stress_MPa','scale_e','scale_s', ...
    'n_bezier_points','fit_sse','bezier_params_optimizer', ...
    'bezier_params_signed'});
writetable(result, 'tensile_curve_targets.csv');
```

`sample_id` must be supplied by the surrounding data pipeline. Keeping `scale_e`, `scale_s`, `nb_total`, preprocessing conventions, and parameter convention with every fitted vector is essential: the parameter vector is not self-describing without them.

## Parameter conventions

### Whip-Bezier

For `nb` main Bezier points, the signed parameter vector is

```text
[sy/scale_s, s1, d12, d23, ..., d(n-1,n), sn]
```

where:

- `sy/scale_s` is the normalized initial yield stress.
- `s1` and `sn` control the curve slope at the first and last main points.
- each `d` controls the slope of one segment of the main polygon.
- the vector length is `nb + 2`.

There are two different coordinate conventions:

| Context | First parameter | Remaining shape parameters |
|---|---:|---:|
| `parameters_fit`, `define_init`, `bezier_from_points`, `ls_objective_bezier_from_points` | non-negative normalized stress | `[0,1]` |
| `define_whip_bezier_control_points`, `yield_curve_bezier` | non-negative normalized stress | approximately `(-1,1)` |

Convert an optimizer result before passing it to the signed-interface functions:

```matlab
bezier_params_signed = [parameters_fit(1); 2*parameters_fit(2:end)-1];
```

The physical initial yield stress is:

```matlab
yield_stress = parameters_fit(1) * scale_s;
```

The Bezier part ends at `scale_e`. This is a normalization/domain choice, **not** a fitted yield strain or fracture strain.

### Swift-Voce

`compute_sw_point` implements the weighted combined law

```text
sigma = alpha*A*(eqps + e0)^n
      + (1-alpha)*(k0 + Q*(1-exp(-beta*eqps)))
```

and its analytical slope. Its seven-parameter order is:

```text
[alpha, A, e0, n, k0, Q, beta]
```

It also accepts a six-parameter reparameterization:

```text
[alpha*A, e0, n, (1-alpha)*k0, (1-alpha)*Q, beta]
```

`define_sw_params.m` only returns hard-coded literature/example values (currently SS2205, with DP590 commented out). The provided scripts do **not** optimize Swift-Voce parameters against experimental data. `least_sq_main.m` instead generates a Swift-Voce reference curve and fits Whip-Bezier to that curve, which is useful for comparing the representations but not for extracting properties from a new tensile test.

## Reconstruct a yield curve from fitted parameters

After fitting, use the files in `yield_curve/` to sample a curve:

```matlab
addpath('yield_curve')
bezier_params_signed = [parameters_fit(1); 2*parameters_fit(2:end)-1];
curve = yield_curve_bezier(bezier_params_signed, scale_e, scale_s, ...
    50, 1.0, 0.001);
```

The returned `curve` is an `N x 2` matrix `[equivalent plastic strain, stress]`. Each quadratic Bezier segment receives 50 sample points in this example. If `extrapolation_max > scale_e`, the function extends the final tangent line using `extrapolation_stepwidth`. Such extrapolation is a modeling assumption and should not be confused with measured post-necking or fracture behavior.

If stresses are needed only at specified strain values, use the optimizer-coordinate evaluator directly:

```matlab
stress_points = bezier_from_points(parameters_fit, scale_e, scale_s, eps_query);
```

`stress_points` is `[eps_query, predicted_stress]` and also uses linear extrapolation after the Bezier domain.

## File reference

### `fit/`

| File | Role | Directly useful to most users? |
|---|---|---|
| `least_sq_main_flex.m` | Main example for fitting Whip-Bezier parameters to a two-column experimental/reference curve. Contains a machine-specific input path that must be changed. Allows later parameters to be fixed. | **Yes: primary fitting entry point.** |
| `least_sq_main.m` | Generates a hard-coded Swift-Voce reference curve, fits Whip-Bezier to it, and plots interpolation/extrapolation. A commented block shows spreadsheet input. | Only for Swift-Voce/Whip-Bezier comparison and experimentation. |
| `ls_objective_bezier_from_points.m` | Evaluates Whip-Bezier stress at supplied strains and returns the sum of squared stress errors used by `fmincon`. | Yes, as part of fitting. |
| `bezier_from_points.m` | Evaluates an already fitted Whip-Bezier curve at specified strain values without computing a loss. | Yes, for reconstruction and derived targets. |
| `define_init.m` | Creates the initial vector and bounds for `fmincon`; first parameter is bounded `[0,10]`, shape parameters `[0,0.998]`. | Supporting fitting utility. |
| `bezier_least_sq_export_iter.m` | `fmincon` output callback that appends parameter, objective, and gradient history to global `optim_data`. | Optional diagnostics; not a material-property extractor. |
| `compute_sw_point.m` | Evaluates combined Swift-Voce stress and slope for supplied parameters and equivalent plastic strain. | Yes only when using/comparing Swift-Voce. |
| `define_sw_params.m` | Supplies hard-coded SS2205 example parameters; commented DP590 values are also present. | No for new experimental data unless replaced with validated parameters. |
| `define_whip_bezier_control_points.m` | Converts signed Whip-Bezier parameters and scales into main/secondary physical control points. | Internal geometry utility. |
| `divide_interval.m` | Maps ordered shape parameters in `[-1,1]` into a monotonically divided angular interval. | Internal utility. |
| `quadratic_bezier_solve_for_middle_point.m` | Finds a quadratic Bezier middle control point from two endpoints and endpoint slopes. | Internal utility. |
| `compute_bezier_point.m` | Samples a Bezier curve from an arbitrary control polygon using Bernstein polynomials. It is not called by the current fitting scripts. | General/legacy helper. |

### `yield_curve/`

| File | Role | Directly useful to most users? |
|---|---|---|
| `yield_curve_bezier.m` | Main export/reconstruction function: samples all quadratic segments and optionally adds linear extrapolation. | **Yes: primary reconstruction entry point.** |
| `define_whip_bezier_control_points.m` | Duplicate of the fitting geometry function. | Internal utility. |
| `compute_bezier_point.m` | Samples each quadratic segment requested by `yield_curve_bezier`. | Internal utility. |
| `divide_interval.m` | Duplicate interval-mapping helper. | Internal utility. |
| `quadratic_bezier_solve_for_middle_point.m` | Duplicate control-point solver. | Internal utility. |

## Outputs relevant to process analysis

For a model that maps casting/process variables to tensile behavior, the most useful output from this folder is the fitted Whip-Bezier representation:

- `parameters_fit` plus `scale_e`, `scale_s`, and `nb_total` as compact full-curve targets.
- `yield_stress_MPa = parameters_fit(1) * scale_s` as a scalar target.
- reconstructed stresses at fixed plastic-strain values, obtained with `bezier_from_points`, as robust and interpretable scalar targets.
- fit quality (`objective_value`, preferably complemented by RMSE and a normalized error) for filtering unreliable fits.

The following commonly requested properties are **not currently calculated**:

- yield strain or 0.2% proof strain;
- Young's modulus;
- ultimate tensile strength (UTS);
- uniform elongation / onset of necking;
- fracture strain or total elongation;
- engineering-to-true stress/strain conversion.

Those properties require the original elastic and/or post-yield tensile curve and explicit extraction rules. A fitted hardening curve expressed versus equivalent plastic strain starts at yield and therefore cannot recover all of them by itself.

## Practical cautions and known limitations

- `least_sq_main_flex.m` contains an absolute Windows path and is an example script, not a reusable batch pipeline.
- No fit results are saved automatically; variables remain in the MATLAB workspace unless explicitly exported.
- The objective is an unweighted sum of squared stress errors. Dense regions of the input curve therefore influence the fit more strongly.
- Units are not checked. `scale_s` and `sig_ref` must use the same stress unit; strain and `scale_e` must use the same strain convention.
- No validation rejects NaNs, duplicate/non-monotonic strain values, negative discriminants from invalid geometries, or optimizer failures.
- Linear extrapolation beyond `scale_e` is mathematical continuation, not experimental evidence.
- The code describes a yield/hardening curve, not the complete tensile curve through necking and failure.
- The example scripts open figures and use workspace/global state. For many specimens, wrap the fit in a function that returns a result structure and records the optimizer exit flag.

## Recommended batch-workflow extension

For production use across many tensile specimens, create one wrapper function that:

1. loads and validates the prepared true-stress/equivalent-plastic-strain data;
2. selects consistent `scale_e`, `scale_s`, `nb_total`, and bounds for every specimen;
3. runs `fmincon` and records `parameters_fit`, objective value, exit flag, and RMSE;
4. reconstructs the curve on a common strain grid for quality control;
5. exports one row per specimen together with specimen/process identifiers;
6. separately derives scalar tensile properties from the original full tensile curve, not from the hardening fit where the necessary information is absent.

This separation preserves both useful target types: compact Whip-Bezier parameters for full-curve prediction and conventional scalar properties for interpretable process analysis.
