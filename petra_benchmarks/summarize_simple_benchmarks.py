#!/usr/bin/env python3
"""Merge run_benchmarks.jl CSVs, solver logs, and portfolio postprocessing.

Usage:
  python3 summarize_simple_benchmarks.py /absolute/path/to/results/run
Creates summary.csv and summary.tex in that directory. Missing jobs remain
explicitly marked 'missing_csv' rather than being mistaken for convergence.
The thresholded portfolio objective is read from the Julia log, which prints
'Thresholded objective:' after zeroing asset weights below 1e-8.
"""
import argparse
import csv
import math
import re
from pathlib import Path
PROBLEMS = ('absolute', 'capacity', 'doublewell', 'portfolio')
SOLVERS = ('petra', 'als', 'alps')
PORTFOLIO_SEEDS = (43, 44, 45)
CASES = (
    [(p, None) for p in PROBLEMS if p != 'portfolio']
    + [('portfolio', seed) for seed in PORTFOLIO_SEEDS]
)
FIELDS = (
    'problem', 'seed', 'solver', 'variant', 'outer_iterations', 'inner_iterations',
    'master_nlp_attempts', 'scenario_prox_evaluations', 'accepted_steps',
    'rejected_steps', 'inconclusive_steps', 'status', 'objective',
    'objective_thresholded', 'master_feasibility', 'consensus_residual',
    'elapsed_seconds', 'error',
)

def petra_step_counts(log_path):
    """Infer '?' classifications from the documented 14-column verbose table.

    Accept/reject counters are read from the final summary, since an
    inconclusive ratio test may subsequently count as an accepted step.
    """
    if not log_path.is_file():
        return '', '', ''
    contents = log_path.read_text(encoding='utf-8', errors='replace')
    def get_field(field):
        match = re.search(r'^\s*' + re.escape(field) + r'\s+(\d+)\s*$',
                          contents, flags=re.MULTILINE)
        return match.group(1) if match else ''
    inconclusive = 0
    for line in contents.splitlines():
        tokens = line.split()
        if (len(tokens) == 14 and tokens[0].isdigit() and
                tokens[1].isdigit() and tokens[-1] == '?'):
            inconclusive += 1
    # If the trace is absent (e.g. a failed run), return blank, not zero.
    has_trace = re.search(r'\bIter\s+Stage\s+F_orig\b', contents) is not None
    return (get_field('accepted_steps'), get_field('rejected_steps'),
            str(inconclusive) if has_trace else '')

def portfolio_thresholded_objective(log_path):
    """Extract the reevaluated objective after thresholding portfolio weights.
    The current Julia runner prints 'Thresholded objective:', while an earlier
    version printed 'Projected objective:'. Accept both for existing logs.
    """
    if not log_path.is_file():
        return ''
    contents = log_path.read_text(encoding='utf-8', errors='replace')
    pattern = (
        r'^\s*(?:Thresholded|Projected) objective:\s*'
        r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$'
    )
    matches = re.findall(pattern, contents, flags=re.MULTILINE | re.IGNORECASE)
    return matches[-1] if matches else ''

def read_single_row(path):
    if not path.is_file():
        return None
    with path.open(newline='', encoding='utf-8') as stream:
        data = list(csv.DictReader(stream))
    if len(data) != 1:
        raise ValueError(f'{path} should contain exactly one result row; found {len(data)}')
    return data[0]

def gather(run_dir):
    rows = []
    for problem, seed in CASES:
        for solver in SOLVERS:
            stem = (f'{problem}_{solver}' if seed is None
                    else f'portfolio_seed{seed}_{solver}')
            data = read_single_row(run_dir / f'{stem}.csv')
            result = {field: '' for field in FIELDS}
            result.update(problem=problem, seed=str(seed) if seed is not None else '',
                          solver=solver)
            if data is None:
                result['status'] = 'missing_csv'
                exit_path = run_dir / f'{stem}.exitcode'
                if exit_path.exists():
                    result['error'] = f'job exit code {exit_path.read_text().strip()}'
                rows.append(result)
                continue
            for field in ('variant', 'outer_iterations', 'inner_iterations',
                          'status', 'objective', 'master_feasibility',
                          'consensus_residual', 'elapsed_seconds', 'error'):
                result[field] = data.get(field, '')
            if problem == 'portfolio':
                result['objective_thresholded'] = (
                    data.get('objective_thresholded') or
                    portfolio_thresholded_objective(run_dir / f'{stem}.log')
                )
            if solver == 'petra':
                # A master NLP is attempted once in each PETRA inner iteration.
                result['master_nlp_attempts'] = data.get('inner_iterations', '')
                result['scenario_prox_evaluations'] = data.get('oracle_calls', '')
                accepted, rejected, inconclusive = petra_step_counts(
                    run_dir / f'{stem}.log'
                )
                result['accepted_steps'] = data.get('accepted_steps') or accepted
                result['rejected_steps'] = data.get('rejected_steps') or rejected
                result['inconclusive_steps'] = data.get('inconclusive_steps') or inconclusive
            else:
                # ALS/ALPS solve PANOC+ inner problems; no master NLP/ratio test.
                result['scenario_prox_evaluations'] = data.get('scenario_prox_calls', '')
            rows.append(result)
    return rows

def format_objective(value):
    """Display an objective with 8 significant digits; preserve the raw CSV."""
    try:
        number = float(value)
    except (ValueError, TypeError):
        return '--'
    return f'{number:.8g}' if math.isfinite(number) else '--'


def tex_escape(s):
    return (str(s).replace('\\', r'\textbackslash{}').replace('_', r'\_')
            .replace('&', r'\&').replace('%', r'\%').replace('#', r'\#'))

def tex_table(rows):
    """Compact PETRA/ALS/ALPS comparison; portfolio uses thresholded costs."""
    def cell(row, field):
        if row is None:
            return '--'
        value = row.get(field)
        if value is None or str(value).strip() == '':
            return '--'
        return tex_escape(value)

    def joined(row, *fields):
        if row is None:
            return '--'
        values = [cell(row, field) for field in fields]
        return '/'.join(values) if any(v != '--' for v in values) else '--'

    def comparable_objective(row):
        if row is None:
            return None
        # Do not silently substitute raw portfolio objectives if the
        # post-threshold objective is unavailable for any of the methods.
        key = ('objective_thresholded' if row['problem'] == 'portfolio'
               else 'objective')
        try:
            val = float(row.get(key, ''))
        except (ValueError, TypeError):
            return None
        return val if math.isfinite(val) else None

    def signed_difference(petra, other):
        base = comparable_objective(petra)
        value = comparable_objective(other)
        if base is None or value is None:
            return '--'
        # Signed method-minus-PETRA difference, e.g. +1.2e-8 or -3.4e-7.
        return re.sub(r'e([+-])0+(\d+)$', r'e\1\2', f'{value - base:+.1e}')

    groups = {}
    for row in rows:
        key = (row['problem'], str(row.get('seed') or ''))
        groups.setdefault(key, {})[row['solver']] = row

    out = [
        r'\begin{table}[t]',
        r'\centering',
        r'\small',
        r'\begin{tabular}{l|ccc|ccc|ccc}',
        r'\toprule',
        r'& \multicolumn{3}{c|}{PETRA} & \multicolumn{3}{c|}{ALS} & \multicolumn{3}{c}{ALPS} \\',
        r'\midrule',
        r'Problem & Iter. & Prox. & Steps & Iter. & Prox. & $\Delta f$ & Iter. & Prox. & $\Delta f$ \\',
        r'\midrule',
    ]

    for (problem, seed), methods in groups.items():
        petra, als, alps = (methods.get(s) for s in SOLVERS)
        label = f'{problem}-{seed}' if seed else problem
        columns = [
            tex_escape(label),
            joined(petra, 'outer_iterations', 'inner_iterations'),
            cell(petra, 'scenario_prox_evaluations'),
            joined(petra, 'accepted_steps', 'rejected_steps',
                   'inconclusive_steps'),
            joined(als, 'outer_iterations', 'inner_iterations'),
            cell(als, 'scenario_prox_evaluations'),
            signed_difference(petra, als),
            joined(alps, 'outer_iterations', 'inner_iterations'),
            cell(alps, 'scenario_prox_evaluations'),
            signed_difference(petra, alps),
        ]
        out.append(' & '.join(columns) + r' \\')
        out.append(r'\midrule')

    if groups:
        out[-1] = r'\bottomrule'
    else:
        out.append(r'\bottomrule')

    out += [
        r'\end{tabular}',
        r'\caption{Simple benchmarks. Iter.: outer/inner iterations; Prox.:',
        r'proximal-oracle evaluations; Steps: PETRA accepted/rejected/inconclusive',
        r'(accepted and inconclusive counts can overlap).',
        r'$\Delta f=f_{\mathrm{method}}-f_{\mathrm{PETRA}}$ is signed.',
        r'For portfolio, all objectives in $\Delta f$ are reevaluated after',
        r'setting asset weights with $|w_i|<10^{-8}$ to zero.',
        r'Dashes denote unavailable values.}',
        r'\label{tab:simple_benchmark_counts}',
        r'\end{table}',
    ]
    return '\n'.join(out) + '\n'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    args = parser.parse_args()
    run_dir = args.run_directory.expanduser().resolve()
    if not run_dir.is_dir():
        parser.error(f'{run_dir} is not a directory')
    rows = gather(run_dir)
    with (run_dir / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (run_dir / 'summary.tex').write_text(tex_table(rows), encoding='utf-8')
    for row in rows:
        name = row['problem']
        if row['seed']:
            name += f"-{row['seed']}"
        print(f"{name:15} {row['solver']:5} {row['status']:22} "
              f"objective={format_objective(row['objective']):>14} "
              f"thresholded={format_objective(row['objective_thresholded']):>14} "
              f"inner={row['inner_iterations'] or '-':>5} "
              f"prox={row['scenario_prox_evaluations'] or '-':>7}")
    print(f'Wrote {run_dir / "summary.csv"} and {run_dir / "summary.tex"}')

if __name__ == '__main__':
    main()
