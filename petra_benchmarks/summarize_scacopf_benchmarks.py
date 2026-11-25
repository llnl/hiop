#!/usr/bin/env python3
"""Summarize SC-ACOPF Slurm results (not the synthetic/simple benchmarks).

Input:  a run directory made by submit_scacopf_benchmarks.sh.
Output: summary.csv, summary.tex, and summary.txt in the same directory.

For each case/time the EF reference is read from PETRA's --ef log; the Julia
runner does not put the EF objective in its solver CSV. Missing EF references
are shown as --, never replaced with the PETRA objective.
"""

import argparse
import csv
import math
import re
from pathlib import Path

SOLVERS = ('petra', 'als', 'alps')
GOOD_STATUS = {'petra': 'converged', 'als': 'first_order', 'alps': 'first_order'}
NUMBER = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?'
EF_OBJECTIVE_RE = re.compile(r'^\s*EF objective:\s*(' + NUMBER + r')\s*$', re.MULTILINE)
EF_SECONDS_RE = re.compile(r'^\s*EF solve time:\s*(' + NUMBER + r')\s*s\s*$', re.MULTILINE)

FIELDS = (
    'case', 'minutes_since_base', 'solver', 'status', 'exitcode',
    'objective', 'ef_objective', 'delta_percent',
    'outer_iterations', 'inner_iterations', 'prox_evaluations',
    'master_feasibility', 'bound_violation', 'consensus_residual',
    'elapsed_seconds', 'ef_seconds', 'error',
    'raw_file', 'rop_file', 'inl_file', 'con_file',
)


def finite_float(text):
    try:
        value = float(str(text).strip().replace('D', 'E').replace('d', 'e'))
    except (ValueError, TypeError):
        return None
    return value if math.isfinite(value) else None


def read_one_csv(path):
    if not path.is_file():
        return None, 'missing result CSV'
    try:
        with path.open(newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
    except (OSError, csv.Error) as exc:
        return None, f'could not parse CSV: {exc}'
    if len(rows) != 1:
        return None, f'expected one CSV row, found {len(rows)}'
    return rows[0], ''


def parse_ef_log(path):
    if not path.is_file():
        return None, None
    content = path.read_text(encoding='utf-8', errors='replace')
    matches = EF_OBJECTIVE_RE.findall(content)
    seconds = EF_SECONDS_RE.findall(content)
    reference = finite_float(matches[-1]) if matches else None
    elapsed = finite_float(seconds[-1]) if seconds else None
    return reference, elapsed


def read_manifest(path):
    with path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        required = ('case', 'minutes_since_base', 'raw_file', 'rop_file', 'inl_file', 'con_file')
        if not reader.fieldnames or not set(required).issubset(reader.fieldnames):
            raise ValueError(f'{path}: missing expected header fields {required}')
        cases = list(reader)
    if not cases:
        raise ValueError(f'{path}: no cases')
    for i, case in enumerate(cases):
        if not case['case'] or not case['minutes_since_base']:
            raise ValueError(f'Incomplete manifest row {i}')
    return cases


def gather(run_dir):
    cases = read_manifest(run_dir / 'manifest.tsv')
    rows = []
    for combo_index, case in enumerate(cases):
        stem = f"case{combo_index:03d}_{case['case']}"
        # --ef is currently executed only as a post-PETRA comparison.
        f_ef, t_ef = parse_ef_log(run_dir / f'{stem}_petra.log')
        for solver in SOLVERS:
            task_stem = f'{stem}_{solver}'
            data, parse_error = read_one_csv(run_dir / f'{task_stem}.csv')
            exit_path = run_dir / f'{task_stem}.exitcode'
            exitcode = exit_path.read_text(encoding='utf-8').strip() if exit_path.is_file() else ''
            data = data or {}
            objective = finite_float(data.get('objective'))
            delta = None
            if f_ef is not None and objective is not None:
                # Match the Julia driver's relative-difference denominator.
                delta = 100.0 * (objective - f_ef) / max(abs(f_ef), 1.0)
            errors = []
            if parse_error:
                errors.append(parse_error)
            if exitcode and exitcode != '0':
                errors.append(f'job exited {exitcode}')
            if f_ef is None:
                errors.append('EF objective unavailable')
            if data and data.get('status', '') != GOOD_STATUS[solver]:
                errors.append(f"solver status={data.get('status', '')}")
            if data.get('error', ''):
                errors.append(data['error'])

            rows.append({
                'case': case['case'],
                'minutes_since_base': case['minutes_since_base'],
                'solver': solver,
                'status': data.get('status', 'missing_csv') if data else 'missing_csv',
                'exitcode': exitcode,
                'objective': data.get('objective', ''),
                'ef_objective': format(f_ef, '.15g') if f_ef is not None else '',
                'delta_percent': format(delta, '.15g') if delta is not None else '',
                'outer_iterations': data.get('outer_iterations', ''),
                'inner_iterations': data.get('inner_iterations', ''),
                'prox_evaluations': data.get(
                    'oracle_calls' if solver == 'petra' else 'scenario_prox_calls', ''
                ),
                'master_feasibility': data.get('master_feasibility', ''),
                'bound_violation': data.get('bound_violation', ''),
                'consensus_residual': data.get('consensus_residual', ''),
                'elapsed_seconds': data.get('elapsed_seconds', ''),
                'ef_seconds': format(t_ef, '.15g') if t_ef is not None else '',
                'error': '; '.join(errors),
                **{k: case[k] for k in ('raw_file', 'rop_file', 'inl_file', 'con_file')},
            })
    return cases, rows


def slash_iterations(row):
    if not row:
        return '--'
    outer = str(row['outer_iterations']).strip()
    inner = str(row['inner_iterations']).strip()
    return f'{outer}/{inner}' if outer and inner else '--'


def prox(row):
    if not row:
        return '--'
    value = str(row['prox_evaluations']).strip()
    return value or '--'


def scientific_percent(value):
    f = finite_float(value)
    if f is None:
        return '--'
    mantissa, exponent = f'{f:+.1e}'.split('e')
    return f'{mantissa}e{int(exponent):+d}'


def tex_escape(s):
    return re.sub(r'([\\_%&#$\{\}])', lambda m: '\\' + m.group(1), str(s))


def latex_table(cases, rows):
    by_case = {(r['case'], r['minutes_since_base'], r['solver']): r for r in rows}
    out = [
        r'\begin{table}[t]',
        r'\centering',
        r'\small',
        r'\setlength{\tabcolsep}{3pt}',
        r'\begin{tabular}{l|rrr|rrr|rrr}',
        r'\toprule',
        r'& \multicolumn{3}{c|}{ProxNlp} & \multicolumn{3}{c|}{ALS} & \multicolumn{3}{c}{ALPS} \\',
        r'\midrule',
        r'\textbf{Problem} & Iter. & Prox. & $\Delta f$ (\%) & Iter. & Prox. & $\Delta f$ (\%) & Iter. & Prox. & $\Delta f$ (\%) \\',
        r'\midrule',
    ]
    for i, case in enumerate(cases):
        key = (case['case'], case['minutes_since_base'])
        label = tex_escape(case['case']) + ' (' + tex_escape(case['minutes_since_base']) + ' min)'
        cells = [label]
        for solver in SOLVERS:
            row = by_case.get((*key, solver))
            cells.extend((slash_iterations(row), prox(row), scientific_percent(row['delta_percent']) if row else '--'))
        out.append(' & '.join(cells) + r' \\')
        if i != len(cases) - 1:
            out.append(r'\midrule')
    out.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\caption{SC-ACOPF benchmarks. Iter. reports outer/inner iterations and Prox. reports '
        r'proximal-oracle evaluations. $\Delta f=100(f_{\mathrm{method}}-f_{\mathrm{EF}})/'
        r'\max\{|f_{\mathrm{EF}}|,1\}$, expressed in percent relative to the '
        r'extensive-form (EF) reference for the same case and response time. '
        r'The EF reference is computed after PETRA. Dashes denote unavailable results; '
        r'check the accompanying CSV for convergence status and feasibility.}',
        r'\label{tab:scacopf_benchmark_counts}',
        r'\end{table}',
    ])
    return '\n'.join(out) + '\n'


def text_report(cases, rows):
    lines = ['SC-ACOPF benchmark summary', '']
    for case in cases:
        label, minute = case['case'], case['minutes_since_base']
        grouped = [r for r in rows if r['case'] == label and r['minutes_since_base'] == minute]
        f_ef = grouped[0]['ef_objective'] if grouped else ''
        lines.append(f'{label} -- minutes_since_base={minute}, EF={f_ef or "UNAVAILABLE"}')
        for r in grouped:
            lines.append(
                f"  {r['solver']:5s} status={r['status']:20s} "
                f"iter={slash_iterations(r):>11s} prox={prox(r):>9s} "
                f"objective={r['objective'] or '--':>18s} "
                f"delta%={scientific_percent(r['delta_percent']):>12s} "
                f"exit={r['exitcode'] or '?'}"
            )
            if r['error']:
                lines.append('    NOTE: ' + r['error'])
        lines.append('')
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path, help='Directory created by submit_scacopf_benchmarks.sh')
    args = parser.parse_args()
    directory = args.run_dir.expanduser().resolve()
    cases, rows = gather(directory)

    with (directory / 'summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (directory / 'summary.tex').write_text(latex_table(cases, rows), encoding='utf-8')
    report = text_report(cases, rows)
    (directory / 'summary.txt').write_text(report, encoding='utf-8')
    print(report)
    print('Wrote:', directory / 'summary.csv')
    print('Wrote:', directory / 'summary.tex')
    print('Wrote:', directory / 'summary.txt')


if __name__ == '__main__':
    main()
