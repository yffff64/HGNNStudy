"""
Build a single Excel workbook summarizing ALL node-classification results.

Sheets:
  - Results : one row per dataset x method (260), with mean/std, status,
              failure reason and elapsed time.
  - Summary : per-dataset success/failed counts and overall totals.

Data source: run_node_cls_status.json (written by run_all_node_cls.py).
"""

import json
import os
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STATUS_FILE = os.path.join(SCRIPT_DIR, 'run_node_cls_status.json')
OUT_XLSX = os.path.join(SCRIPT_DIR, 'node_cls_results.xlsx')

METHODS = [
    'MLP', 'CEGCN', 'CEGAT', 'HGNN', 'HyperGCN', 'HCHA', 'LEGCN',
    'HyperND', 'PhenomNN', 'SheafHyperGNN', 'HJRL', 'DPHGNN', 'TFHNN',
    'HNHN', 'UniGCNII', 'AllSetformer', 'EDHNN', 'HyperGT', 'EHNN', 'TMPHN'
]
DATASETS = [
    'cora', 'coauthor_cora', 'NTU2012', 'ModelNet40', 'actor', 'twitch',
    'pubmed', 'amazon', 'coauthor_dblp', 'walmart-trips-100',
    'yelp', 'pokec', 'trivago'
]

with open(STATUS_FILE, 'r', encoding='utf-8') as f:
    status = json.load(f)

# ---------- styles ----------
header_fill = PatternFill('solid', fgColor='2F5496')
header_font = Font(bold=True, color='FFFFFF', size=11)
center = Alignment(horizontal='center', vertical='center')
left = Alignment(horizontal='left', vertical='center')
thin = Side(style='thin', color='BFBFBF')
border = Border(left=thin, right=thin, top=thin, bottom=thin)
ok_fill = PatternFill('solid', fgColor='E2EFDA')
fail_fill = PatternFill('solid', fgColor='FCE4D6')
ds_font = Font(bold=True)
alt_fill = PatternFill('solid', fgColor='F2F2F2')

wb = Workbook()

# ===== Sheet 1: Results =====
ws = wb.active
ws.title = 'Results'
headers = ['Dataset', 'Method', 'Reproduced Mean (%)', 'Reproduced Std (%)',
           'Status', 'Failure Reason', 'Time (s)']
ws.append(headers)
for c in range(1, len(headers) + 1):
    cell = ws.cell(row=1, column=c)
    cell.fill = header_fill
    cell.font = header_font
    cell.alignment = center
    cell.border = border

row = 2
for di, ds in enumerate(DATASETS):
    for method in METHODS:
        key = f'{ds}/{method}'
        info = status.get(key)
        if info is None:
            mean_v = std_v = state = reason = t_v = ''
            state = 'pending'
        else:
            state = info.get('state', '')
            mean_v = info.get('mean', '')
            std_v = info.get('std', '')
            reason = '' if state == 'success' else info.get('reason', '')
            t_v = info.get('time_s', '')

        ws.cell(row=row, column=1, value=ds)
        ws.cell(row=row, column=2, value=method)
        ws.cell(row=row, column=3, value=mean_v)
        ws.cell(row=row, column=4, value=std_v)
        ws.cell(row=row, column=5, value=state)
        ws.cell(row=row, column=6, value=reason)
        ws.cell(row=row, column=7, value=t_v)

        for c in range(1, 8):
            cell = ws.cell(row=row, column=c)
            cell.border = border
            cell.alignment = left if c in (1, 2, 6) else center

        ws.cell(row=row, column=1).font = ds_font
        fill = ok_fill if state == 'success' else fail_fill
        for c in range(1, 8):
            ws.cell(row=row, column=c).fill = fill
        row += 1

ws.freeze_panes = 'A2'
widths = [20, 16, 20, 18, 10, 22, 10]
for i, w in enumerate(widths, start=1):
    ws.column_dimensions[get_column_letter(i)].width = w

# ===== Sheet 2: Summary =====
ws2 = wb.create_sheet('Summary')
ws2.append(['Dataset', 'Success', 'Failed/Skipped', 'Pending', 'Total'])
for c in range(1, 6):
    cell = ws2.cell(row=1, column=c)
    cell.fill = header_fill
    cell.font = header_font
    cell.alignment = center
    cell.border = border

tot_ok = tot_fail = tot_pend = 0
r = 2
for ds in DATASETS:
    ok = fail = pend = 0
    for method in METHODS:
        info = status.get(f'{ds}/{method}')
        if info is None:
            pend += 1
        elif info.get('state') == 'success':
            ok += 1
        else:
            fail += 1
    ws2.append([ds, ok, fail, pend, ok + fail + pend])
    tot_ok += ok
    tot_fail += fail
    tot_pend += pend
    for c in range(1, 6):
        cell = ws2.cell(row=r, column=c)
        cell.border = border
        cell.alignment = left if c == 1 else center
    r += 1

ws2.append(['TOTAL', tot_ok, tot_fail, tot_pend, tot_ok + tot_fail + tot_pend])
for c in range(1, 6):
    cell = ws2.cell(row=r, column=c)
    cell.font = Font(bold=True)
    cell.fill = PatternFill('solid', fgColor='DDEBF7')
    cell.border = border
    cell.alignment = left if c == 1 else center

for i, w in enumerate([20, 10, 16, 10, 8], start=1):
    ws2.column_dimensions[get_column_letter(i)].width = w

wb.save(OUT_XLSX)
print(f'Saved workbook: {OUT_XLSX}')
print(f'Result rows: {row - 2} | success={tot_ok} failed={tot_fail} pending={tot_pend}')
