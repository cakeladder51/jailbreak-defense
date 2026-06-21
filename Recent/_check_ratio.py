import csv

rows = list(csv.DictReader(open('results_parallel/results_Qwen2.5-3B.csv', encoding='utf-8')))
print('Columns:', list(rows[0].keys()))
benign = [r for r in rows if r.get('prompt_type') == 'benign']
print('Benign count:', len(benign))

ratio_col = None
for col in rows[0].keys():
    if 'ratio' in col.lower():
        ratio_col = col
        break
print('Ratio column:', ratio_col)

if ratio_col:
    ratios = sorted([(float(r.get(ratio_col) or 0), i) for i, r in enumerate(benign)], reverse=True)
    ratios = [(v, benign[i]) for v, i in ratios]
    print('Top 5 ratios:', [x[0] for x in ratios[:5]])
    hits = [(v, r) for v, r in ratios if v > 1.4]
    print(f'\n{len(hits)} hits > 1.4:')
    for v, r in hits:
        print(f'\nRATIO: {v}')
        print(f'PROMPT: {r.get("prompt", "")[:400]}')
        print(f'RESPONSE: {r.get("response", "")[:800]}')
        print('---')
