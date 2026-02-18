import sys

out = []

for fname in ['Data/expression_log.csv', 'Data/cnv_log.csv']:
    out.append(f"\n=== {fname} ===")
    with open(fname, 'r') as f:
        header = f.readline().strip()
    
    cols = header.split(',')
    out.append(f"  Header fields: {len(cols)}")
    out.append(f"  Header[0:3]: {cols[:3]}")
    
    row_names = []
    with open(fname, 'r') as f:
        f.readline()
        for i in range(5):
            line = f.readline().strip()
            if line:
                row_names.append(line.split(',')[0])
    out.append(f"  First 5 row-index values: {row_names}")

with open('_debug_output.txt', 'w') as f:
    f.write('\n'.join(out))

print("DONE - check _debug_output.txt")
