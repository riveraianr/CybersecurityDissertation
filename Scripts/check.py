with open(r'C:\Users\river\scan_results.txt', 'r') as f:
    lines = f.readlines()
print(f"Total lines: {len(lines)}")
labels_row = [1 if 'Priority: Yes' in line else 0 for line in lines[::2]]  # Row X lines
labels_detail = [1 if 'Priority: Yes' in line else 0 for line in lines[1::2]]  # Detail lines
print(f"Row labels (all): {sum(labels_row)} 1s out of {len(labels_row)}")
print(f"Detail labels (all): {sum(labels_detail)} 1s out of {len(labels_detail)}")
print(f"Sample rows: {lines[:4]}")