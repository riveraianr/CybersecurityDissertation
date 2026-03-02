with open(r'C:\Users\river\scan_results_new.txt', 'r') as f:
    lines = f.readlines()
labels = [1 if 'Priority: Yes' in line else 0 for line in lines if 'Priority:' in line]
print(f"Total labels: {len(labels)}")
print(f"First 10 labels: {labels[:10]}")
print(f"Unique classes: {set(labels)}")
print(f"Count of 1s: {sum(labels)}")