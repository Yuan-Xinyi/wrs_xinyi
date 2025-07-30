import json

json_file = "kdt_query_time.jsonl"
time = []

with open(json_file, "r") as f:
    for line in f:
        time.append(json.loads(line))
print(f"Total query time: {sum(time) / 1_000_000:.4f} seconds")