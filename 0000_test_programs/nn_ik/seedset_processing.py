import json
import numpy as np

filename = "successful_seed_dataset.json"
output_filename = "effective_seedset_1M.npz"

data = []
with open(filename, "r") as f:
    for line in f:
        try:
            # Remove the empty JSON object `{}` at the beginning of the first line (if present)
            if line.startswith("{}"):
                line = line[2:].strip()  # Remove `{}` and strip any extra spaces
            parsed_line = json.loads(line)  # Parse the remaining valid JSON object
            data.append(parsed_line)  # Add the parsed JSON object to the result list
        except json.JSONDecodeError as e:
            print(f"Failed to parse line: {line.strip()}, Error: {e}")  # Log lines with parsing errors

# Convert the data to NumPy arrays or a structured format
# Assuming the data is a list of dictionaries with consistent keys:
keys = data[0].keys() if data else []  # Get keys from the first dictionary
arrays = {key: np.array([item[key] for item in data]) for key in keys}

# Save the data as an .npz file
np.savez(output_filename, **arrays)

print(f"Data successfully saved to {output_filename}")
