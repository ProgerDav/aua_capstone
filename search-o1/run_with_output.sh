#!/bin/bash

# Output file
output_file="run_output.txt"

# Start time
start_time=$(date +%s)

# Run the script and capture all output (both stdout and stderr)
{ time bash run.sh; } &> "$output_file"

# End time
end_time=$(date +%s)

# Calculate runtime in seconds
runtime=$((end_time - start_time))

# Append runtime information to the output file
echo -e "\n\nTotal execution time: ${runtime} seconds ($(date -u -d @${runtime} +"%H:%M:%S"))" >> "$output_file"

echo "Execution completed. All output saved to $output_file"