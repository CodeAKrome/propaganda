#!/usr/bin/env python3
import sys
from datetime import datetime, timedelta, timezone

# Default to 1 day if no argument provided
days = 1

# Check if command line argument is provided
if len(sys.argv) > 1:
    try:
        days = int(sys.argv[1])
    except ValueError:
        print(f"Error: '{sys.argv[1]}' is not a valid integer", file=sys.stderr)
        sys.exit(1)

# Get current time in UTC
now_utc = datetime.now(timezone.utc)

# Subtract specified number of days
previous_day_utc = now_utc - timedelta(days=days)

# Format as ISO 8601 with 'Z' suffix
previous_timestamp = previous_day_utc.strftime('%Y-%m-%dT%H:%M:%SZ')

# Print it (or save to file if needed)
print(previous_timestamp)

# Optional: save to file
with open('db/timestamp.txt', 'w') as f:
    f.write(previous_timestamp)