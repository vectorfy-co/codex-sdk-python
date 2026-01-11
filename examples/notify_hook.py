"""Example notify hook that prints the JSON payload.

Usage:
  1) Add to ~/.codex/config.toml:
     notify = ["python3", "/absolute/path/to/examples/notify_hook.py"]
  2) Run codex via the SDK; the script will be invoked after each turn.
"""

import json
import sys


def main() -> None:
    if len(sys.argv) < 2:
        print("Missing payload")
        return
    try:
        payload = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        print(sys.argv[1])
        return
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
