# curl_probe — README

Short description

- **What it is**: A simple helper script that uses the external `curl` binary to measure HTTP response time (total time) for requests and stores per-port measurements as JSON files.

Requirements

- **Python**: Python 3 (script uses stdlib only).
- **curl**: The `curl` binary must be installed and available on the system `PATH`.

How the script currently works

- The script file is `scripts/curl_probe.py` and is intended to be run directly with Python.
- There is no command-line argument parsing implemented — the `__main__` section runs with a set of built-in defaults (see "Defaults and behavior" below).
- Running the script will iterate over a range of ports and for each port perform a number of probes for several file sizes, printing progress to stdout and writing results to JSON files in the `results/` directory.
- The probe itself is provided by the `probe(url, timeout=..., curl_path='curl')` function which calls `curl` and returns either a `float` (seconds) or `None` on failure.

Defaults and behavior

- Default base URL: `http://tcpdynamics.uk` (variable `url` in `_main`).
- Default ports: `4000..4039` (a range created in `_main`).
- Default filesizes probed: `['16K', '64K', '256K', '1M', '2M', '4M']`.
- Default iterations per filesize: `50`.
- The script sleeps briefly between iterations (`0.5s`) and also sleeps `10s` between ports to avoid overwhelming the server.
- For each port the script writes a JSON file to the repository `results/` directory named `<port>.json`.

JSON output format

Each per-port JSON file contains the following keys:

- **port**: integer port number probed.
- **base_url**: the base URL used (string).
- **filesizes**: list of filesize path components that were probed (list of strings).
- **iterations**: number of iterations per filesize (integer).
- **measurements**: mapping (object) from filesize string to a list of measured values. Each list entry is either a `float` (response time in seconds) or `null` (when `curl` failed or the probe dropped).

Example (trimmed) JSON structure

```
{
    "port": 4035,
    "base_url": "http://tcpdynamics.uk",
    "filesizes": ["16K", "64K", "256K"],
    "iterations": 50,
    "measurements": {
        "16K": [0.012345, 0.010987, null, ...],
        "64K": [0.045678, 0.043210, ...]
    }
}
```

Using the probe function from other code

You can import the `probe` function directly and call it from your own Python code. `probe` returns a `float` (seconds) on success or `None` on failure.

```py
from scripts.curl_probe import probe

result = probe('http://tcpdynamics.uk:4035/256K', timeout=5)
if result is None:
        print('request failed or packet dropped')
else:
        print(f'response time: {result:.6f}s')
```

There is also a helper `probe_and_store(target_url, port, filesizes, iterations, results_dir)` in the same module that runs multiple probes and writes the JSON file for a single port. You can call that from your own script if you prefer programmatic control.

How to run the script

Run the script as-is to execute with the built-in defaults:

```bash
python3 scripts/curl_probe.py
```

Notes and customization

- Because the script currently contains hard-coded defaults in `_main`, edit the source if you want to change the base URL, port range, filesizes or number of iterations. The constants to change are defined near the top of `_main()`.
- If you prefer command-line configurable behavior, consider adding `argparse` parsing in `_main()` and using the provided `probe` and `probe_and_store` functions.
- The script captures `curl` output; if `curl` is not found on `PATH`, `probe()` returns `None`.

Contact / Improvements

- If you'd like, I can help add CLI argument parsing, a progress-summary writer, or an option to upload/aggregate results. Tell me which feature you want next.

