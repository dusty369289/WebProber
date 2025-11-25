#!/usr/bin/env python3
"""Simple probe that uses curl to measure response time.

Sends a request with `curl` and returns the response time in seconds
as a float when successful, or `None` when the request fails or the
packet is dropped.

Requires: `curl` available on PATH.
"""
from __future__ import annotations

import subprocess
import json
import math
import argparse
from pathlib import Path
from typing import Optional
import time


def probe(url: str, timeout: int = 99, curl_path: str = "curl") -> Optional[float]:
    """Call `curl` to probe `url` and return time_total in seconds.

    Returns:
      float: response time in seconds on success
      None: when curl exits with non-zero status (timeout/drop/err)
    """
    # -s: silent, -o /dev/null: discard body, -w '%{time_total}': print total time
    cmd = [
        curl_path,
        "-s",
        "-o",
        "/dev/null",
        "-w",
        "%{time_total}",
        "--max-time",
        str(timeout),
        url,
    ]

    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        # curl not installed / not found
        return None

    if res.returncode != 0:
        return None

    out = res.stdout.strip()
    try:
        return float(out)
    except (ValueError, TypeError):
        return None
    
def probe_and_store(target_url: str, port: int, filesizes: list[str], iterations: int, results_dir: Path) -> None:
    """Probe the given URL multiple times and store results to a JSON file."""
    measurements: dict[str, list[Optional[float]]] = {}
    for fsize in filesizes:
        text = "Size {}".format(fsize)
        print("=" * math.floor((64-len(text))/2)+ text + "=" * math.ceil((64-len(text))/2))
        full_url = f"{target_url}:{port}/{fsize}"
        runs: list[Optional[float]] = []
        for i in range(iterations):
            timeout = int(3) # dynamic timeout based on filesize
            result = probe(full_url, timeout=int(timeout))
            if result is None:
                result_str = "==None=="
            else:
                result_str = f"{result:.6f}"
            fstring = f"{fsize}"
            if len(fstring) < 4:
                fstring = fstring + " "
            while len(fstring) < 4:
                fstring = " " + fstring
            print(f"Port {port} | Size {fstring} | Iteration {i+1:02d}/{iterations:02d} | Timeout {int(timeout):02d} -> {result_str}")
            runs.append(result)
            time.sleep(0.5)  # brief pause between iterations
        measurements[fsize] = runs
        

    # store per-port results to JSON file named by port number
    out = {
        "port": port,
        "base_url": target_url,
        "filesizes": filesizes,
        "iterations": iterations,
        "measurements": measurements,
    }
    out_path = results_dir / f"{port}.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)

    print(f"Wrote results for port {port} -> {out_path} (iterations={iterations})")


def _main() -> None:
    iterations = 50

    ports = [x for x in range(4000,4040)]
    #ports = [4009]
    url = "http://tcpdynamics.uk"
    filesizes = ["16K", "64K", "256K", "1M", "2M", "4M",]
    # ensure results directory exists (placed at repo root alongside `scripts/`)
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    for port in ports:
        text = "Port {}".format(port)
        print("=" * 64)
        print("=" * math.floor((64-len(text))/2)+ text + "=" * math.ceil((64-len(text))/2))
        print("=" * 64)
        probe_and_store(
            target_url=url,
            port=port,
            filesizes=filesizes,
            iterations=iterations,
            results_dir=results_dir,
        )
        print("Sleeping for 10 seconds before next port...")
        time.sleep(10)  # sleep between ports to avoid overwhelming the server


if __name__ == "__main__":
    _main()
