#!/usr/bin/env python3
"""Summarize saved synthetic studies; no fitting or acceptance decisions."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence/completion"


def interval(k, n):
    """Two-sided 95% Wilson interval, conditional on n delivered estimates."""
    if n == 0:
        return None
    z = 1.959963984540054
    p = k / n
    denominator = 1 + z * z / n
    middle = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return dict(covered=k, denominator=n, rate=p, lower=middle-radius, upper=middle+radius)


def main():
    inputs = {}
    result = {"interval": "95% Wilson; coverage conditional on delivered estimates",
              "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "qualification": json.loads((OUT / "latest.json").read_text())}
    for kind in ["ar", "group", "repeated", "permutation"]:
        path = OUT / (kind + "-calibration.jsonl")
        inputs[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        cases = [json.loads(line) for line in path.read_text().splitlines()]
        rows = []
        for case in cases:
            if kind == "group":
                for method in case["methods"]:
                    rows.append(dict(scenario=case["scenario"], n=case["n"],
                                     attempts=case["replicates"], method=method["method"],
                                     inference=method["inference"], successes=method["successes"],
                                     failures=method["failures"],
                                     coverage=interval(method["covered"], method["successes"])))
            elif kind == "permutation":
                rows.append(dict(n=case["n"], method="Freedman-Lane" if case["freedman_lane"] else "sign flip",
                                 attempts=case["replicates"], features=case["features"], draws=case["draws"],
                                 family_error=interval(case["family_rejections"], case["replicates"]),
                                 marginal_error=interval(case["marginal_rejections"], case["replicates"])))
            else:
                row = {k: case[k] for k in ["scenario", "successes", "failures"]}
                row["attempts"] = case["replicates"]
                if kind == "ar":
                    row.update(same_feature=case["same_feature"],
                               coverage=interval(case["covered"], case["successes"]))
                else:
                    row.update(n=case["subjects"],
                               coverage=[interval(k, case["successes"]) for k in case["covered"]],
                               mean_prepare_ms=case["prepare_ms"] / case["replicates"],
                               mean_fit_ms=case["fit_ms"] / case["replicates"])
                rows.append(row)
        result[kind] = rows
    result["inputs"] = inputs
    (OUT / "calibration-summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
