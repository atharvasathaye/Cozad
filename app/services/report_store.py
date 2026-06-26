import os
import json
from typing import Dict, Any

REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)


def save_report(report_id: str, report: Dict[str, Any]) -> str:
    path = os.path.join(REPORTS_DIR, f"{report_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return path


def load_report(report_id: str) -> Dict[str, Any]:
    path = os.path.join(REPORTS_DIR, f"{report_id}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Report not found: {report_id}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)