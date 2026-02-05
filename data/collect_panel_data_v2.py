#!/usr/bin/env python3
"""
Panel Data Collection Script v2 - Cost-Optimized

Uses the exact same queries that worked before, just looped over months.

Usage:
    python collect_panel_data_v2.py --project-id YOUR_PROJECT_ID --yes
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from calendar import monthrange
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.bigquery_client import create_bigquery_client
from analysis.revised.queries import VelocityQueries, ThroughputQueries, ComplexityQueries

DATA_DIR = Path(__file__).parent
PANEL_DIR = DATA_DIR / 'panel'
LOG_FILE = DATA_DIR / 'collection_log.json'


def load_log():
    if LOG_FILE.exists():
        with open(LOG_FILE) as f:
            return json.load(f)
    return {'collected': {}, 'errors': {}}


def save_log(log):
    log['last_updated'] = datetime.now().isoformat()
    with open(LOG_FILE, 'w') as f:
        json.dump(log, f, indent=2)


QUERY_CLASSES = {
    'velocity': VelocityQueries.pr_lead_time,
    'complexity': ComplexityQueries.pr_complexity,
    'throughput': ThroughputQueries.developer_throughput,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-id', required=True)
    parser.add_argument('--start-year', type=int, default=2021)
    parser.add_argument('--start-month', type=int, default=1)
    parser.add_argument('--end-year', type=int, default=2025)
    parser.add_argument('--end-month', type=int, default=6)
    parser.add_argument('--metrics', default='velocity,complexity,throughput')
    parser.add_argument('--skip-existing', action='store_true')
    parser.add_argument('--yes', '-y', action='store_true')
    parser.add_argument('--max-cost-per-query', type=float, default=5.0)

    args = parser.parse_args()

    metrics = [m.strip() for m in args.metrics.split(',')]

    # Generate months
    months = []
    y, m = args.start_year, args.start_month
    while (y, m) <= (args.end_year, args.end_month):
        months.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1

    print("=" * 60)
    print("PANEL DATA COLLECTION v2")
    print("=" * 60)
    print(f"Months: {len(months)}")
    print(f"Metrics: {metrics}")
    print(f"Max cost per query: ${args.max_cost_per_query}")

    if not args.yes:
        confirm = input("\nProceed? (y/n): ")
        if confirm.lower() != 'y':
            return

    client = create_bigquery_client(args.project_id)
    log = load_log()

    for metric in metrics:
        if metric not in QUERY_CLASSES:
            print(f"Unknown metric: {metric}")
            continue

        query_fn = QUERY_CLASSES[metric]
        output_dir = PANEL_DIR / metric
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"METRIC: {metric.upper()}")
        print('='*60)

        for year, month in months:
            key = f"{metric}_{year}_{month:02d}"
            output_path = output_dir / f"{metric}_{year}_{month:02d}.csv"

            if args.skip_existing and output_path.exists():
                print(f"  Skipping {year}-{month:02d} (exists)")
                continue

            if args.skip_existing and key in log.get('collected', {}):
                print(f"  Skipping {year}-{month:02d} (in log)")
                continue

            print(f"  Collecting {year}-{month:02d}...", end=" ", flush=True)

            try:
                query = query_fn(year, month)
                df = client.run_query(query, max_cost_usd=args.max_cost_per_query)

                if df is not None and not df.empty:
                    df.to_csv(output_path, index=False)
                    log['collected'][key] = {
                        'path': str(output_path),
                        'rows': len(df),
                        'collected_at': datetime.now().isoformat()
                    }
                    print(f"OK ({len(df)} rows)")
                else:
                    print("No data")

            except Exception as e:
                error_msg = str(e)
                if "exceeds max allowed" in error_msg:
                    print(f"TOO EXPENSIVE")
                else:
                    print(f"ERROR: {error_msg[:50]}")
                log['errors'][key] = error_msg

            save_log(log)
            time.sleep(1)  # Rate limit

    print("\n" + "=" * 60)
    print("COLLECTION COMPLETE")
    print("=" * 60)
    print(f"Collected: {len(log.get('collected', {}))}")
    print(f"Errors: {len(log.get('errors', {}))}")


if __name__ == '__main__':
    main()
