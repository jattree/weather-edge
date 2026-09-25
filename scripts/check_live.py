"""Quick check of live trading status (read-only).

    python scripts/check_live.py                 # default project DB
    python scripts/check_live.py --db path/to/weather_edge.db

The DB path defaults to $WEATHER_EDGE_DB, else the project's
persistence.DEFAULT_DB_PATH.
"""
import argparse
import os
import sqlite3
import sys


def default_db_path() -> str:
    env = os.environ.get("WEATHER_EDGE_DB")
    if env:
        return env
    from weather_edge.persistence import DEFAULT_DB_PATH
    return str(DEFAULT_DB_PATH)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Summarise live_trades")
    ap.add_argument("--db", default=None, help="Path to weather_edge.db")
    args = ap.parse_args(argv)
    if args.db is None:
        args.db = default_db_path()
    return args


args = parse_args()
if not os.path.exists(args.db):
    sys.exit(f"DB not found: {args.db}")
# Read-only: never create an empty DB by mistake
conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
conn.row_factory = sqlite3.Row

rows = conn.execute("SELECT * FROM live_trades ORDER BY placed_at DESC").fetchall()
trades = [dict(r) for r in rows]

total = len(trades)
open_t = [t for t in trades if t["status"] in ("open", "partial")]
filled = [t for t in trades if t["status"] == "filled"]
cancelled = [t for t in trades if t["status"] == "cancelled"]
won = [t for t in trades if t["status"] == "won"]
lost = [t for t in trades if t["status"] == "lost"]

print("=== LIVE TRADING SUMMARY ===")
print(f"Total orders: {total}")
print(f"Open: {len(open_t)}, Filled: {len(filled)}, Cancelled: {len(cancelled)}")
print(f"Won: {len(won)}, Lost: {len(lost)}")
print(f"Total P&L: ${sum(t['pnl'] or 0 for t in trades):.2f}")
print(f"Total fees: ${sum(t['fee_usd'] or 0 for t in trades):.2f}")
print(f"Capital at risk: ${sum(t['size_usd'] for t in open_t):.2f}")
print()

print("=== OPEN/PARTIAL ORDERS ===")
for t in open_t:
    fs = t["filled_shares"] or 0
    city = t["city_id"].upper()
    desc = (t["description"] or "")[:50]
    print(f"  {city:4} {t['side']:3} {t['status']:8} {fs:5.1f}/{t['size_shares']:5.1f} @ {t['limit_price']:.3f} ${t['size_usd']:6.2f} | {desc}")

print()
print("=== FILLED TRADES ===")
for t in filled:
    city = t["city_id"].upper()
    desc = (t["description"] or "")[:50]
    cb = t["cost_basis"] or 0
    fee = t["fee_usd"] or 0
    print(f"  {city:4} {t['side']:3} {t['filled_shares']:5.1f} @ {t['avg_fill_price']:.3f} cost=${cb:.2f} fee=${fee:.2f} | {desc}")

print()
print(f"=== CANCELLED: {len(cancelled)} orders ===")

print()
print("=== RESOLVED (WON/LOST) ===")
for t in won + lost:
    city = t["city_id"].upper()
    pnl = t["pnl"] or 0
    cb = t["cost_basis"] or 0
    pr = t["proceeds"] or 0
    print(f"  {city:4} {t['side']:3} {t['status']:4} P&L=${pnl:.2f} cost=${cb:.2f} proceeds=${pr:.2f}")

if not (won or lost):
    print("  (no resolved trades yet)")

conn.close()
