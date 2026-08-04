"""Materialize `data/agent/inventory.db` from the committed CSV seeds (Issue #100,
`docs/agent_design.md` Section 7).

Mirrors `docs/serving_design.md` Section 5's posture: the database is runtime state that does
not outlive the demo. It is gitignored, rebuilt from the committed, human-diffable CSV seeds
in `src/agent/inventory/seed/`, and materialization is idempotent -- calling `build_db()` at
every process startup is safe. By default an existing database is left untouched (so orders
placed during a running demo survive a restart of the same container/volume); pass
`reset=True` (or run this module with `--reset`) to drop and rebuild from the seeds, which is
what `docker compose down -v` or a fresh clone effectively does anyway by removing the file.

`sqlite3` and `csv` are both standard library -- zero new dependencies, per Issue #100.

Reproducing:

    python -m src.agent.inventory.build_db [--reset] [--db-path PATH]
"""
from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SEED_DIR = Path(__file__).resolve().parent / "seed"
PARTS_SEED_PATH = SEED_DIR / "parts.csv"
ORDERS_SEED_PATH = SEED_DIR / "orders.csv"
DB_PATH = REPO_ROOT / "data" / "agent" / "inventory.db"

# Exactly the schema decided in docs/agent_design.md Section 7 -- not re-litigated here.
SCHEMA_SQL = """
CREATE TABLE parts (
  part_number       TEXT PRIMARY KEY,
  description       TEXT NOT NULL,
  bearing_type      TEXT,
  quantity_on_hand  INTEGER NOT NULL CHECK (quantity_on_hand >= 0),
  unit_price_usd    REAL    NOT NULL CHECK (unit_price_usd >= 0),
  lead_time_days    INTEGER NOT NULL,
  location          TEXT    NOT NULL
);

CREATE TABLE orders (
  order_id     INTEGER PRIMARY KEY AUTOINCREMENT,
  part_number  TEXT    NOT NULL REFERENCES parts(part_number),
  quantity     INTEGER NOT NULL CHECK (quantity > 0),
  bearing_id   TEXT,
  requested_by TEXT    NOT NULL,
  approved_by  TEXT    NOT NULL,
  approved_at  TEXT    NOT NULL,
  created_at   TEXT    NOT NULL,
  status       TEXT    NOT NULL DEFAULT 'placed'
);
"""

PARTS_COLUMNS = [
    "part_number",
    "description",
    "bearing_type",
    "quantity_on_hand",
    "unit_price_usd",
    "lead_time_days",
    "location",
]

ORDERS_COLUMNS = [
    "order_id",
    "part_number",
    "quantity",
    "bearing_id",
    "requested_by",
    "approved_by",
    "approved_at",
    "created_at",
    "status",
]


def _read_seed_rows(path: Path) -> list[dict[str, str]]:
    """Read a seed CSV, skipping the `#`-prefixed header-comment lines (see the seed files
    themselves for what they document -- the real-vs-invented data disclosure for `parts.csv`
    in particular)."""
    with path.open(newline="", encoding="utf-8") as f:
        data_lines = [line for line in f if not line.startswith("#")]
    return list(csv.DictReader(data_lines))


def _load_parts(conn: sqlite3.Connection) -> int:
    rows = _read_seed_rows(PARTS_SEED_PATH)
    if not rows:
        raise ValueError(f"{PARTS_SEED_PATH} contains no part rows")
    for row in rows:
        missing = [c for c in PARTS_COLUMNS if c not in row]
        if missing:
            raise ValueError(f"{PARTS_SEED_PATH} row {row!r} missing columns {missing}")
    conn.executemany(
        f"INSERT INTO parts ({', '.join(PARTS_COLUMNS)}) "
        f"VALUES ({', '.join('?' for _ in PARTS_COLUMNS)})",
        [
            (
                row["part_number"],
                row["description"],
                row["bearing_type"] or None,
                int(row["quantity_on_hand"]),
                float(row["unit_price_usd"]),
                int(row["lead_time_days"]),
                row["location"],
            )
            for row in rows
        ],
    )
    return len(rows)


def _validate_orders_seed_is_header_only() -> None:
    """`orders.csv` is committed header-only (Section 7): the demo's order history starts
    empty, and every row that ever exists in the `orders` table was written by
    `place_order`'s transactional path during a run. This only checks the header shape
    matches the schema; it deliberately reads no data rows into the database."""
    with ORDERS_SEED_PATH.open(newline="", encoding="utf-8") as f:
        data_lines = [line for line in f if not line.startswith("#")]
    reader = csv.reader(data_lines)
    header = next(reader)
    if header != ORDERS_COLUMNS:
        raise ValueError(
            f"{ORDERS_SEED_PATH} header {header!r} does not match expected columns "
            f"{ORDERS_COLUMNS!r}"
        )
    remaining = list(reader)
    if remaining:
        raise ValueError(
            f"{ORDERS_SEED_PATH} must be header-only; found {len(remaining)} data row(s)"
        )


def build_db(db_path: Path = DB_PATH, *, reset: bool = False) -> int:
    """Materialize the SQLite database at `db_path` from the committed CSV seeds.

    Idempotent: if `db_path` already exists and `reset` is False, this is a no-op and the
    existing database (including any orders placed at runtime) is left untouched. With
    `reset=True`, the existing file is removed and rebuilt from the seeds.

    Returns the number of part rows seeded (0 if the database already existed and was left
    untouched).
    """
    _validate_orders_seed_is_header_only()

    db_path = Path(db_path)
    if db_path.exists():
        if not reset:
            return 0
        db_path.unlink()

    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(SCHEMA_SQL)
        n_parts = _load_parts(conn)
        conn.commit()
    except Exception:
        conn.close()
        db_path.unlink(missing_ok=True)
        raise
    else:
        conn.close()
        return n_parts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and rebuild the database from the seeds, discarding any runtime orders.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DB_PATH,
        help=f"Where to materialize the database (default: {DB_PATH}).",
    )
    args = parser.parse_args()

    n_parts = build_db(args.db_path, reset=args.reset)
    if n_parts:
        print(f"Built {args.db_path} with {n_parts} seeded part(s).")
    else:
        print(f"{args.db_path} already exists; left untouched (pass --reset to rebuild).")


if __name__ == "__main__":
    main()
