"""Explicit and additive migration. Does not modify existing meeting records."""
import argparse
import os
from pathlib import Path


def migrate(url: str, apply: bool = False):
    path = Path(__file__).resolve().parent.parent / 'storage' / 'migrations' / '002_intelligence.sql'
    source = path.read_text(encoding='utf-8')
    if not apply:
        print('DRY RUN: additive Phase 05 schema only, existing tables unchanged')
        print('SQL file:', path)
        return
    if not url:
        raise ValueError('DATABASE_URL required to apply migration')
    import psycopg
    with psycopg.connect(url) as conn:
        with conn.transaction():
            for stmt in source.split(';'):
                stmt = stmt.strip()
                if stmt:
                    conn.execute(stmt)
    print('PHASE 05 DATABASE MIGRATION APPLIED')


if __name__ == '__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument('--apply', action='store_true')
    args = arg.parse_args()
    migrate(os.getenv('DATABASE_URL', ''), apply=args.apply)
