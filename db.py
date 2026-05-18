"""
db.py
-----
Shared database connection helpers for the RAG system.
Provides a psycopg DB_KWARGS dict and a context-manager get_conn()
for use in both run_pipeline.py and app.py.
"""

import os
import re
from urllib.parse import unquote
from dotenv import load_dotenv

load_dotenv()

_raw_url = os.getenv("DATABASE_URL")
if not _raw_url:
    raise RuntimeError("DATABASE_URL environment variable is not set.")

# urlparse misparses URLs when the password contains characters like %9N
# (invalid percent-encoding). We use a regex instead.
_URL_RE = re.compile(
    r"^(?:postgresql|postgres)://([^:]+):(.+)@([^:@/]+):(\d+)/(.+)$"
)
_m = _URL_RE.match(_raw_url)
if not _m:
    raise RuntimeError(f"Could not parse DATABASE_URL: {_raw_url!r}")

_user, _raw_pass, _host, _port, _dbname = _m.groups()

DB_KWARGS = {
    "host": _host,
    "port": int(_port),
    "dbname": _dbname,
    "user": _user,
    "password": unquote(_raw_pass),
    # Disable server-side prepared statements — required for Supabase's
    # PgBouncer pooler running in transaction mode (port 6543).
    "prepare_threshold": None,
    "sslmode": "require",
    "connect_timeout": 10,
}

