"""Package and on-disk schema versions.

SCHEMA_VERSION is the only supported trace format in v1. See
docs/adr/0001-v1-trace-schema.md.
"""

__version__ = "0.1.0"
SCHEMA_VERSION = "1.0.0"

# Hard cap so the simple JSON/JSONL format cannot become a full-vocab dump.
MAX_TOP_K = 50
DEFAULT_TOP_K = 5
