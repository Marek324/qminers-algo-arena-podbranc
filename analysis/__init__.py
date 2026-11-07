"""Market tape analysis: replay a tape and report its statistics as structured JSON/text."""

from .report import SCHEMA_VERSION, analyze

__all__ = ["SCHEMA_VERSION", "analyze"]
