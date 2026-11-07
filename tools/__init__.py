"""Building blocks to COPY into a strategy file. Never import them from a strategy.

Only one .py file is submitted, so a strategy cannot import from `tools/`. Each
tool here is self-contained: paste the class (and the imports listed at the top
of its file) into the strategy. See AGENTS.md, "Tools".
"""
