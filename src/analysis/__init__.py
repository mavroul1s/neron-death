"""Post-hoc analysis. **Never imported by training code** (CLAUDE.md §4).

`tests/test_layout.py` enforces that one-way dependency, so an accidental
`from .analysis import ...` in train.py fails the suite rather than quietly
coupling the two.
"""
