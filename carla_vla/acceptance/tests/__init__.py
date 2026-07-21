"""Unit tests for the D0 acceptance protocol formulas.

Run from repo root:
  python -m unittest carla_vla.acceptance.tests.test_protocol -v
  python -m unittest carla_vla.acceptance.tests.test_schemas -v
  python -m unittest carla_vla.acceptance.tests.test_aggregation -v

A combined runner:
  python -m carla_vla.acceptance.tests
"""
from . import test_protocol, test_schemas, test_aggregation  # noqa: F401