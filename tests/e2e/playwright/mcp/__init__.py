"""
Playwright MCP E2E Testing Module

This module contains helpers and test configurations for running
visual E2E tests via Playwright MCP tools. Tests execute in real-time
allowing visual monitoring of the descent/ascent cycle.

Usage:
    Tests are executed via Playwright MCP tools in conversation,
    not via pytest. See quickstart.md for execution instructions.
"""

from .helpers import (
    TestDataset,
    SourceFile,
    JoinConfig,
    DescentConfig,
    AscentConfig,
    ExpectedOutput,
    CategorizeConfig,
    ExtractConfig,
    AggregateConfig,
    DimensionConfig,
    SCHOOLS_DATASET,
    ADEME_DATASET,
)

__all__ = [
    "TestDataset",
    "SourceFile",
    "JoinConfig",
    "DescentConfig",
    "AscentConfig",
    "ExpectedOutput",
    "CategorizeConfig",
    "ExtractConfig",
    "AggregateConfig",
    "DimensionConfig",
    "SCHOOLS_DATASET",
    "ADEME_DATASET",
]
