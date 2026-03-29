#!/usr/bin/env python3
# =============================================================================
# run_tests.py — Test Runner with formatted report
# =============================================================================
"""
Discovers and runs all test suites, then prints a structured summary.

Usage:
    python3 run_tests.py              # all tests
    python3 run_tests.py --unit       # unit tests only
    python3 run_tests.py --integration # integration test only
"""

import sys
import os
import argparse
import unittest
import time

# ── Ensure pipeline modules are importable ────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python-test'))
TEST_DIR = os.path.dirname(__file__)


def run_suite(label: str, pattern: str) -> unittest.TestResult:
    loader = unittest.TestLoader()
    suite  = loader.discover(TEST_DIR, pattern=pattern)

    print(f"\n{'='*62}")
    print(f"  {label}")
    print(f"{'='*62}")

    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        tb_locals=False,
    )
    start  = time.time()
    result = runner.run(suite)
    elapsed = time.time() - start
    print(f"\n  Ran {result.testsRun} test(s) in {elapsed:.2f}s")
    return result


def print_summary(results: list[tuple[str, unittest.TestResult]]):
    print(f"\n{'='*62}")
    print("  FINAL SUMMARY")
    print(f"{'='*62}")

    total_run     = sum(r.testsRun      for _, r in results)
    total_ok      = sum(r.testsRun - len(r.failures) - len(r.errors) - len(r.skipped)
                        for _, r in results)
    total_fail    = sum(len(r.failures) for _, r in results)
    total_error   = sum(len(r.errors)   for _, r in results)
    total_skipped = sum(len(r.skipped)  for _, r in results)

    for label, r in results:
        ok   = r.testsRun - len(r.failures) - len(r.errors) - len(r.skipped)
        icon = "✓" if not r.failures and not r.errors else "✗"
        print(f"  {icon}  {label:<30}  "
              f"passed={ok}  failed={len(r.failures)}  "
              f"errors={len(r.errors)}  skipped={len(r.skipped)}")

    print(f"\n  Total tests : {total_run}")
    print(f"  Passed      : {total_ok}")
    print(f"  Failed      : {total_fail}")
    print(f"  Errors      : {total_error}")
    print(f"  Skipped     : {total_skipped}")

    if total_fail == 0 and total_error == 0:
        print("\n  ✓  ALL TESTS PASSED")
    else:
        print("\n  ✗  SOME TESTS FAILED — see output above for details")
    print(f"{'='*62}\n")

    return total_fail + total_error


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit",        action="store_true")
    parser.add_argument("--integration", action="store_true")
    args = parser.parse_args()

    results = []

    if args.integration:
        results.append(("Integration Tests",
                         run_suite("Integration Tests — Flask API ↔ PostgreSQL",
                                   "test_integration.py")))
    elif args.unit:
        results.append(("Unit: Cleaning",
                         run_suite("Unit Tests — cleaning.py",  "test_cleaning.py")))
        results.append(("Unit: Features",
                         run_suite("Unit Tests — features.py",  "test_features.py")))
        results.append(("Unit: Transforms",
                         run_suite("Unit Tests — transforms.py", "test_transforms.py")))
    else:
        results.append(("Unit: Cleaning",
                         run_suite("Unit Tests — cleaning.py",  "test_cleaning.py")))
        results.append(("Unit: Features",
                         run_suite("Unit Tests — features.py",  "test_features.py")))
        results.append(("Unit: Transforms",
                         run_suite("Unit Tests — transforms.py", "test_transforms.py")))
        results.append(("Integration",
                         run_suite("Integration Tests — Flask API ↔ PostgreSQL",
                                   "test_integration.py")))

    exit_code = print_summary(results)
    sys.exit(exit_code)
