#!/usr/bin/env python
"""
Script to run tests for the Movement Analysis system
"""
import os
import sys
import pytest
import argparse


def parse_args(args=None):
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run Movement Analysis tests')

    parser.add_argument('--test-dir', type=str, default='tests',
                        help='Directory containing tests')
    parser.add_argument('--cov', action='store_true',
                        help='Enable coverage reporting')
    parser.add_argument('--verbose', '-v', action='count', default=0,
                        help='Increase verbosity (can be repeated)')
    parser.add_argument('--test-file', type=str,
                        help='Run a specific test file')
    parser.add_argument('--show-print', action='store_true',
                        help='Show print statements from tests')

    return parser.parse_args(args)


def main():
    """Run tests"""
    args = parse_args()

    # Build pytest arguments
    pytest_args = []

    # Pass -s if show-print is enabled
    if args.show_print:
        pytest_args.append('-s')

    # Set verbosity
    if args.verbose > 0:
        pytest_args.extend(['-' + 'v' * args.verbose])

    # Add coverage if requested
    if args.cov:
        pytest_args.extend(['--cov=src', '--cov-report', 'term-missing'])

    # Set test directory or file
    if args.test_file:
        pytest_args.append(os.path.join(args.test_dir, args.test_file))
    else:
        pytest_args.append(args.test_dir)

    # Run tests
    print(f"Running tests with arguments: {' '.join(pytest_args)}")
    return pytest.main(pytest_args)


if __name__ == "__main__":
    sys.exit(main())