#!/usr/bin/env python3


def fibonacci(n):
    """Calculate the nth Fibonacci number."""
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python fibonacci.py <n>")
        sys.exit(1)

    try:
        n = int(sys.argv[1])
        result = fibonacci(n)
        print(result)
    except ValueError:
        print("Error: Input must be an integer")
        sys.exit(1)
