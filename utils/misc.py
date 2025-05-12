import functools

import polars as pl

from utils.definitions import CACHE_DIR


def cache_data(cache_path=None, use_cache=True, file_format="parquet"):
    """
    Decorator for caching function results to disk.

    Args:
        cache_path: Path to save cache file (relative to CACHE_DIR)
        use_cache: Whether to use caching (default: True)
        file_format: Format to save data ('parquet' or 'csv')
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get cache file path
            path = cache_path or f"{func.__name__}.{file_format}"
            full_path = CACHE_DIR / path

            # Check for cache dir
            if not CACHE_DIR.exists():
                CACHE_DIR.mkdir(parents=True, exist_ok=True)

            # Use cache if it exists and use_cache is True
            if use_cache and full_path.exists():
                if file_format == "parquet":
                    return pl.read_parquet(full_path)
                elif file_format == "csv":
                    return pl.read_csv(full_path)
                else:
                    raise ValueError(f"Unsupported file format: {file_format}")

            # Call the original function
            result = func(*args, **kwargs)

            # Cache the result
            if use_cache:
                if file_format == "parquet":
                    result.write_parquet(full_path)
                elif file_format == "csv":
                    result.write_csv(full_path)
                else:
                    raise ValueError(f"Unsupported file format: {file_format}")

            return result

        return wrapper

    return decorator
