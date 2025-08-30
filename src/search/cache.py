"""Cache utilities for metadata lookups."""

from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar
from cachetools import TTLCache, keys

T = TypeVar("T")

# Cache for 5 minutes by default
DEFAULT_TTL = 300

# Shared cache instances
meta_cache = TTLCache(maxsize=1024, ttl=DEFAULT_TTL)

def cached(cache: TTLCache, key_func: Optional[Callable] = None):
    """Decorator to cache function results in a TTLCache.
    
    Args:
        cache: The TTLCache instance to use
        key_func: Optional function to generate cache keys. If not provided,
                 uses cachetools.keys.hashkey
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Skip self argument for methods
            cache_args = args[1:] if args else ()
            k = key_func(*cache_args, **kwargs) if key_func else keys.hashkey(*cache_args, **kwargs)
            try:
                return cache[k]
            except KeyError:
                v = func(*args, **kwargs)
                try:
                    cache[k] = v
                except ValueError:
                    pass  # value too large, skip caching
                return v
        return wrapper
    return decorator

def clear_caches():
    """Clear all caches. Call this after reloading metadata."""
    meta_cache.clear()
