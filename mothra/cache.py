from joblib import Memory

# The main script will override this as necessary.
# By default, no caching is performed (backend=None).
memory = Memory(backend=None)
