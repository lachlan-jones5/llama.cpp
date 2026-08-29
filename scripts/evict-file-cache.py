#!/usr/bin/env python3
"""Evict one file from the page cache, and report how much of it is still resident.

Benchmarks that read from disk are meaningless if the file is already cached. This drops the cache for a
single named file with posix_fadvise(POSIX_FADV_DONTNEED) rather than dropping the whole system's cache,
so it needs no privileges and does not disturb anything else running on the machine.

Usage: evict-file-cache.py <path> [<path> ...]
"""

import ctypes
import ctypes.util
import os
import sys


def resident_pages(fd, size):
    """Fraction of the file currently in the page cache, via mincore(2). None if it cannot be determined."""
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)

    if not hasattr(libc, "mincore") or size == 0:
        return None

    page = os.sysconf("SC_PAGESIZE")
    n_pages = (size + page - 1) // page

    # map read-only; mincore reports per-page residency without faulting anything in
    mm = ctypes.CDLL(None)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                          ctypes.c_int, ctypes.c_int, ctypes.c_long]

    PROT_READ = 1
    MAP_SHARED = 1

    addr = libc.mmap(None, size, PROT_READ, MAP_SHARED, fd, 0)
    if addr == ctypes.c_void_p(-1).value or addr is None:
        return None

    try:
        vec = (ctypes.c_ubyte * n_pages)()
        if libc.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(size), vec) != 0:
            return None
        return sum(v & 1 for v in vec) / n_pages
    finally:
        libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        libc.munmap(ctypes.c_void_p(addr), size)
        del mm


def evict(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size

        before = resident_pages(fd, size)

        # ask the kernel to drop this file's clean pages; dirty pages need a flush first
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)

        after = resident_pages(fd, size)

        def pct(x):
            return "unknown" if x is None else f"{100.0 * x:.1f}%"

        print(f"{path}\n  size {size / (1 << 30):.2f} GiB, cached before {pct(before)}, after {pct(after)}")

        if after is not None and after > 0.01:
            print("  WARNING: still largely cached - another process may hold it mapped")
    finally:
        os.close(fd)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    for p in sys.argv[1:]:
        evict(p)
