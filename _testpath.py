"""_testpath.py -- one call that makes the source tree importable, on any box.

WHY THIS EXISTS
---------------
The package is not pip-installed on a development box, so tests must import it
straight out of the checkout. Doing that ad hoc produced a family of import
failures, each patched separately before the pattern was obvious:

  * a script with no path setup at all          -> ModuleNotFoundError
  * a script inserting its OWN directory        -> ModuleNotFoundError, because
    the importable name `bitsandbytes` lives INSIDE that directory, so what
    sys.path needs is its PARENT
  * a script inserting the checkout root        -> works only where the checkout
    root happens to be the package parent
  * a script hard-coding one machine's path     -> works on exactly one machine

The layout genuinely differs between the two development boxes, which is why a
constant cannot work:

    R5   checkout D:\\work\\bitsandbytes-CPU\\
           bitsandbytes\\            <- the package
           bitsandbytes-CPU\\bitsandbytes\\  <- also importable, via a .pth file

    i5   checkout C:\\Users\\GanYv\\bnb_repo\\
           bitsandbytes\\            <- the package, and the ONLY copy
         (there is no bnb_repo\\bitsandbytes\\bitsandbytes)

So this module SEARCHES for a directory that contains an importable
`bitsandbytes` package, checking the script's directory, its ancestors, and its
immediate children -- and then puts both that directory and the checkout root on
sys.path.

USAGE -- the first statement in a test script, before importing bnb:

    import _testpath  # noqa: F401

`bootstrap()` is also provided for scripts that cannot rely on this file being
importable yet (it locates this file by walking up from the caller). It is safe
to call repeatedly and from any working directory.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _is_pkg_parent(d):
    """True if d contains an importable `bitsandbytes` package."""
    pkg = os.path.join(d, "bitsandbytes")
    return os.path.isfile(os.path.join(pkg, "__init__.py"))


def find_pkg_parent(start=None):
    """Return the directory that must be on sys.path for `import bitsandbytes`.

    Checks `start` (default: this file's directory), then each ancestor, then
    each immediate subdirectory. Falls back to `start` so callers always get a
    usable path.
    """
    d = os.path.abspath(start or _HERE)
    if _is_pkg_parent(d):
        return d
    up = d
    while True:
        parent = os.path.dirname(up)
        if parent == up:
            break
        up = parent
        if _is_pkg_parent(up):
            return up
    try:
        for name in sorted(os.listdir(d)):
            cand = os.path.join(d, name)
            if os.path.isdir(cand) and _is_pkg_parent(cand):
                return cand
    except OSError:
        pass
    return d


def ensure_on_path():
    """Put the package parent and the checkout root on sys.path.

    Returns (pkg_parent, checkout_root).
    """
    pkg_parent = find_pkg_parent()
    for d in (pkg_parent, _HERE):
        if d not in sys.path:
            sys.path.insert(0, d)
    return pkg_parent, _HERE


def bootstrap(caller_file=None):
    """Locate this module's directory by walking up from `caller_file`, put it on
    sys.path, and run ensure_on_path().

    For scripts that must not assume `import _testpath` already resolves. Always
    installs the path that make_bitsandbytes_importable() needs, so a silent
    fallback still leaves the caller in the best state available.
    """
    if caller_file is None:
        import inspect
        caller_file = inspect.stack()[1].filename
    d = os.path.dirname(os.path.abspath(caller_file))
    while not os.path.isfile(os.path.join(d, "_testpath.py")):
        up = os.path.dirname(d)
        if up == d:
            break
        d = up
    if os.path.isfile(os.path.join(d, "_testpath.py")) and d not in sys.path:
        sys.path.insert(0, d)
    return ensure_on_path()


PKG_PARENT, CHECKOUT_ROOT = ensure_on_path()
