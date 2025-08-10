# mypy: allow-untyped-defs
r"""This package adds support for ROCm/ROCTX used in profiling."""

from contextlib import contextmanager


try:
    from torch._C import _roctx
except ImportError:

    class _ROCTXStub:
        @staticmethod
        def _fail(*args, **kwargs):
            raise RuntimeError(
                "ROCTX functions not installed. Are you sure you have a ROCm build?"
            )

        rangePushA = _fail
        rangePop = _fail
        markA = _fail

    _roctx = _ROCTXStub()  # type: ignore[assignment]

__all__ = ["range_push", "range_pop", "range_start", "range_end", "mark", "range"]


def range_push(msg):
    """
    Push a range onto a stack of nested range span.  Returns zero-based depth of the range that is started.

    Args:
        msg (str): ASCII message to associate with range
    """
    return _roctx.rangePushA(msg)


def range_pop():
    """Pop a range off of a stack of nested range spans.  Returns the  zero-based depth of the range that is ended."""
    return _roctx.rangePop()


def range_start(msg) -> int:
    """
    Mark the start of a range with string message. It returns an unique handle
    for this range to pass to the corresponding call to rangeEnd().

    A key difference between this and range_push/range_pop is that the
    range_start/range_end version supports range across threads (start on one
    thread and end on another thread).

    Args:
        msg (str): ASCII message to associate with range

    Returns:
        int: range handle to pass to range_end()

    Warning:
        range_start/range_end functions are not supported on Windows.
    """
    return _roctx.rangeStartA(msg)


def range_end(range_handle):
    """
    Mark the end of a range for a given range_handle. The range_handle is
    the value returned by the corresponding range_start call.

    Args:
        range_handle (int): handle returned from range_start()

    Warning:
        range_start/range_end functions are not supported on Windows.
    """
    return _roctx.rangeEnd(range_handle)


def mark(msg):
    """
    Describe an instantaneous event that occurred at some point.

    Args:
        msg (str): ASCII message to associate with the event.
    """
    return _roctx.markA(msg)


@contextmanager
def range(msg, *args, **kwargs):
    """
    Context manager / decorator that pushes an ROCTX range at the beginning
    of its scope, and pops it at the end.

    If extra arguments are given, they are passed as arguments to msg.format().

    Args:
        msg (str): message to associate with the range
    """
    range_push(msg.format(*args, **kwargs))
    try:
        yield
    finally:
        range_pop()
