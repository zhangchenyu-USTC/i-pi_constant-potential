"""Thread-local XML checkpoint serialization, independent of output formats."""

from contextlib import contextmanager
from contextvars import ContextVar

checkpoint_mode = ContextVar("ipi_checkpoint_mode", default=False)
float_format = ContextVar("ipi_serialization_float_format", default=None)


@contextmanager
def checkpoint_serialization():
    """Round-trip IEEE doubles and emit explicitly required restart fields."""

    mode_token = checkpoint_mode.set(True)
    format_token = float_format.set("%.16e")
    try:
        yield
    finally:
        float_format.reset(format_token)
        checkpoint_mode.reset(mode_token)


@contextmanager
def default_serialization(precise=False):
    """Capture defaults at the same precision used when comparing their values."""

    mode_token = checkpoint_mode.set(False)
    format_token = float_format.set("%.16e" if precise else None)
    try:
        yield
    finally:
        float_format.reset(format_token)
        checkpoint_mode.reset(mode_token)
