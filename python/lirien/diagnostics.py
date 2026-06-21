import os
import threading
from contextlib import contextmanager
from typing import Dict, Tuple

from . import lirien_bridge

_local_state = threading.local()
_tracing_stack = []

# Component names for tracing
LIVENESS = "liveness"
VERIFY = "verify"
Z3 = "verify::z3"
SSA = "ssa"
BACKEND = "backend"
BRIDGE = "bridge"
ALL = "all"


@contextmanager
def no_verification():
    """
    Context manager to temporarily disable Z3 verification for any code compiled
    within this block.
    """
    old_state = getattr(_local_state, "no_verification", False)
    _local_state.no_verification = True
    try:
        yield
    finally:
        _local_state.no_verification = old_state


def _is_verification_disabled() -> bool:
    return getattr(_local_state, "no_verification", False)


@contextmanager
def tracing(config: Dict[str, str]):
    """
    Context manager to temporarily configure granular tracing for specific Lirien components.
    """
    _tracing_stack.append(config)
    merged = {"all": "info"}
    for c in _tracing_stack:
        merged.update(c)
    configure_tracing(merged)
    try:
        yield
    finally:
        _tracing_stack.pop()
        merged = {"all": "info"}
        for c in _tracing_stack:
            merged.update(c)
        configure_tracing(merged)


def configure_tracing(config: Dict[str, str]):
    """
    Configure granular tracing for Lirien components.
    """
    lirien_bridge.configure_tracing(config)


def get_cpu_info() -> Dict[str, str]:
    """
    Get information about the host CPU architecture and enabled SIMD features.
    """
    return lirien_bridge.get_cpu_info()


class VerificationError(Exception):
    """Raised when Lirien formal verification or JIT compilation fails in strict mode."""

    pass


def format_verification_error(func_name: str, source: str, error: str) -> str:
    import re

    # Try to find offset in the error message
    match = re.search(r"at offset (\d+)", error)
    if match:
        offset = int(match.group(1))
        # Remove the offset info from the error message for cleaner display
        clean_error = error.replace(match.group(0), "").strip()

        # Find line and column from offset
        lines = source.splitlines()
        curr_offset = 0
        target_line_idx = 0
        target_col = 0
        for i, line in enumerate(lines):
            line_len = len(line) + 1  # +1 for newline
            if curr_offset <= offset < curr_offset + line_len:
                target_line_idx = i
                target_col = offset - curr_offset
                break
            curr_offset += line_len

        # Format pretty error
        res = [f"Lirien Verification Failed for '{func_name}': {clean_error}"]
        res.append(f"  at line {target_line_idx + 1}, col {target_col + 1}:")
        res.append("")

        # Context lines
        start_idx = max(0, target_line_idx - 1)
        end_idx = min(len(lines), target_line_idx + 2)
        for i in range(start_idx, end_idx):
            prefix = "> " if i == target_line_idx else "  "
            res.append(f"{prefix}{i + 1:4} | {lines[i]}")
            if i == target_line_idx:
                res.append("       | " + " " * target_col + "^")

        return "\n".join(res)

    return f"Lirien Verification Failed for '{func_name}': {error}"


def _setup_logging(log_level: str) -> Tuple[str, str]:
    """Override LILA_LOG level and return (log_level, old_log_level) for restoration."""
    if log_level:
        old_log = os.environ.get("LILA_LOG", "info")
        lirien_bridge.set_log_level(log_level)
        os.environ["LILA_LOG"] = log_level
        return log_level, old_log
    return None, None


def _restore_logging(log_level: str, old_log: str):
    """Restore the original LILA_LOG level."""
    if log_level:
        lirien_bridge.set_log_level(old_log)
        os.environ["LILA_LOG"] = old_log
