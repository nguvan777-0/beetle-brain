"""Universal seed system.

Seeds are strings. Any input — text, hex, binary, integers — hashes cleanly
to a uint64 via SHA-256. The string is what gets stored and displayed;
the integer is only used internally for numpy's RNG.
"""
import hashlib
import secrets


def to_int(name) -> int:
    """Hash any string or bytes to a uint64 numpy seed."""
    data = name if isinstance(name, (bytes, bytearray)) else str(name).encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:8], 'big')


def random_name() -> str:
    """Generate a short random alphanumeric seed (8 hex chars)."""
    return secrets.token_hex(4)   # e.g. "a3f92b1c"


def parse(x) -> str:
    """Coerce any seed input to a canonical string.

    None        → fresh random alphanumeric name
    bytes       → hex representation of those bytes
    anything    → str(x).strip()
    """
    if x is None:
        return random_name()
    if isinstance(x, (bytes, bytearray)):
        return x.hex()
    return str(x).strip()
