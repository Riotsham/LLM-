import os
import socket
from pathlib import Path


def resolve_env_path(path_value: str) -> str:
    value = (path_value or "").strip()
    if not value:
        return ""
    home = str(Path.home())
    value = value.replace("${HOME}", home).replace("$HOME", home)
    value = os.path.expandvars(value)
    value = os.path.expanduser(value)
    return value


def resolve_ssh_key_path(path_value: str, cwd: Path | None = None) -> str:
    candidates: list[Path] = []
    resolved = resolve_env_path(path_value)
    if resolved:
        candidates.append(Path(resolved))

    home = Path.home()
    candidates.extend([home / ".ssh" / "id_ed25519", home / ".ssh" / "id_rsa"])

    base = cwd if cwd else Path.cwd()
    candidates.append(base / "sshkey")

    seen: set[str] = set()
    for candidate in candidates:
        normalized = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        if candidate.exists() and candidate.is_file():
            return str(candidate)
    return ""


def resolve_rpi_host(port: str, default_host: str = "raspberrypi.local") -> str:
    primary = (os.getenv("RPI_HOST", "") or "").strip()
    host_list_raw = (os.getenv("RPI_HOSTS", "") or "").strip()

    candidates: list[str] = []
    if primary:
        candidates.append(primary)
    if host_list_raw:
        candidates.extend([h.strip() for h in host_list_raw.split(",") if h.strip()])
    if default_host and default_host not in candidates:
        candidates.append(default_host)

    # Deduplicate while preserving order.
    unique_candidates: list[str] = []
    seen: set[str] = set()
    for host in candidates:
        if host not in seen:
            seen.add(host)
            unique_candidates.append(host)

    if not unique_candidates:
        return ""
    if len(unique_candidates) == 1:
        return unique_candidates[0]

    port_int = int(port or 22)
    for host in unique_candidates:
        try:
            with socket.create_connection((host, port_int), timeout=0.8):
                return host
        except Exception:
            continue

    # Return first configured candidate even if currently unreachable.
    return unique_candidates[0]
