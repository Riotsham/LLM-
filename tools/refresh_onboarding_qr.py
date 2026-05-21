from __future__ import annotations

import json
import os
import shlex
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib import parse as urlparse
from urllib import request as urlrequest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception:
        return
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            os.environ[key] = value

try:
    import qrcode
except ModuleNotFoundError:
    qrcode = None

from api.server import ensure_server_started
from utils.rpi_ssh import resolve_rpi_host, resolve_ssh_key_path


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_onboarding_public_base() -> str:
    configured = (os.getenv("ONBOARDING_PUBLIC_BASE_URL") or "").strip().rstrip("/")

    port = int(os.getenv("ONBOARDING_SERVER_PORT") or 8765)
    rpi_host = (os.getenv("RPI_HOST") or "").strip()
    rpi_prefix = ".".join(rpi_host.split(".")[:3]) if rpi_host and "." in rpi_host else ""

    candidates: list[str] = []
    preferred_lan_ip = ""

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            preferred_lan_ip = s.getsockname()[0]
        if preferred_lan_ip and not preferred_lan_ip.startswith("127."):
            candidates.append(preferred_lan_ip)
    except Exception:
        pass

    host = socket.gethostname()
    try:
        for family, _, _, _, sockaddr in socket.getaddrinfo(host, None):
            if family != socket.AF_INET:
                continue
            ip = str(sockaddr[0]).strip()
            if not ip or ip.startswith("127."):
                continue
            candidates.append(ip)
    except Exception:
        pass

    # Deduplicate while preserving order.
    unique_candidates: list[str] = []
    seen: set[str] = set()
    for ip in candidates:
        if ip not in seen:
            seen.add(ip)
            unique_candidates.append(ip)

    if configured:
        parsed = urlparse.urlparse(configured if "://" in configured else f"http://{configured}")
        configured_host = (parsed.hostname or "").strip().lower()
        local_ips = {ip.lower() for ip in unique_candidates}
        rpi_host_lc = rpi_host.lower()

        def _is_ipv4_literal(host_value: str) -> bool:
            try:
                socket.inet_aton(host_value)
                return host_value.count(".") == 3
            except OSError:
                return False

        # Avoid generating phone QR URLs that point to loopback or a non-local Pi host.
        if configured_host not in {"localhost", "127.0.0.1"}:
            if configured_host in local_ips:
                return configured
            # Reject stale literal IPs that are no longer assigned to this laptop.
            if _is_ipv4_literal(configured_host):
                configured_host = ""
            else:
                try:
                    resolved_ips = {
                        str(sockaddr[0]).strip().lower()
                        for family, _, _, _, sockaddr in socket.getaddrinfo(configured_host, None)
                        if family == socket.AF_INET
                    }
                    if resolved_ips & local_ips:
                        return configured
                except Exception:
                    # Keep user-supplied DNS names when we cannot resolve here.
                    return configured
            if not (rpi_host_lc and configured_host == rpi_host_lc and rpi_host_lc not in local_ips):
                if configured_host:
                    return configured

    # Prefer IP in same /24 subnet as Raspberry Pi if available.
    if rpi_prefix:
        for ip in unique_candidates:
            if ip.startswith(f"{rpi_prefix}."):
                return f"http://{ip}:{port}"

    if unique_candidates:
        return f"http://{unique_candidates[0]}:{port}"

    return "http://127.0.0.1:8765"


def _json_request(url: str, method: str = "GET", payload: dict | None = None, timeout: float = 5.0) -> dict | None:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urlrequest.Request(url=url, data=data, method=method.upper(), headers=headers)
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
        return json.loads(raw) if raw else {}
    except Exception:
        return None


def _download_qr_image(url: str, target: Path) -> Path | None:
    if not url:
        return None
    target.parent.mkdir(parents=True, exist_ok=True)

    if qrcode is not None:
        try:
            qr = qrcode.QRCode(
                version=None,
                error_correction=qrcode.constants.ERROR_CORRECT_H,
                box_size=12,
                border=4,
            )
            qr.add_data(url)
            qr.make(fit=True)
            img = qr.make_image(fill_color="black", back_color="white")
            img.save(target)
            return target
        except Exception:
            pass

    try:
        encoded = urlparse.quote(url, safe="")
        qr_url = (
            "https://api.qrserver.com/v1/create-qr-code/"
            f"?size=420x420&ecc=H&qzone=4&format=png&color=000-000-000&bgcolor=255-255-255&data={encoded}"
        )
        with urlrequest.urlopen(qr_url, timeout=8) as resp:
            data = resp.read()
        target.write_bytes(data)
        return target
    except Exception:
        return None


def _push_to_rpi(local_state: Path, local_qr: Path | None) -> bool:
    port = os.getenv("RPI_PORT", "22").strip() or "22"
    host = resolve_rpi_host(port=port)
    if not host:
        print("RPI host not resolved. Check RPI_HOST or RPI_HOSTS.")
        return False

    user = os.getenv("RPI_USER", "pi").strip() or "pi"
    remote_dir = os.getenv("RPI_REMOTE_DIR", "/tmp/llm_tts").strip() or "/tmp/llm_tts"
    ssh_key_path = resolve_ssh_key_path(os.getenv("RPI_SSH_KEY", ""), cwd=Path(__file__).resolve().parents[1])
    target = f"{user}@{host}"
    ssh_cmd = ["ssh", "-p", port, "-o", "BatchMode=yes"]
    scp_cmd = ["scp", "-P", port, "-o", "BatchMode=yes"]
    if ssh_key_path and Path(ssh_key_path).exists():
        ssh_cmd.extend(["-i", ssh_key_path, "-o", "IdentitiesOnly=yes"])
        scp_cmd.extend(["-i", ssh_key_path, "-o", "IdentitiesOnly=yes"])

    try:
        subprocess.run([*ssh_cmd, target, f"mkdir -p {shlex.quote(remote_dir)}"], check=True, timeout=8)
        subprocess.run([*scp_cmd, str(local_state.resolve()), f"{target}:{remote_dir}/onboarding_state.json"], check=True, timeout=12)
        if local_qr and local_qr.exists():
            subprocess.run([*scp_cmd, str(local_qr.resolve()), f"{target}:{remote_dir}/onboarding_qr.png"], check=True, timeout=12)
        return True
    except Exception as exc:
        print(f"Failed to push to RPI: {exc}")
        return False


def main() -> None:
    _load_env_file(ROOT / ".env")
    host, port = ensure_server_started()
    local_base = f"http://127.0.0.1:{port}"
    base_url = _read_onboarding_public_base()

    ttl_minutes = 20
    session = _json_request(
        f"{local_base}/api/onboarding/session",
        method="POST",
        payload={"ttl_minutes": ttl_minutes},
        timeout=5.0,
    )
    if not session or not session.get("token"):
        print("Failed to create onboarding session. Is the server running?")
        return

    onboarding_url = f"{base_url}/onboarding/latest"
    payload = {
        "status": "waiting",
        "url": onboarding_url,
        "message": "Scan QR and submit profile",
        "updated_at": _now_iso(),
    }

    with tempfile.TemporaryDirectory() as td:
        temp_dir = Path(td)
        local_state = temp_dir / "onboarding_state.json"
        local_state.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
        local_qr = _download_qr_image(onboarding_url, temp_dir / "onboarding_qr.png")

        if not local_qr:
            print("Failed to generate QR image.")
        else:
            print(f"QR generated for {onboarding_url}")

        pushed = _push_to_rpi(local_state, local_qr)
        if pushed:
            print("Onboarding state pushed to Pi.")
        else:
            print("Onboarding state NOT pushed to Pi.")


if __name__ == "__main__":
    main()
