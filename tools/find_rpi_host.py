from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.rpi_ssh import resolve_ssh_key_path


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


def _write_rpi_host_to_env(path: Path, host_ip: str) -> bool:
    try:
        if path.exists():
            lines = path.read_text(encoding="utf-8").splitlines()
        else:
            lines = []
    except Exception:
        return False

    updated = False
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _ = stripped.split("=", 1)
        if key.strip() == "RPI_HOST":
            lines[idx] = f'RPI_HOST="{host_ip}"'
            updated = True
            break

    if not updated:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append(f'RPI_HOST="{host_ip}"')

    try:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return True
    except Exception:
        return False


def _local_ipv4() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return str(s.getsockname()[0]).strip()
    except Exception:
        return ""


def _arp_ips() -> list[str]:
    try:
        proc = subprocess.run(
            ["arp", "-a"],
            capture_output=True,
            text=True,
            timeout=4,
            check=False,
        )
        text = proc.stdout or ""
    except Exception:
        return []
    ips = re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text)
    unique: list[str] = []
    seen: set[str] = set()
    for ip in ips:
        if ip.startswith("127."):
            continue
        if ip in seen:
            continue
        seen.add(ip)
        unique.append(ip)
    return unique


def _is_ssh_open(ip: str, port: int, timeout: float = 0.35) -> bool:
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except Exception:
        return False


def _can_auth_ssh(ip: str, port: int, user: str, key_path: str) -> bool:
    cmd = [
        "ssh",
        "-p",
        str(port),
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=2",
    ]
    if key_path and Path(key_path).exists():
        cmd.extend(["-i", key_path, "-o", "IdentitiesOnly=yes"])
    cmd.extend([f"{user}@{ip}", "echo ok"])
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False)
        return proc.returncode == 0 and "ok" in (proc.stdout or "").lower()
    except Exception:
        return False


def _refresh_qr_assets() -> bool:
    refresh_script = ROOT / "tools" / "refresh_onboarding_qr.py"
    if not refresh_script.exists():
        print("QR refresh script not found; skipping QR refresh.")
        return False
    try:
        proc = subprocess.run(
            [sys.executable, str(refresh_script)],
            cwd=str(ROOT),
            timeout=90,
            check=False,
        )
        return proc.returncode == 0
    except Exception:
        return False


def _deploy_and_restart_ui(host: str, user: str, port: int, key_path: str) -> bool:
    local_ui = ROOT / "rpi_robo_eye_ui.py"
    if not local_ui.exists():
        print("Local rpi_robo_eye_ui.py not found; skipping UI deploy.")
        return False

    remote_ui = (os.getenv("RPI_EYE_UI_PATH", "") or "").strip() or f"/home/{user}/rpi_robo_eye_ui.py"
    ssh_cmd = ["ssh", "-p", str(port), "-o", "BatchMode=yes", "-o", "ConnectTimeout=4"]
    scp_cmd = ["scp", "-P", str(port), "-o", "BatchMode=yes", "-o", "ConnectTimeout=4"]
    if key_path and Path(key_path).exists():
        ssh_cmd.extend(["-i", key_path, "-o", "IdentitiesOnly=yes"])
        scp_cmd.extend(["-i", key_path, "-o", "IdentitiesOnly=yes"])

    target = f"{user}@{host}"
    remote_dir = str(Path(remote_ui).parent).replace("\\", "/")
    ui_quoted = remote_ui.replace("'", "'\"'\"'")
    stop_cmd = "pkill -f '^python3 .*/rpi_robo_eye_ui.py$' >/dev/null 2>&1 || true"
    start_cmd = (
        f"mkdir -p '{remote_dir}'; "
        "export DISPLAY=:0; "
        f"export XAUTHORITY=/home/{user}/.Xauthority; "
        f"nohup python3 '{ui_quoted}' >/tmp/rpi_robo_eye_ui.log 2>&1 </dev/null &"
    )
    verify_cmd = "pgrep -af 'python3 .*/rpi_robo_eye_ui.py'"
    try:
        subprocess.run([*scp_cmd, str(local_ui.resolve()), f"{target}:{remote_ui}"], timeout=15, check=True)
        subprocess.run([*ssh_cmd, target, stop_cmd], timeout=8, check=False)
        subprocess.run([*ssh_cmd, target, start_cmd], timeout=10, check=False)
        verify = subprocess.run([*ssh_cmd, target, verify_cmd], timeout=8, check=False, capture_output=True, text=True)
        return verify.returncode == 0 and "rpi_robo_eye_ui.py" in (verify.stdout or "")
    except Exception:
        return False


def main() -> None:
    env_path = ROOT / ".env"
    _load_env_file(env_path)
    user = os.getenv("RPI_USER", "pi").strip() or "pi"
    port = int((os.getenv("RPI_PORT", "22").strip() or "22"))
    key_path = resolve_ssh_key_path(os.getenv("RPI_SSH_KEY", ""), cwd=ROOT)

    local_ip = _local_ipv4()
    prefix = ".".join(local_ip.split(".")[:3]) if local_ip and "." in local_ip else ""
    arp_ips = _arp_ips()
    candidates = [ip for ip in arp_ips if (not prefix or ip.startswith(f"{prefix}."))]

    if not candidates:
        print("No ARP candidates found. Connect Pi and laptop to same network, then retry.")
        return

    print(f"Scanning {len(candidates)} candidate IPs for SSH auth as {user}...")
    for ip in candidates:
        if not _is_ssh_open(ip, port):
            continue
        if _can_auth_ssh(ip, port, user=user, key_path=key_path):
            print(f"FOUND_RPI_HOST={ip}")
            saved = _write_rpi_host_to_env(env_path, ip)
            if saved:
                print(f'Updated .env with RPI_HOST="{ip}"')
            else:
                print(f'Could not update .env automatically. Set manually: RPI_HOST="{ip}"')
            print("Deploying latest Pi UI...")
            if _deploy_and_restart_ui(ip, user=user, port=port, key_path=key_path):
                print("Pi UI deployed and restarted.")
            else:
                print("Pi UI deploy failed; continuing.")
            print("Refreshing onboarding QR/state on Pi...")
            if _refresh_qr_assets():
                print("QR refresh completed.")
            else:
                print("QR refresh failed. Run: python tools/refresh_onboarding_qr.py")
            return

    print("Could not auto-discover Raspberry Pi host via SSH.")
    print("Try: on Pi run `hostname -I`, then set that IP as RPI_HOST in .env")


if __name__ == "__main__":
    main()
