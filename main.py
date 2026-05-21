import os
import shlex
import socket
import subprocess
import sys
import webbrowser
from pathlib import Path

from api.server import ensure_server_started
from utils.rpi_ssh import resolve_rpi_host, resolve_ssh_key_path


def _load_env_file(path: str = ".env") -> None:
    """Load KEY=VALUE pairs from a .env file into process environment."""
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ[key] = value


def _acquire_single_instance_lock() -> socket.socket | None:
    """Prevent multiple concurrent main.py processes on the same machine."""
    lock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
        lock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    try:
        lock.bind(("127.0.0.1", 45765))
        lock.listen(1)
        return lock
    except OSError:
        lock.close()
        return None


def _maybe_start_rpi_eye_ui() -> None:
    port = os.getenv("RPI_PORT", "22").strip() or "22"
    host = resolve_rpi_host(port=port)
    if not host:
        return

    user = os.getenv("RPI_USER", "pi").strip() or "pi"
    ssh_key_path = resolve_ssh_key_path(os.getenv("RPI_SSH_KEY", ""), cwd=Path(__file__).resolve().parent)
    ui_path = (os.getenv("RPI_EYE_UI_PATH", "") or "").strip() or f"/home/{user}/rpi_robo_eye_ui.py"

    ssh_cmd = ["ssh", "-p", port, "-o", "BatchMode=yes"]
    if ssh_key_path:
        ssh_cmd.extend(["-i", ssh_key_path, "-o", "IdentitiesOnly=yes"])

    target = f"{user}@{host}"
    stop_cmd = "pkill -f 'python3 .*/rpi_robo_eye_ui.py' >/dev/null 2>&1 || true"
    verify_cmd = "pgrep -af 'python3 .*/rpi_robo_eye_ui.py'"
    remote_dir = ui_path.rsplit("/", 1)[0] if "/" in ui_path else f"/home/{user}"
    start_script = (
        f"mkdir -p {shlex.quote(remote_dir)}; "
        "sleep 0.3; "
        "export DISPLAY=:0; "
        f"export XAUTHORITY={shlex.quote(f'/home/{user}/.Xauthority')}; "
        f"nohup /usr/bin/python3 {shlex.quote(ui_path)} >/tmp/rpi_robo_eye_ui.log 2>&1 </dev/null &"
    )
    start_cmd = f"bash -lc {shlex.quote(start_script)}"
    try:
        subprocess.run([*ssh_cmd, target, stop_cmd], check=False, timeout=8)
        subprocess.run([*ssh_cmd, target, start_cmd], check=False, timeout=12)
        verify = subprocess.run([*ssh_cmd, target, verify_cmd], check=False, timeout=8, capture_output=True, text=True)
        if verify.returncode != 0 or "rpi_robo_eye_ui.py" not in (verify.stdout or ""):
            print(
                "Warning: Pi UI stop command ran, but new process was not detected. "
                "Check /tmp/rpi_robo_eye_ui.log on Pi."
            )
    except Exception:
        pass


def _maybe_start_dashboard() -> None:
    """Start local dashboard server and optionally open dashboard in browser."""
    try:
        host, port = ensure_server_started()
    except Exception:
        return

    if os.getenv("ASSISTANT_OPEN_DASHBOARD", "1").strip().lower() not in {"1", "true", "yes", "on"}:
        return

    display_host = host
    if host in {"0.0.0.0", "::"}:
        display_host = "127.0.0.1"
    dashboard_url = f"http://{display_host}:{port}/dashboard"
    try:
        webbrowser.open_new_tab(dashboard_url)
    except Exception:
        pass


if __name__ == "__main__":
    _load_env_file()
    if os.getenv("ASSISTANT_MAIN_RUNNING", "0").strip() == "1":
        print("Detected recursive/duplicate main.py launch. Exiting this child process.")
        sys.exit(1)
    os.environ["ASSISTANT_MAIN_RUNNING"] = "1"
    _instance_lock = _acquire_single_instance_lock()
    if _instance_lock is None:
        print("Another assistant instance is already running. Close older main.py processes and retry.")
        sys.exit(1)
    _maybe_start_dashboard()
    _maybe_start_rpi_eye_ui()
    try:
        from session_manager import run_session
    except Exception as exc:
        print(f"Assistant session could not start, but dashboard/UI launch was attempted. Error: {exc}")
        print("Install dependencies with: pip install -r requirement.txt")
        sys.exit(1)
    run_session()
