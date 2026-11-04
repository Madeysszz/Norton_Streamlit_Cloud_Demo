import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]


def main():
    reports = ROOT / "test-results"
    reports.mkdir(exist_ok=True)
    result = {"status": "failed", "data_mode": "synthetic", "private_data_used": False,
              "health_http_status": None, "app_http_status": None}
    process = None
    environment = dict(os.environ)
    environment.pop("NORTON_DATA_ROOT", None)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    try:
        with (reports / "cloud-streamlit.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"),
                 "--server.address=127.0.0.1", f"--server.port={port}"],
                cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            try:
                deadline = time.monotonic() + 40
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError("Cloud demo exited before becoming ready")
                    try:
                        with urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=1) as response:
                            result["health_http_status"] = response.status
                        break
                    except (URLError, TimeoutError):
                        time.sleep(.25)
                if result["health_http_status"] != 200:
                    raise RuntimeError("Cloud demo health check did not return HTTP200")
                with urlopen(f"http://127.0.0.1:{port}/", timeout=5) as response:
                    result["app_http_status"] = response.status
                if result["app_http_status"] != 200:
                    raise RuntimeError("Cloud demo root page did not return HTTP200")
                result["status"] = "passed"
            finally:
                process.terminate()
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
    finally:
        result["server_stopped"] = process is not None and process.poll() is not None
        (reports / "cloud-smoke.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
