# # run_app.py — one-click launcher that opens your browser
# from __future__ import annotations
# import os, sys, socket, time, subprocess, threading, webbrowser

# APP_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "app.py"))

# def _pick_port(start=8501, tries=20) -> int:
#     for p in range(start, start + tries):
#         with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#             try:
#                 s.bind(("127.0.0.1", p))
#                 return p
#             except OSError:
#                 continue
#     raise RuntimeError("No free port found")

# def _wait_port(port: int, timeout: float = 30.0) -> bool:
#     deadline = time.time() + timeout
#     while time.time() < deadline:
#         try:
#             with socket.create_connection(("127.0.0.1", port), timeout=0.5):
#                 return True
#         except OSError:
#             time.sleep(0.3)
#     return False

# def _open_when_ready(url: str, port: int):
#     if _wait_port(port, timeout=40):
#         try:
#             webbrowser.open(url, new=1, autoraise=True)
#         except Exception:
#             pass

# def main():
#     if not os.path.exists(APP_FILE):
#         print(f"Could not find app.py at: {APP_FILE}")
#         return 1

#     port = _pick_port(8501)
#     url  = f"http://localhost:{port}"
#     py   = sys.executable
#     cmd  = [
#         py, "-m", "streamlit", "run", APP_FILE,
#         "--server.headless=false", "--server.address=localhost", f"--server.port={port}",
#     ]

#     # Print the link immediately, then open browser as soon as the port is live
#     print("\nLaunching Streamlit…")
#     print(f"Your app will be available at: {url}\n")

#     threading.Thread(target=_open_when_ready, args=(url, port), daemon=True).start()

#     # Stream Streamlit logs to this console
#     proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
#     try:
#         for line in iter(proc.stdout.readline, ""):
#             print(line, end="")
#     except KeyboardInterrupt:
#         pass
#     finally:
#         return proc.wait()

# if __name__ == "__main__":
#     raise SystemExit(main())
# run_app.py — launch Streamlit and open exactly one browser tab
from __future__ import annotations
import os, sys, socket, time, subprocess, threading, webbrowser

APP_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "app.py"))

def _pick_port(start=8501, tries=20) -> int:
    for p in range(start, start + tries):
        try:
            with socket.socket() as s:
                s.bind(("127.0.0.1", p))
            return p
        except OSError:
            continue
    raise RuntimeError("No free port found")

def _wait_port(port: int, timeout: float = 40.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.3)
    return False

def _open_once_when_ready(url: str, port: int):
    if _wait_port(port):
        try:
            webbrowser.open(url, new=1, autoraise=True)
        except Exception:
            pass

def main():
    if not os.path.exists(APP_FILE):
        print(f"Could not find app.py at: {APP_FILE}")
        return 1

    port = _pick_port()
    url  = f"http://localhost:{port}"
    print("\nLaunching Streamlit…")
    print(f"Your app will be available at: {url}\n")

    # Open the browser ONCE when the server is actually listening
    threading.Thread(target=_open_once_when_ready, args=(url, port), daemon=True).start()

    py  = sys.executable
    cmd = [
        py, "-m", "streamlit", "run", APP_FILE,
        "--server.headless=true",          # <- prevent Streamlit from auto-opening
        "--server.address=localhost",
        f"--server.port={port}",
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for line in proc.stdout:
            print(line, end="")
    except KeyboardInterrupt:
        pass
    return proc.wait()

if __name__ == "__main__":
    raise SystemExit(main())
