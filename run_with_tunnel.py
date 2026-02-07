import subprocess
import sys
import time
import os
from pathlib import Path
from urllib.request import urlopen

# --- SETTINGS ---
PORT = "8502"
SUBDOMAIN = "infant_app"        # public URL: https://infant_app.loca.lt
APP_FILE = "app.py"             # or "app_try.py" if needed
# ---------------

BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

streamlit_proc = None
tunnel_proc = None


def kill_process(proc):
    """Kills a subprocess safely on Windows."""
    if proc is None:
        return
    try:
        proc.terminate()
        time.sleep(1)
        proc.kill()
    except Exception:
        pass


def free_port(port: str):
    """Force-free a port on Windows."""
    try:
        find_cmd = f'netstat -ano | findstr :{port}'
        output = subprocess.check_output(find_cmd, shell=True).decode(errors="ignore")

        for line in output.splitlines():
            parts = line.split()
            pid = parts[-1]
            if pid.isdigit():
                print(f"Closing process on port {port}: PID {pid}")
                subprocess.call(f"taskkill /PID {pid} /F", shell=True)
    except Exception:
        pass


def get_tunnel_password() -> str:
    """
    Fetch the LocalTunnel password from https://loca.lt/mytunnelpassword.
    Returns the text (stripped). If it fails, returns a message.
    """
    url = "https://loca.lt/mytunnelpassword"
    try:
        with urlopen(url, timeout=10) as resp:
            text = resp.read().decode("utf-8", errors="ignore").strip()
        return text
    except Exception as e:
        return f"[could not fetch password: {e}]"


def main():
    global streamlit_proc, tunnel_proc

    # Always free the port BEFORE starting
    free_port(PORT)

    # 1) Start Streamlit
    streamlit_cmd = [
        sys.executable, "-m", "streamlit",
        "run", APP_FILE,
        "--server.port", PORT,
    ]
    print("Starting Streamlit:", " ".join(streamlit_cmd))
    streamlit_proc = subprocess.Popen(streamlit_cmd)

    # Give Streamlit a few seconds to start
    time.sleep(5)

    # 2) Start the LocalTunnel process via shell (so Windows can find npx)
    tunnel_cmd = f"npx localtunnel --port {PORT} --subdomain {SUBDOMAIN}"
    print("Starting LocalTunnel:", tunnel_cmd)
    tunnel_proc = subprocess.Popen(tunnel_cmd, shell=True)

    # Wait a bit so the tunnel actually comes up
    time.sleep(8)

    public_url = f"https://{SUBDOMAIN}.loca.lt"
    password = get_tunnel_password()

    print("\n====================================")
    print("the url is:")
    print(f"  {public_url}")
    print("\nthe password is:")
    print(f"  {password}")
    print("====================================\n")
    print("Press STOP or Ctrl+C to close everything.\n")

    try:
        # Wait for Streamlit to finish
        streamlit_proc.wait()
    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping processes…")
        kill_process(streamlit_proc)
        kill_process(tunnel_proc)
        free_port(PORT)
        print("✔ Clean shutdown complete. Port freed.")


if __name__ == "__main__":
    main()
