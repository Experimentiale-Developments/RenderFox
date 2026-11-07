#!/usr/bin/env python3
"""Render Fox watchdog.

Watches the local MediaMTX relay, the encoder feeding it and the uplinks reading it,
and restarts an uplink systemd has given up on. Expects the backup uplink only when
uplink.env names a backup ingest.
"""

import json
import logging
import os
import subprocess
import time
import urllib.request
import urllib.error

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [watchdog] %(levelname)s: %(message)s"
)

MEDIAMTX_API = "http://127.0.0.1:9997/v3"
CHECK_INTERVAL_SEC = 5.0
RELAY_DIR = os.environ.get("RF_RELAY_HOME", os.path.dirname(os.path.abspath(__file__)))
RUN_DIR = os.environ.get("RF_RUN_DIR", os.path.join(os.path.dirname(RELAY_DIR), "run"))
HEARTBEAT_PATH = os.path.join(RUN_DIR, "watchdog-heartbeat")


def uplink_services():
    """The uplinks uplink.env asks for: the primary always, the backup if it names one."""
    services = ["renderfox-uplink"]
    try:
        with open(os.path.join(RELAY_DIR, "uplink.env")) as f:
            for line in f:
                key, _, value = line.strip().partition("=")
                if key == "BACKUP_INGEST_URL" and value.strip('"\''):
                    services.append("renderfox-uplink-backup")
    except OSError:
        pass
    return services


def fetch_json(endpoint: str):
    url = f"{MEDIAMTX_API}{endpoint}"
    req = urllib.request.Request(url, headers={"User-Agent": "RenderFox-Watchdog/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        logging.warning("Failed to query MediaMTX API %s: %s", endpoint, e)
    return None


def restart_service(unit_name: str, reason: str):
    logging.warning("Restarting %s: %s", unit_name, reason)
    try:
        subprocess.run(["systemctl", "restart", unit_name], check=True, timeout=10.0)
        logging.info("Successfully restarted %s", unit_name)
    except Exception as e:
        logging.error("Failed to restart %s: %s", unit_name, e)


def check_service_active(unit_name: str) -> bool:
    try:
        res = subprocess.run(
            ["systemctl", "is-active", unit_name],
            capture_output=True,
            text=True,
            timeout=5.0
        )
        return res.stdout.strip() == "active"
    except Exception:
        return False


def main():
    logging.info("Starting Render Fox stream watchdog...")
    os.makedirs(RUN_DIR, exist_ok=True)

    last_inbound_bytes = -1
    last_inbound_change_time = time.time()
    last_uplink_restart = {}

    while True:
        now = time.time()
        services = uplink_services()

        # Update heartbeat
        try:
            with open(HEARTBEAT_PATH, "w") as f:
                f.write(f"{now:.3f}\n")
        except Exception:
            pass

        # 1. MediaMTX path /fm. With every uplink reading, there is nothing more to check:
        # asking systemd about them every few seconds costs a process each time.
        path_info = fetch_json("/paths/get/fm")
        healthy = bool(path_info) and path_info.get("ready", False) and len(path_info.get("readers", [])) >= len(services)

        # 2. Otherwise, systemd's view of the uplinks
        if not healthy:
            for svc in services:
                if not check_service_active(svc):
                    if now - last_uplink_restart.get(svc, 0.0) > 15.0:
                        restart_service(svc, "service not active in systemd")
                        last_uplink_restart[svc] = now

        if path_info:
            is_ready = path_info.get("ready", False)
            inbound_bytes = path_info.get("bytesReceived", 0)
            readers = path_info.get("readers", [])
            reader_count = len(readers)

            if is_ready:
                if inbound_bytes != last_inbound_bytes:
                    last_inbound_bytes = inbound_bytes
                    last_inbound_change_time = now
                elif now - last_inbound_change_time > 20.0:
                    logging.warning(
                        "Inbound stream stalled: no bytes received for %.1fs",
                        now - last_inbound_change_time
                    )

            if is_ready and reader_count < len(services):
                logging.info(
                    "Active uplink readers: %d/%d connected. Checking services...",
                    reader_count, len(services)
                )

        # 3. Check RTMP connections for discarded frames (stalled socket), twice a minute
        conns_info = fetch_json("/rtmpconns/list") if int(now) % 30 < CHECK_INTERVAL_SEC else None
        if conns_info:
            items = conns_info.get("items", [])
            for conn in items:
                if conn.get("state") == "read":
                    discarded = conn.get("outboundFramesDiscarded", 0)
                    if discarded > 30:
                        logging.warning(
                            "Reader %s has %d discarded frames (connection choked)",
                            conn.get("id"),
                            discarded
                        )

        time.sleep(CHECK_INTERVAL_SEC)


if __name__ == "__main__":
    main()
