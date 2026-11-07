#!/bin/sh
# Installs or refreshes the Render Fox services for this checkout. Run with sudo; setup.sh
# calls it. The services run as the checkout's owner, from wherever it lives.
#
#   sudo ./install.sh                 the relay, the encoder, the uplinks and the watchdog
#   sudo ./install.sh --test-ingest   also the stand-in for YouTube (tools/test-ingest)
set -eu
RF=$(dirname "$(readlink -f "$0")")
RF_USER=$(stat -c %U "$RF")
[ "$(id -u)" -eq 0 ] || { echo "run with sudo" >&2; exit 1; }

units="renderfox-relay renderfox-encode renderfox-uplink renderfox-uplink-backup renderfox-watchdog"
[ "${1:-}" = "--test-ingest" ] && units="$units renderfox-test-ingest"
for u in $units; do
    sed -e "s|@RF_HOME@|$RF|g" -e "s|@RF_USER@|$RF_USER|g" "$RF/systemd/$u.service.in" > "/etc/systemd/system/$u.service"
    chmod 644 "/etc/systemd/system/$u.service"
done
install -m 644 "$RF/relay/99-renderfox-tcp.conf" /etc/sysctl.d/
sysctl -q --system
ln -sf "$RF/renderfox" /usr/local/bin/renderfox
systemctl daemon-reload
# The backup uplink and the test ingest are switched on by `renderfox youtube` and by hand.
systemctl enable -q renderfox-relay renderfox-encode renderfox-uplink renderfox-watchdog
echo "installed for $RF_USER at $RF: $units"
