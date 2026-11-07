#!/bin/bash
# Copies the relay's stream to the ingest unchanged: no decoding, a few percent of a
# core. systemd restarts it whenever either end drops, and the encoder behind the
# relay keeps running meanwhile, so a network blip costs seconds, not a restart.
#
# Usage:
#   uplink.sh [primary|backup]
#
# uplink.env beside it holds PRIMARY_INGEST_URL and, for ingests with a backup (YouTube),
# BACKUP_INGEST_URL, and optionally CA_FILE; `renderfox youtube`, `twitch` or `rtmp` write
# it. Certificates are always verified.
set -euo pipefail
HOME_DIR=${RF_RELAY_HOME:-$(dirname "$(readlink -f "$0")")}
# shellcheck source=/dev/null
source "$HOME_DIR/uplink.env"

TARGET_ROLE="${1:-primary}"
case "$TARGET_ROLE" in
    backup)
        TARGET_URL="${BACKUP_INGEST_URL:-}"
        if [ -z "$TARGET_URL" ]; then
            echo "Error: BACKUP_INGEST_URL is not set in $HOME_DIR/uplink.env" >&2
            exit 1
        fi
        ;;
    primary|*)
        TARGET_URL="${PRIMARY_INGEST_URL:-}"
        if [ -z "$TARGET_URL" ]; then
            echo "Error: PRIMARY_INGEST_URL is not set in $HOME_DIR/uplink.env; run renderfox youtube (or twitch, rtmp)" >&2
            exit 1
        fi
        ;;
esac

# The URL ends in the stream key, which is not for the journal.
echo "Starting Render Fox uplink ($TARGET_ROLE) to ${TARGET_URL%/*}/..."

# One malloc arena and freed memory handed back sooner: about 4 MB less per ffmpeg, at
# the same CPU. It only copies packets.
export MALLOC_ARENA_MAX=1 MALLOC_TRIM_THRESHOLD_=131072

exec ffmpeg -hide_banner -loglevel warning -nostdin -nostats \
    -rw_timeout 5000000 -i rtmp://127.0.0.1:1935/fm \
    -map 0 -c copy \
    -f flv -flvflags no_duration_filesize -rw_timeout 5000000 \
    -tcp_nodelay 1 \
    -tls_verify 1 -ca_file "${CA_FILE:-/etc/ssl/certs/ca-certificates.crt}" "$TARGET_URL"
