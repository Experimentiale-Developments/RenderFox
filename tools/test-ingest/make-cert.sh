#!/bin/sh
# Self-signed certificate for the dummy ingest, valid for "localhost". The uplink is
# pointed at it with CA_FILE=dummy.crt, so certificate checking stays switched on in
# the test exactly as it is against YouTube.
cd "$(dirname "$0")"
[ -f dummy.crt ] && exit 0
openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 -nodes -days 3650 \
    -keyout dummy.key -out dummy.crt -subj /CN=localhost -addext subjectAltName=DNS:localhost 2>/dev/null
chmod 600 dummy.key
