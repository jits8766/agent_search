#!/bin/sh
# Stack Safeguard requires the container to serve HTTPS to the load balancer.
# The ALB does not validate the backend cert, so an ephemeral self-signed cert
# generated per task start is sufficient (no cert material lives in image layers).
set -e

CERT_DIR=/etc/clickhouse-server/certs
if [ ! -f "$CERT_DIR/server.crt" ]; then
    mkdir -p "$CERT_DIR"
    openssl req -subj "/CN=clickhouse" -new -newkey rsa:2048 -days 3650 \
        -nodes -x509 -keyout "$CERT_DIR/server.key" -out "$CERT_DIR/server.crt"
    chown -R clickhouse:clickhouse "$CERT_DIR"
fi

exec /entrypoint.sh "$@"
