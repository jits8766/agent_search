#!/bin/sh
# Stack Safeguard requires the container to serve HTTPS to the load balancer.
# The ALB does not validate the backend cert, so an ephemeral self-signed cert
# generated per task start is sufficient (no cert material lives in image layers).
#
# TLS must be enabled from inside the image: the Katana publish/promote flow does
# not inject configs/katana-qdrant.yaml's `environment:` block into the ECS task
# definition, so we cannot rely on those vars being present at runtime. We set
# them here (honoring any override) so Qdrant always serves HTTPS on 8443 — the
# same self-contained approach ClickHouse uses by baking tls.xml into its image.
set -e

CERT_DIR=/qdrant/tls
if [ ! -f "$CERT_DIR/cert.pem" ]; then
    mkdir -p "$CERT_DIR"
    openssl req -subj "/CN=qdrant" -new -newkey rsa:2048 -days 3650 \
        -nodes -x509 -keyout "$CERT_DIR/key.pem" -out "$CERT_DIR/cert.pem"
fi

export QDRANT__SERVICE__HTTP_PORT="${QDRANT__SERVICE__HTTP_PORT:-8443}"
export QDRANT__SERVICE__ENABLE_TLS="${QDRANT__SERVICE__ENABLE_TLS:-true}"
export QDRANT__TLS__CERT="${QDRANT__TLS__CERT:-$CERT_DIR/cert.pem}"
export QDRANT__TLS__KEY="${QDRANT__TLS__KEY:-$CERT_DIR/key.pem}"

exec ./entrypoint.sh "$@"
