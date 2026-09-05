#!/usr/bin/env python3
"""TLS bootstrap: ensure a self-signed cert exists, then exec uvicorn over HTTPS.

Replaces the former shell + openssl-CLI entrypoint so the runtime image needs
neither a shell nor the openssl binary (hardened, minimal base image). The cert
is generated per task start via the `cryptography` library (already a pinned
dependency) into the container filesystem — no cert material lives in image
layers. The ALB does not validate the backend cert, so an ephemeral self-signed
cert is sufficient (same approach as the ClickHouse/Qdrant TLS sidecars).
"""

import datetime as dt
import os

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

CERT_DIR = "/app/tls"
CERT_PATH = os.path.join(CERT_DIR, "cert.pem")
KEY_PATH = os.path.join(CERT_DIR, "key.pem")


def ensure_cert() -> None:
    """Generate a self-signed cert + key once per container filesystem."""
    if os.path.exists(CERT_PATH) and os.path.exists(KEY_PATH):
        return
    os.makedirs(CERT_DIR, exist_ok=True)

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name(
        [x509.NameAttribute(NameOID.COMMON_NAME, "auc-semantic-search")]
    )
    now = dt.datetime.now(dt.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=1))
        .not_valid_after(now + dt.timedelta(days=3650))
        .sign(key, hashes.SHA256())
    )

    # Private key first, mode 0600 so it is never group/world-readable.
    fd = os.open(KEY_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as key_file:
        key_file.write(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.TraditionalOpenSSL,
                serialization.NoEncryption(),
            )
        )
    with open(CERT_PATH, "wb") as cert_file:
        cert_file.write(cert.public_bytes(serialization.Encoding.PEM))


def main() -> None:
    ensure_cert()
    os.makedirs("/app/output", exist_ok=True)
    # exec so uvicorn becomes PID 1 and receives signals directly.
    os.execvp(
        "uvicorn",
        [
            "uvicorn",
            "semantic_search.app:app",
            "--host", "0.0.0.0",
            "--port", "8085",
            "--ssl-keyfile", KEY_PATH,
            "--ssl-certfile", CERT_PATH,
        ],
    )


if __name__ == "__main__":
    main()
