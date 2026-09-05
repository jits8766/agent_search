ARG BASE_IMAGE=clickhouse/clickhouse-server:24.8-alpine
FROM ${BASE_IMAGE}

RUN apk add --no-cache openssl

# Only the TLS config is baked in. memory.xml is intentionally left out — its
# 22 GiB cap is sized for the local docker-compose stack and must not apply to
# the smaller ECS tasks (see docs/guides/deploy-clickhouse-qdrant.md).
COPY clickhouse-config/tls.xml /etc/clickhouse-server/config.d/tls.xml
COPY docker/clickhouse/entrypoint-tls.sh /usr/local/bin/entrypoint-tls.sh
# Pre-create the cert dir owned by the clickhouse user so the entrypoint can
# write its per-start self-signed cert without root.
RUN chmod +x /usr/local/bin/entrypoint-tls.sh \
    && mkdir -p /etc/clickhouse-server/certs \
    && chown -R clickhouse:clickhouse /etc/clickhouse-server/certs

# Drop privileges: container must not run as root (Semgrep security gate).
USER clickhouse

ENTRYPOINT ["/usr/local/bin/entrypoint-tls.sh"]
