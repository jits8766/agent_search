ARG BASE_IMAGE=qdrant/qdrant:v1.13.2
FROM ${BASE_IMAGE}

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends openssl \
    && rm -rf /var/lib/apt/lists/*

COPY docker/qdrant/entrypoint-tls.sh /usr/local/bin/entrypoint-tls.sh
# Pre-create the cert dir and hand /qdrant to a non-root user so the entrypoint
# can write its per-start self-signed cert and Qdrant can persist data without
# root. Group 0 + g+w keeps it OpenShift/arbitrary-uid friendly.
RUN chmod +x /usr/local/bin/entrypoint-tls.sh \
    && mkdir -p /qdrant/tls \
    && (getent passwd qdrant >/dev/null 2>&1 \
        || useradd -r -u 1000 -g 0 -d /qdrant -s /sbin/nologin qdrant) \
    && chown -R 1000:0 /qdrant \
    && chmod -R g+w /qdrant

# Drop privileges: last USER must not be root (Semgrep security gate).
USER 1000

ENTRYPOINT ["/usr/local/bin/entrypoint-tls.sh"]
