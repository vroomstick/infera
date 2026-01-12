# Custom PostgreSQL with pgvector + TimescaleDB
# Base: PostgreSQL 16

FROM postgres:16-bookworm

LABEL maintainer="Infera"
LABEL description="PostgreSQL 16 with pgvector and TimescaleDB extensions"

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    postgresql-server-dev-16 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install pgvector extension
RUN cd /tmp \
    && git clone --branch v0.7.4 https://github.com/pgvector/pgvector.git \
    && cd pgvector \
    && make \
    && make install \
    && rm -rf /tmp/pgvector

# Install TimescaleDB extension
RUN echo "deb https://packagecloud.io/timescale/timescaledb/debian/ bookworm main" > /etc/apt/sources.list.d/timescaledb.list \
    && curl -fsSL https://packagecloud.io/timescale/timescaledb/gpgkey | gpg --dearmor -o /etc/apt/trusted.gpg.d/timescaledb.gpg \
    && apt-get update \
    && apt-get install -y --no-install-recommends timescaledb-2-postgresql-16 \
    && rm -rf /var/lib/apt/lists/*

# Configure PostgreSQL for TimescaleDB
RUN echo "shared_preload_libraries = 'timescaledb'" >> /usr/share/postgresql/postgresql.conf.sample

# Copy initialization script
COPY init.sql /docker-entrypoint-initdb.d/

# Cleanup build dependencies to reduce image size
RUN apt-get purge -y --auto-remove build-essential git postgresql-server-dev-16 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 5432

