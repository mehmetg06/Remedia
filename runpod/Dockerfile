# syntax=docker/dockerfile:1.7
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ARG DEBIAN_FRONTEND=noninteractive
ARG GNINA_VERSION=v1.3
ARG REMEDIA_COMMIT=unknown

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    REMEDIA_HOME=/workspace/Remedia \
    REMEDIA_WORKSPACE=/workspace \
    GNINA_PATH=/usr/local/bin/gnina \
    PATH=/opt/remedia-fpocket/bin:${PATH}

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
        bzip2 ca-certificates curl git rsync wget \
    && rm -rf /var/lib/apt/lists/*

COPY runpod/requirements.txt /tmp/remedia-runpod-requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /tmp/remedia-runpod-requirements.txt

RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
      | tar -xj -C /usr/local/bin --strip-components=1 bin/micromamba \
    && micromamba create -y -p /opt/remedia-fpocket -c conda-forge -c bioconda fpocket \
    && micromamba clean --all --yes

RUN curl --fail --location --retry 3 \
      "https://github.com/gnina/gnina/releases/download/${GNINA_VERSION}/gnina" \
      --output /usr/local/bin/gnina \
    && chmod 0755 /usr/local/bin/gnina \
    && /usr/local/bin/gnina --version

WORKDIR /opt/remedia-src
COPY . /opt/remedia-src
RUN printf '%s\n' "${REMEDIA_COMMIT}" > /opt/remedia-src/.image_commit \
    && chmod +x /opt/remedia-src/runpod/start-remedia.sh

EXPOSE 8888
WORKDIR /workspace

ENTRYPOINT []
CMD ["/opt/remedia-src/runpod/start-remedia.sh"]
