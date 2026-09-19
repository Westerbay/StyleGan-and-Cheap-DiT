FROM node:22-bookworm-slim

LABEL org.opencontainers.image.source="https://github.com/Westerbay/StyleGan-and-Cheap-DiT"
LABEL org.opencontainers.image.licenses="MIT"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv git ca-certificates tini \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv

WORKDIR /app/stylegan-and-cheap-dit
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Pin the separate UI repository so a rebuild uses the same frontend source.
ARG UI_REF=147b5d7c1aa3d7e356845a1c0c32c542eeb6f490
RUN git init /app/ui-for-generative-models \
    && cd /app/ui-for-generative-models \
    && git remote add origin https://github.com/Westerbay/ui-for-generative-models.git \
    && git fetch --depth=1 origin "$UI_REF" \
    && git checkout --detach FETCH_HEAD \
    && rm -rf .git \
    && cd application && npm ci && npm run build

COPY . .
# Fail early when the build context contains LFS pointers instead of weights.
RUN python scripts/check_models.py
RUN chmod +x docker/entrypoint.sh

EXPOSE 5173 7050
ENTRYPOINT ["/usr/bin/tini", "--", "/app/stylegan-and-cheap-dit/docker/entrypoint.sh"]
