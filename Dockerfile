FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV NVM_DIR=/root/.nvm
ENV NODE_VERSION=22.12.0

RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    curl \
    python3 \
    python3-pip \
    ca-certificates \
    bash \
    && rm -rf /var/lib/apt/lists/*

RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash

RUN bash -c "source $NVM_DIR/nvm.sh \
    && nvm install $NODE_VERSION \
    && nvm use $NODE_VERSION \
    && nvm alias default $NODE_VERSION"

ENV PATH=$NVM_DIR/versions/node/v$NODE_VERSION/bin:$PATH

RUN node -v && npm -v

WORKDIR /app

RUN git clone https://gitlab.com/Westerbay/stylegan-and-cheap-dit.git
WORKDIR /app/stylegan-and-cheap-dit
RUN git lfs install
RUN git lfs pull
RUN pip3 install --no-cache-dir -r requirements.txt

WORKDIR /app
RUN git clone https://gitlab.com/Westerbay/ui-for-generative-models.git
WORKDIR /app/ui-for-generative-models/application
RUN npm install

EXPOSE 5173 7050

CMD bash -c "cd /app/stylegan-and-cheap-dit && python3 api_ldm.py & npm run start"

