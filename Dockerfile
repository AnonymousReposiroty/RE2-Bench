FROM ubuntu:22.04
WORKDIR /re2bench

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    gnupg \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    build-essential \
    gfortran \
    git \
    graphviz \
    libgraphviz-dev \
    pkg-config \
    libsndfile1 \
    libfreetype6-dev \
    libpng-dev \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m ensurepip && \
    python3.12 -m pip install --upgrade pip

# Use python3.12 as the default python3/pip3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/pip3 pip3 /usr/local/bin/pip3 1 && \
    ln -sf /usr/bin/python3 /usr/bin/python

COPY ./requirements-pypi.txt ./requirements-git.txt /re2bench/

# Install stable PyPI packages first
RUN pip3 install --no-cache-dir --ignore-installed -r /re2bench/requirements-pypi.txt

# Install git-based packages with retry for transient GitHub errors
ENV GIT_HTTP_LOW_SPEED_LIMIT=1000 \
    GIT_HTTP_LOW_SPEED_TIME=60
RUN retry() { \
      local n=0 max=5 delay=15; \
      while true; do "$@" && return || { \
        n=$((n+1)); \
        [ $n -ge $max ] && echo ">>> Failed after $max attempts" && return 1; \
        echo ">>> Attempt $n/$max failed, retrying in ${delay}s..."; \
        sleep $delay; delay=$((delay*2)); \
      }; done; \
    } && \
    retry pip3 install --no-cache-dir -r /re2bench/requirements-git.txt

# Pass API key at runtime: docker run -e OPEN_ROUTER_KEY ...
ENV OPEN_ROUTER_KEY=""

COPY ./analysis /re2bench/analysis
COPY ./dataset /re2bench/dataset
COPY ./scripts /re2bench/scripts
COPY ./prompts /re2bench/prompts
COPY ./results /re2bench/results

