FROM debian:stable-slim

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
	 build-essential \
	 python3 \
	 python3-pip \
	 python3-numpy \
	 python3-tqdm \
	 cmake \
	 git \
	 libboost-dev \
	 libboost-program-options-dev \
 && rm -rf /var/lib/apt/lists/* \
 && apt-get clean

WORKDIR /usr/src/app

COPY gcare ./gcare
COPY *.py .
COPY ../custom-run-build.sh ./custom-run-build.sh

RUN mkdir -p ./gcare/build \
 && cd ./gcare/build \
 && cmake .. \
 && make -j \
 && cd ../.. \
 && rm -rf ./gcare/build/*.o ./gcare/build/CMakeFiles \
 && apt-get purge -y --auto-remove build-essential cmake git \
 && rm -rf /root/.cache
