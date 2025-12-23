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
	 libboost-all-dev \
	 lldb \
	 gdb \
	 strace \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

COPY . .

RUN mkdir -p ./gcare/build 
RUN cd ./gcare/build \
 && cmake .. \
 && make -j
