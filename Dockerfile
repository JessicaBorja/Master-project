FROM nvidia/cuda:11.6.0-base-ubuntu20.04

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda --version
RUN conda init bash

RUN apt-get -qq update && apt-get -qqy upgrade
RUN apt-get -qqy install xserver-xorg-core xserver-xorg-video-dummy libxcursor1 x11vnc unzip pciutils software-properties-common kmod gcc make linux-headers-generic wget
COPY start.sh /root/start.sh
# RUN start.sh
