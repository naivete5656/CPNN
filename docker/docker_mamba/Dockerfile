FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel
LABEL maintainer="nishimura"

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime
ENV TZ=Asia/Tokyo

ENV PYTHONDONTWRITEBYTECODE=1

RUN apt update && apt install -y --no-install-recommends sudo \
    libsm6 libxext6 libgtk2.0-dev libjpeg-dev libpng-dev libgl1-mesa-glx librdmacm1 libibverbs1 libsndfile1-dev \
    librdmacm1 libibverbs1 libopenmpi-dev\
    g++ less zsh \
    ibverbs-providers git gcc tesseract-ocr espeak-ng ca-certificates bzip2 \
    cmake tree htop bmon iotop build-essential openjdk-8-jdk-headless\ 
    ffmpeg curl wget 

RUN apt-get update && \
   DEBIAN_FRONTEND=noninteractive \
   apt-get -y install default-jre-headless && \
   apt-get clean && \
   rm -rf /var/lib/apt/lists/*

RUN apt install -y libsndfile1-dev
RUN apt update && apt-get install -y libxml-parser-perl

RUN /opt/conda/bin/python -mpip install --no-cache-dir\
    scipy\
    loguru\
    requests\
    lightning\
    einops\
    addict\
    wandb\
    timm\
    scvi-tools\
    ema_pytorch\
    huggingface_hub 

RUN /opt/conda/bin/python -mpip install --no-cache-dir https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.2/causal_conv1d-1.5.2+cu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
RUN /opt/conda/bin/python -mpip install --no-cache-dir https://github.com/state-spaces/mamba/releases/download/v2.2.5/mamba_ssm-2.2.5+cu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
RUN /opt/conda/bin/conda clean -ya

ARG UID
ARG USER
ARG PASSWORD

RUN groupadd -g 1000 ${USER}_group
RUN useradd -m --uid=${UID} --gid=${USER}_group --groups=sudo ${USER}
RUN echo ${USER}:${PASSWORD} | chpasswd
RUN echo 'root:root' | chpasswd
USER ${USER}



CMD ["/bin/zsh"]