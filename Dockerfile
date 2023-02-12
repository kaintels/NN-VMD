FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

RUN apt-get update;
RUN apt-get dist-upgrade --yes;
RUN apt-get install --yes git;
RUN apt-get install --yes wget curl unzip;
RUN apt-get clean;

ENV JULIA_VERSION=1.8.5


RUN mkdir /opt/julia-${JULIA_VERSION} && \
    cd /tmp && \
    wget -q https://julialang-s3.julialang.org/bin/linux/x64/`echo ${JULIA_VERSION} | cut -d. -f 1,2`/julia-${JULIA_VERSION}-linux-x86_64.tar.gz && \
    tar xzf julia-${JULIA_VERSION}-linux-x86_64.tar.gz -C /opt/julia-${JULIA_VERSION} --strip-components=1 && \
    rm /tmp/julia-${JULIA_VERSION}-linux-x86_64.tar.gz

RUN ln -fs /opt/julia-*/bin/julia /usr/local/bin/julia

COPY ./ ./
RUN bash data.sh
RUN pip install -r requirements.txt
RUN julia requirements.jl
RUN python julia_setting.py

RUN echo "##########################"
RUN echo "data processing finished. enjoy."
RUN echo "please EXECUTE 'docker run -it --gpus all nn-vmd:latest bash train.sh'"
RUN echo "you can modify train.sh"
RUN echo "##########################"

WORKDIR /workspace


