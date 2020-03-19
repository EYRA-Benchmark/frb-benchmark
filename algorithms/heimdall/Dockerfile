#FROM nvidia/cuda:10.1-base
# ARTS cluster has driver 390.25, max CUDA version 9.1
FROM nvidia/cuda:9.1-devel-ubuntu17.04

# unsupported distro repos are moved
RUN sed -i -re 's/([a-z]{2}\.)?archive.ubuntu.com|security.ubuntu.com/old-releases.ubuntu.com/g' /etc/apt/sources.list

# To get rid of "(TERM is not set, so the dialog frontend is not usable.)"
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update &&\
    apt-get install -y software-properties-common &&\
    apt-get install -y build-essential git cmake automake autoconf autotools-dev csh libtool m4 libboost-all-dev

#RUN apt-get update
#RUN apt-get install -y software-properties-common
##RUN add-apt-repository -s -y ppa:kernsuite/kern-4
#RUN add-apt-repository -y multiverse
#RUN add-apt-repository -y restricted

# run again to fix added repos
#RUN sed -i -re 's/([a-z]{2}\.)?archive.ubuntu.com|security.ubuntu.com/old-releases.ubuntu.com/g' /etc/apt/sources.list
## ppa repo is also gone, replace by xenial and hope it works
#RUN sed -i 's/zesty/xenial/g' /etc/apt/sources.list.d/kernsuite-ubuntu-kern-4-zesty.list

#RUN apt-get update
#RUN apt-get install -y python
#RUN apt-get install -y heimdall-astro python

# install psrdada
WORKDIR /opt/frbbench
ADD psrdada.tar.gz /opt/frbbench/

RUN cd psrdada &&\
    ./bootstrap &&\
    ./configure --prefix=/usr/local &&\
    make -j &&\
    make install

# install dedisp
RUN git clone https://github.com/ewanbarr/dedisp-flexi.git &&\
    cd dedisp-flexi &&\
    make -j 32 &&\
    make install 

# install heimdall
# hack to get to compile wiht newer CUDA
RUN sed -i 's|#define __CUDACC_VER__|//#define __CUDACC_VER__|g' /usr/local/cuda/include/crt/common_functions.h
WORKDIR /opt/frbbench
RUN git clone https://git.code.sf.net/p/heimdall-astro/code heimdall-astro-code &&\
    cd heimdall-astro-code &&\
    ./bootstrap &&\
    ./configure --with-cuda-dir=/usr/local/cuda &&\
    chmod +x libtool &&\
    cp libtool /usr/local/bin &&\
    make -j &&\ 
    make install &&\
    ldconfig /usr/local/lib

WORKDIR /opt/frbbench
COPY run.sh .
COPY get_fil_header.py .
RUN chmod a+x run.sh

# add frbbench user; UID should be same as arts on ARTS cluster
RUN useradd -u 1006 -ms /bin/bash frbbench
RUN chown -R frbbench /opt/frbbench
USER frbbench

CMD ["./run.sh"]
