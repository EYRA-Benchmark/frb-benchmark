# Copyright (C) 2019 by Xinping Deng
# Licensed under the Academic Free License version 3.0
# This program comes with ABSOLUTELY NO WARRANTY.
# You are free to modify and redistribute this code as long
# as you do not remove the above attribution and reasonably
# inform receipients that you have modified the original work.

FROM nvidia/cuda:10.1-devel-ubuntu16.04
#FROM nvidia/cuda:9.2-devel-ubuntu18.04
MAINTAINER Xinping Deng "xinping.deng@gmail.com"

# To get rid of "(TERM is not set, so the dialog frontend is not usable.)"
ARG DEBIAN_FRONTEND=noninteractive
# To use bash during build
SHELL ["/bin/bash", "-c"]          

RUN apt-get -y check && \
    apt-get -y update && \
    apt-get -y upgrade && \
    apt-get -y install apt-utils software-properties-common && \
    apt-get -y update --fix-missing && \
    apt-get -y upgrade && \
    apt-get -y install build-essential git cmake python-dev python3-dev python-pip unzip wget automake autoconf autotools-dev csh libtool m4 apt-utils &&\
    apt-get clean

# Install python packages
RUN pip install --upgrade pip -U && \
    hash pip && \
    pip install -U setuptools wheel numpy scipy ephem matplotlib katcp tornado astropy cmd2 -U

# Define directory for source code, OSTYPE and installation directory
ENV SOURCE           /root
ENV CRAFT            /root
ENV PREFIX           /usr/local
ENV INSTALL_BIN      $PREFIX/bin
ENV INSTALL_INCLUDES $PREFIX/include
ENV INSTALL_LIB      $PREFIX/lib
ENV INSTALL_SHARE    $PREFIX/share
ENV OSTYPE           linux-gnu
ENV LOGIN_ARCH	     linux
ENV CUDA_DIR         /usr/local/cuda
ENV CUDA_HOME        $CUDA_DIR
ENV CUDA_LIB         $CUDA_DIR/lib64
ENV CUDA_INCLUDE     $CUDA_DIR/include
ENV C_INCLUDE_PATH   $C_INCLUDE_PATH:$INCLUDE_INCLUDE
ENV LD_LIBRARY_PATH  $LD_LIBRARY_PATH:$INSTALL_LIB

# Install CRAFT
WORKDIR $SOURCE
RUN git clone https://bitbucket.csiro.au/scm/craf/craft.git -b dadain
WORKDIR $SOURCE/craft
RUN pip install -r requirements.txt
ENV PATH=$SOURCE/src/python:$PATH
ENV PYTHONPATH=$SOURCE/src/python:$PYTHONPATH

# install CUB - FREDDA needs this to compile
WORKDIR $SOURCE/include/
RUN wget https://github.com/NVlabs/cub/archive/1.8.0.zip
RUN unzip 1.8.0.zip
RUN rm 1.8.0.zip
ENV CUB_PATH=$SOURCE/include/cub-1.8.0/cub

# Install psrdada
WORKDIR $SOURCE
RUN git clone https://github.com/xinpingdeng/psrdada.git
WORKDIR $SOURCE/psrdada
RUN ./bootstrap && \
    ./configure --prefix=$PREFIX --with-cuda-include-dir=$CUDA_INCLUDE --with-cuda-lib-dir=$CUDA_LIB && \
    make -j $(nproc)&& \
    make install
WORKDIR $SOURCE
RUN rm -rf psrdada

## install psrdada
#WORKDIR $SOURCE/include
#RUN git clone git://git.code.sf.net/p/psrdada/code psrdada 
#WORKDIR $SOURCE/include/psrdada
#
#COPY PsrdadaMakefile.am Makefile.am
#
#RUN pwd && mkdir build/ && \
#    ./bootstrap && \
#    ./configure --prefix=/usr/local && \
#    make && \
#    make install && \
#    make clean
#
#ENV CUDA=/usr/local/cuda

# install FREDDA
ENV CUDA=/usr/local/cuda
ENV FREDDA=$SOURCE/craft/cuda-fdmt/cudafdmt/src/
WORKDIR $FREDDA
ENV PATH=$FREDDA:$PATH
#RUN cp fredda.mk.template fredda.mk &&\
#    sed -i s/60/70/g fredda.mk &&\
#    make; \
#    cp Makefile Makefile.old && \
#    sed -i s/-std=c\+\+11//g Makefile &&\
#    make; \
#    cp Makefile.old Makefile &&\
#    make
RUN cp fredda.mk.template fredda.mk &&\
    sed -i s/60/70/g fredda.mk &&\
    make
#-j 24

COPY run.sh /
COPY get_fil_header.py /

RUN chmod a+x /run.sh

CMD ["/run.sh"]
