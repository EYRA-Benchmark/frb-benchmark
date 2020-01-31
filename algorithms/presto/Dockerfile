FROM gcc:9
WORKDIR /root/install

RUN useradd -ms /bin/bash frbbench

# install cmake 
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get -y update && apt-get -y install apt-utils 
RUN update-alternatives --install /usr/bin/gfortran gfortran /usr/local/bin/gfortran 999
RUN apt-get -y update && \
    apt-get -y install cmake && \
    apt-get -y install apt-utils && \
    apt-get -y upgrade && \
    apt-get -y remove --purge gfortran && \
    apt-get -y install gfortran && \
    apt-get clean

# install a few other basic tools
RUN apt-get -y install tcsh libgl1-mesa-dev libgsl0-dev libblas-dev liblapack-dev&& \
    apt-get clean


RUN mkdir -p /opt/frbbench && \
    chown frbbench /opt/frbbench

WORKDIR /opt/frbbench

RUN wget -q https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/frbbench/miniconda && \
    rm Miniconda3-latest-Linux-x86_64.sh

ENV LD_LIBRARY_PATH="/usr/local/lib"

# tempo 
ENV TEMPO=/opt/frbbench/tempo 
ENV PATH=$PATH:/opt/frbbench/tempo/bin

RUN git clone http://git.code.sf.net/p/tempo/tempo

WORKDIR $TEMPO
RUN chmod +x prepare 
RUN head /opt/frbbench/tempo/prepare
RUN /opt/frbbench/tempo/prepare && \
    LDFLAGS="-L/opt/frbbench/miniconda/lib" \
    CPPFLAGS="-I/opt/frbbench/miniconda/include" \
    ./configure --prefix=/opt/frbbench/tempo --with-blas=mkl_rt --with-lapack=mkl_rt&& \
    make && \
    make install && \
    cd util/print_resid && \
    make

ENV PATH="/opt/frbbench/miniconda/bin:/opt/frbbench/bin:${PATH}"

# install basic Python packages
RUN conda install -y numpy cython scipy pyqt && \
    conda clean -a
RUN sync
# install libstempo (before other Anaconda packages, esp. matplotlib, so there's no libgcc confusion)
# get 2.3.3, specifically at 25 Aug 2017 (git sha 9cb7552)

# non-standard-Anaconda packages
RUN pip install healpy line_profiler jplephem corner numdifftools 

USER root
RUN sed -i -e "s#main#main non-free#g" /etc/apt/sources.list
RUN apt-get update && apt-get -y install pgplot5 libfftw3-dev libcfitsio-dev libopenblas-dev

RUN apt-get update && apt-get -y install pgplot5 libfftw3-dev libcfitsio-dev latex2html glib2.0-dev pkg-config

ENV PGPLOT_DIR=/usr/lib/pgplot5 
ENV PGPLOT_FONT=/usr/lib/pgplot5/grfont.dat 
ENV PGPLOT_INCLUDES=/usr/include 
ENV PGPLOT_BACKGROUND=white 
ENV PGPLOT_FOREGROUND=black 
ENV PGPLOT_DEV=/xs

ENV PSRHOME=/opt/frbbench

ENV PRESTO=$PSRHOME/presto 
ENV PATH=$PATH:$PRESTO/bin 
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PRESTO/lib 
ENV PYTHONPATH=$PYTHONPATH:$PRESTO/lib/python

WORKDIR /opt/frbbench
RUN git clone https://github.com/scottransom/presto.git

WORKDIR $PRESTO/src
RUN make prep && \
    make

WORKDIR $PRESTO
RUN pip install futures pyslalib
RUN python setup.py install

USER frbbench
WORKDIR /home/frbbench
