FROM jupyter/datascience-notebook:82b978b3ceeb

USER root
RUN mkdir /opt/pulsar
RUN chown -R jovyan /opt/pulsar
RUN groupadd -g 1347800170 nanograv-data
RUN usermod -a -G nanograv-data jovyan
RUN sed -i -e s#jessie\ main#jessie\ main\ non-free#g /etc/apt/sources.list
RUN apt-get update -y && apt-get install -y \
    autoconf \
    libtool \
    pgplot5 \
    libfftw3-bin \
    libfftw3-dbg \
    libfftw3-dev \
    libfftw3-double3 \
    libfftw3-long3 \
    libfftw3-quad3 \
    libfftw3-single3 \
    libcfitsio-dev \ 
    libglib2.0-dev \
    libx11-dev \ 
    swig \
    pkg-config \ 
    openssh-client
USER jovyan
# make calceph
RUN wget --no-check-certificate -q https://www.imcce.fr/content/medias/recherche/equipes/asd/calceph/calceph-2.3.2.tar.gz  && \
    tar zxvf calceph-2.3.2.tar.gz && \
    cd calceph-2.3.2 && \
    ./configure --prefix=/opt/pulsar && \
    make && make install && \
    cd .. && rm -rf calceph-2.3.2 calceph-2.3.2.tar.gz

# make tempo2
RUN wget -q https://bitbucket.org/psrsoft/tempo2/get/master.tar.gz && \
    tar zxf master.tar.gz && \
    cd psrsoft-tempo2-* && \
    ./bootstrap && \    
    CPPFLAGS="-I/opt/pulsar/include" LDFLAGS="-L/opt/pulsar/lib" ./configure --prefix=/opt/pulsar --with-calceph=/opt/pulsar/ && \
    make && make install && \
    mkdir /opt/pulsar/share/tempo2 && \
    cp -Rp T2runtime/* /opt/pulsar/share/tempo2/. && \
    cd .. && rm -rf psrsoft-tempo2-* master.tar.gz

# get extra ephemeris
RUN cd /opt/pulsar/share/tempo2/ephemeris && \
    wget -q ftp://ssd.jpl.nasa.gov/pub/eph/planets/bsp/de435t.bsp && \
    wget -q ftp://ssd.jpl.nasa.gov/pub/eph/planets/bsp/de436t.bsp 
#    wget -q https://github.com/nanograv/tempo/raw/master/ephem/DE435.1950.2050 && \
#    wget -q https://github.com/nanograv/tempo/raw/master/ephem/DE436.1950.2050

# install libstempo (before other Anaconda packages, esp. matplotlib, so there's no libgcc confusion)
RUN git clone https://github.com/vallis/libstempo.git && \
    cd libstempo && \
    pip install .  --global-option="build_ext" --global-option="--with-tempo2=/opt/pulsar" && \
    cp -rp demo /home/jovyan/libstempo-demo && chown -R jovyan /home/jovyan/libstempo-demo && \
    bash -c "source activate python2 && \
    pip install .  --global-option="build_ext" --global-option=\"--with-tempo2=/opt/pulsar\"" && \
    cd .. && rm -rf libstempo


# non-standard-Anaconda packages
RUN bash -c "source activate python2 && pip install healpy acor line_profiler"
RUN pip install healpy acor line_profiler

# install PTMCMCSampler
RUN git clone https://github.com/jellis18/PTMCMCSampler && \
    cd PTMCMCSampler && \
    bash -c "source activate python2 && pip install . && \
    cd .. && rm -rf PTMCMCSampler"

# install PAL2 (do not remove it)
RUN bash -c "git clone https://github.com/jellis18/PAL2.git && \
    cd PAL2 && \
    source activate python2 && pip install . && \
    cp -rp demo /home/jovyan/PAL2-demo && chown -R jovyan /home/jovyan/PAL2-demo && \
    cd .. && rm -rf PAL2"

# install NX01 (rather, check it out and copy it to the jovyan user)
#RUN git clone https://github.com/stevertaylor/NX01.git && \
#    cp -rp NX01 /home/jovyan/NX01 && chown -R jovyan /home/jovyan/NX01 


ENV PGPLOT_DIR=/usr/lib/pgplot5 
ENV PGPLOT_FONT=/usr/lib/pgplot5/grfont.dat 
ENV PGPLOT_INCLUDES=/usr/include 
ENV PGPLOT_BACKGROUND=white 
ENV PGPLOT_FOREGROUND=black 
ENV PGPLOT_DEV=/xs

ENV PSRHOME=/opt/pulsar

ENV PRESTO=$PSRHOME/presto 
ENV PATH=$PATH:$PRESTO/bin 
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PRESTO/lib 
ENV PYTHONPATH=$PYTHONPATH:$PRESTO/lib/python

RUN git clone https://github.com/scottransom/presto.git
RUN mv presto $PSRHOME/presto

WORKDIR $PRESTO/src
RUN make prep && \
    make
WORKDIR $PRESTO/python
RUN /opt/conda/envs/python2/bin/python setup.py install --home=$PRESTO 

ENV PSRCHIVE=$PSRHOME/psrchive 
ENV PATH=$PATH:$PSRHOME/bin 
ENV C_INCLUDE_PATH=$C_INCLUDE_PATH:$PSRCHIVE/install/include 
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PSRCHIVE/install/lib 
ENV PYTHONPATH=$PYTHONPATH:$PSRCHIVE/install/lib/python2.7/site-packages 

RUN git clone git://git.code.sf.net/p/psrchive/code psrchive
RUN mv psrchive $PSRHOME
WORKDIR $PSRCHIVE
#RUN sed -i -e s/libgnu_la_SOURCES\ +=/libgnu_la_SOURCES\ =/g Util/third/gnu/Makefile.am
USER root
RUN apt install -y tcsh
USER jovyan
RUN /bin/bash -c 'source /opt/conda/bin/activate python2; \
    git checkout c262f4;\
    ./bootstrap; \
    ./configure F77=gfortran --prefix=$PSRHOME --enable-shared CFLAGS="-fPIC" FFLAGS="-fPIC";\
    make && make install && make clean;'
RUN cp /opt/pulsar/lib/python2.7/site-packages/* /opt/conda/envs/python2/lib/python2.7/site-packages/

ENV NANOGRAVDATA=/nanograv/data

COPY start-singleuser.sh /usr/local/bin/start-singleuser.sh
COPY notebook-setup.sh /usr/local/bin/notebook-setup.sh
COPY 00-nanograv.py /home/jovyan/.ipython/profile_default/startup/00-nanograv.py

RUN mkdir /home/jovyan/.local
RUN ln -sf /home/jovyan/work/custom/lib /home/jovyan/.local/lib
RUN ln -sf /home/jovyan/work/custom/bin /home/jovyan/.local/bin


USER root
RUN apt-get install -y tcsh libsuitesparse-dev rsync dvipng
USER jovyan

COPY requirements.txt /var/tmp/requirements.txt
RUN bash -c "source activate python2 && pip install -r /var/tmp/requirements.txt"
RUN pip install -r /var/tmp/requirements.txt

# tempo 
ENV TEMPO=$PSRHOME/tempo 
ENV PATH=$PATH:$PSRHOME/tempo/bin
ENV TEMPO2=$PSRHOME/share/tempo2

RUN git clone http://git.code.sf.net/p/tempo/tempo
RUN mv tempo $PSRHOME

WORKDIR ${TEMPO}
RUN ./prepare && \
    ./configure --prefix=$PSRHOME/tempo && \
    make && \
    make install && \
    cd util/print_resid && \
    make

# install Piccard

WORKDIR /home/jovyan

#RUN git clone https://github.com/vhaasteren/piccard.git && \
#    cd piccard && \
#    sed -i -e s#liomp5#lgomp#g setup.py && \
#    /bin/bash -c "source /opt/conda/bin/activate python2 && python setup.py install" && \
#    cd /home/jovyan/work && ln -s /home/jovyan/piccard piccard 

RUN pip install git+https://github.com/nanograv/PINT.git && \
    /bin/bash -c "source /opt/conda/bin/activate python2 && pip install git+https://github.com/nanograv/PINT.git"

USER root
RUN groupadd -g 1001 pulsar
RUN usermod -g nanograv-data -a -G users,pulsar jovyan
USER jovyan
RUN mkdir /home/jovyan/.ssh
COPY sshkey /home/jovyan/.ssh/sshkey
COPY sshkey.pub /home/jovyan/.ssh/sshkey.pub
COPY ssh_client_config /home/jovyan/.ssh/config
COPY known_hosts /home/jovyan/.ssh/known_hosts
USER root
RUN chown -R jovyan /home/jovyan/.ssh
RUN chmod 700 /home/jovyan/.ssh
RUN chmod 644 /home/jovyan/.ssh/sshkey.pub
RUN chmod 600 /home/jovyan/.ssh/sshkey
RUN chmod 644 /home/jovyan/.ssh/known_hosts
USER jovyan
COPY tai2tt_bipm2016.clk /opt/pulsar/share/tempo2/clock/tai2tt_bipm2016.clk
COPY tai2tt_bipm2017.clk /opt/pulsar/share/tempo2/clock/tai2tt_bipm2017.clk
COPY ao2gps.clk /opt/pulsar/share/tempo2/clock/ao2gps.clk
COPY gbt2gps.clk /opt/pulsar/share/tempo2/clock/gbt2gps.clk
COPY ncyobs2obspm.clk /opt/pulsar/share/tempo2/clock/ncyobs2obspm.clk
COPY obspm2gps.clk /opt/pulsar/share/tempo2/clock/obspm2gps.clk
COPY time_nancay.dat /opt/pulsar/tempo/clock/time_nancay.dat
COPY time_gbt.dat /opt/pulsar/tempo/clock/time_gbt.dat
COPY time_ao.dat /opt/pulsar/tempo/clock/time_ao.dat

USER root
RUN apt install -y libmagickwand-dev
RUN /bin/bash -c "source /opt/conda/bin/activate python2 && pip install Wand"
RUN pip install Wand
RUN bash -c "source activate python2 && pip install git+git://github.com/telegraphic/PyGSM.git"
RUN bash -c "pip install git+git://github.com/telegraphic/PyGSM.git"
RUN apt-get install -y nano vim emacs less nvidia-cuda-dev
RUN bash -c "source activate python2 && pip install pycuda"
RUN pip install pycuda
USER jovyan
RUN git clone https://github.com/demorest/tempo_utils.git && \
    cd tempo_utils && \
    python setup.py install && \
    /bin/bash -c "source /opt/conda/bin/activate python2 && python setup.py install"
RUN /bin/bash -c "source /opt/conda/bin/activate python2 && conda install -y scikit-sparse"
COPY policy.xml /etc/ImageMagick-6/policy.xml
WORKDIR /home/jovyan/work
