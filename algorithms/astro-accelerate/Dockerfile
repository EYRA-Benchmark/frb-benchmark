#FROM nvidia/cuda:9.2-devel
# segfault when running on ARTS cluster with cuda 9
FROM nvidia/cuda:8.0-devel-ubuntu16.04

# To get rid of "(TERM is not set, so the dialog frontend is not usable.)"
ARG DEBIAN_FRONTEND=noninteractive

# TODO: remove vim
RUN apt-get update \
    && apt-get install -y build-essential git wget python3 python3-pip vim-nox\
    && pip3 install numpy\
    && wget https://github.com/Kitware/CMake/releases/download/v3.15.1/cmake-3.15.1-Linux-x86_64.sh \
    && mkdir /opt/cmake\
    && sh cmake-3.15.1-Linux-x86_64.sh --skip-license --prefix=/opt/cmake\
    && ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake\
    && git clone https://github.com/AstroAccelerateOrg/astro-accelerate.git /opt/astro-accelerate

WORKDIR /opt/astro-accelerate

# only build for 1080Ti = compute 6.1
# this flag is required when using cuda 8, because otherwise it will attempt
# to build for compute 7.0, which is unsupported
RUN mkdir build && cd build && cmake ../ -DCUDA_ARCH=6.1 && make

RUN cp build/astro-accelerate /usr/local/bin/ \
    && cp build/libastroaccelerate.so /usr/local/lib/

RUN useradd -u 1006 -ms /bin/bash frbbench
WORKDIR /opt/frbbench

COPY find_candidates.py .
COPY py_astro_accelerate.py .
COPY get_fil_header.py .
COPY run.sh .

RUN chown -R frbbench /opt/frbbench
USER frbbench

CMD ["./run.sh"]
