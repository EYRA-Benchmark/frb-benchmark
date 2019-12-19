FROM nvidia/cuda:9.2-devel

RUN apt-get update \
    && apt-get install -y build-essential git wget\
    && wget https://github.com/Kitware/CMake/releases/download/v3.15.1/cmake-3.15.1-Linux-x86_64.sh \
    && mkdir /opt/cmake\
    && sh cmake-3.15.1-Linux-x86_64.sh --skip-license --prefix=/opt/cmake\
    && ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake\
    && git clone https://github.com/AstroAccelerateOrg/astro-accelerate.git /opt/astro-accelerate

WORKDIR /opt/astro-accelerate

RUN mkdir build && cd build && cmake ../ && make

FROM nvidia/cuda:9.2-runtime
COPY --from=0 /opt/astro-accelerate/build/astro-accelerate /usr/local/bin/
COPY --from=0 /opt/astro-accelerate/build/libastroaccelerate.so /usr/local/lib/

RUN apt-get update && apt-get install -y python3 python3-pip && pip3 install numpy
RUN mkdir /app
WORKDIR /app

COPY find_candidates.py /app/
COPY py_astro_accelerate.py /app/
COPY get_fil_header.py /app/
COPY run.sh /app/

ENTRYPOINT ["/bin/bash", "-c"]

CMD ["/app/run.sh"]
