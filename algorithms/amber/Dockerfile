# Use NVIDIA Docker image
FROM nvidia/opencl:devel-ubuntu18.04

# Install all necessary system packages
RUN apt-get -qq -y update && apt-get -qq -y install \
    build-essential \
    git \
    cmake \
    libgtest-dev \
    opencl-headers \
    && apt-get clean

# Install Google Test
WORKDIR /usr/src/gtest/build
RUN cmake .. && make
RUN cp *.a /usr/lib

# add frbbench user; UID should be same as arts on ARTS cluster
RUN useradd -u 1006 -ms /bin/bash frbbench

# Install AMBER
WORKDIR /opt/frbbench/amber
RUN git clone https://github.com/AA-ALERT/AMBER_setup.git -b development
ENV SOURCE_ROOT="/opt/amber/src"
ENV INSTALL_ROOT="/usr/local"
WORKDIR /opt/frbbench/amber/AMBER_setup
RUN ./amber.sh install development
ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"

# Run AMBER for the challenge
WORKDIR /opt/frbbench
COPY conf ./conf/
COPY scenarios ./scenarios/
COPY run.sh .
COPY get_fil_header.py .
RUN chmod a+x run.sh
RUN chown -R frbbench /opt/frbbench
USER frbbench
CMD ["./run.sh"]
