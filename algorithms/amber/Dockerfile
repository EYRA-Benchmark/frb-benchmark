# Use NVIDIA Docker image
FROM nvidia/opencl:devel-ubuntu18.04

# Install all necessary system packages
WORKDIR /
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

# Install AMBER
WORKDIR /opt/amber
RUN git clone https://github.com/AA-ALERT/AMBER_setup.git -b development
ENV SOURCE_ROOT="/opt/amber/src"
ENV INSTALL_ROOT="/usr/local"
WORKDIR /opt/amber/AMBER_setup
RUN ./amber.sh install development
ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"

# Run AMBER for the challenge
WORKDIR /amber_run
COPY conf ./conf/
COPY challenge_sprint.sh .
COPY challenge.sh .
COPY get_fil_header.py .
RUN chmod u+x challenge.sh
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["/amber_run/challenge.sh"]
