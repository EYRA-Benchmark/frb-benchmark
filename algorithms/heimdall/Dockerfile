FROM nvidia/cuda:10.1-base

RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository -y ppa:kernsuite/kern-4
RUN add-apt-repository -y multiverse
RUN add-apt-repository -y restricted
RUN apt-get update
RUN apt-get install -y heimdall-astro python

COPY run.sh /
COPY get_fil_header.py /

RUN chmod a+x /run.sh

CMD ["/run.sh"]
# heimdall -f /data/simpulse_nfrb20_DM35-1977_214sec_20190603-0837.fil
