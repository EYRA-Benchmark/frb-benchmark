# frb-benchmark
This repository contains Dockerfiles for running algorithms for the [Fast Radio Burst detection benchmark](https://www.eyrabenchmark.net/benchmark/4fcec5b8-40ad-4ca7-a663-c4f96c52bd19) on the [Eyra Benchmark Platform](https://www.eyrabenchmark.net) and the code for the evaluation of these algorithms.

## Interface
The algorithms are provided as Docker image and implement a specific interface.
Each Docker image will read data from the file `/data/input/test_data` (a filterbank file, containing injected FRBs - for details and examples, see [the benchmark details](https://www.eyrabenchmark.net/benchmark/4fcec5b8-40ad-4ca7-a663-c4f96c52bd19)), and should write detected FRB candidates to a the file `/data/output`. One line per candidate, with the following fields (separated with a space character): `<DM> <S/N> <TIME> <LOG_2_DOWNSAMPLE> <FREQ_REF>`.
These output files are then evaluated by the `evaluation` algorithm, which expects the algorithms output at `/data/input/implementation_output`, and the actual injected FRBs at `/data/input/ground_truth`. It will then output a JSON file containing totals (detected/missed/false positives) to `/data/output`.

## Running manually
Most of the algorithms in this repository require a GPU to run. This also requires some [additional Docker configuration](https://github.com/NVIDIA/nvidia-docker). The algorithms have been tested on an AWS `p3.2xlarge` (with a single NVIDIA Tesla V100 GPU) machine, running Ubuntu 18.04 and prepared using the [setup.sh](setup.sh) file.
To build a the Docker image for an algorithm, clone this repository and run the following command for example:
- `docker build -t frb-heimdall:1 algorithms/heimdall`.
- To make the data available to the Docker container it's best to prepare a data directory (e.g. `/data`), which will be 'mounted' into the container. `mkdir /data && mkdir /data/input && cp <filterbank_file> /data/input/test_data`.
- Then run the container, while mounting the data folder: `docker run --rm -it --gpus all -v /data:/data frb-heimdall:1`. After completion, output should be at `/data/output`.
- Next you could run the evaluation:
```
docker build -t frb-evaluation:1 evaluation
mv /data/output /data/implementation_output
cp <ground_truth_file> /data/ground_truth
docker run --rm -it -v /data:/data frb-evaluation:1
```
Again, output should end up in `/data/output`.
