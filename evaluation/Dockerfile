FROM python:3.7
WORKDIR /frb-eval
COPY ./Pipfile .
COPY ./Pipfile.lock .

RUN pip install pipenv && \
    pipenv install --system --ignore-pipfile --deploy

COPY frb_eyra_analysis frb_eyra_analysis
COPY setup.py .
RUN pip install .    

ENTRYPOINT ["sh", "-c"]
CMD ["python frb_eyra_analysis/blind_detection.py /data/input/ground_truth --fn_cand_files=/data/input/implementation_output > /data/output"]
