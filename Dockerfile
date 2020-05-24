FROM python:3.7-slim-buster

RUN apt-get update \
  && apt-get install -qq libglib2.0-dev libsm-dev libxrender-dev libxext-dev \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

COPY ./hill_climbing_study ./hill_climbing_study
COPY ./requirements.txt ./
COPY ./run.sh ./

RUN pip install -r requirements.txt
RUN mkdir -p /var/artifacts
RUN mkdir -p /var/file_outputs

ENV ARTIFACTS_DESTINATION=/var/artifacts
ENV FILE_OUTPUTS_DESTINATION=/var/file_outputs

ENTRYPOINT ["./run.sh"]
