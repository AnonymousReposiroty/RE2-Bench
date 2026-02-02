FROM python:3.12
WORKDIR /re2bench

COPY ./requirements.txt /re2bench/requirements.txt

RUN apt-get update


RUN pip3 install -r /re2bench/requirements.txt

COPY ./dataset /re2bench/dataset
COPY ./results /re2bench/results
COPY ./scripts /re2bench/scripts
COPY ./prompts /re2bench/prompts

