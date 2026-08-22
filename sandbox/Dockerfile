FROM python:3.11-slim

RUN pip install --no-cache-dir pandas

RUN useradd --uid 1000 --create-home sandbox
USER sandbox

WORKDIR /workspace
