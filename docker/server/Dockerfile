# The Bedrock agent-pack server (bedrock.server), built from this repository.
#
#   docker build -f docker/server/Dockerfile -t bedrock-server .
#
# Run from the repository root: the build context is the whole repo. Add
# chemistry and sequence tools with --build-arg EXTRAS=server,postgres,recorder,speakers,chem
#
# whisper.cpp (whisper-cli, whisper-vad-speech-segments) and the speech and VAD
# model files are not in this image, exactly as before: mount them and point
# MEETING_MODEL_DIR (and PATH) at them, or the meeting recorder reports itself
# unavailable and everything else runs. See docs/server/DEPLOYMENT.md.
FROM python:3.12-slim

ARG EXTRAS=server,postgres,recorder,speakers

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[${EXTRAS}]"

# Data (database, workspace, recordings) lives on a volume, and nothing may
# download a model at runtime: this server is air-gapped.
ENV BEDROCK_SERVER_DATA_DIR=/var/lib/bedrock \
    HF_HUB_OFFLINE=1
VOLUME /var/lib/bedrock
EXPOSE 8000

CMD ["bedrock-server", "--host", "0.0.0.0", "--port", "8000"]
