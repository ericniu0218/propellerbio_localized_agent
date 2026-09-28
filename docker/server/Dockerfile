FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
# RDKit/Biopython are optional (guarded imports) and slow to build in a
# minimal image; install the core web/serving deps here. Add them to this
# install step once the target image's base has the toolchain for them, or
# install from conda-forge in a real deployment image.
RUN pip install --no-cache-dir \
    fastapi uvicorn[standard] pydantic pyyaml sqlalchemy sse-starlette httpx psycopg2-binary

COPY app ./app
COPY config ./config
COPY contracts ./contracts
COPY data ./data

ENV PYTHONPATH=/app
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
