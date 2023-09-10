FROM python:3.9-slim as python-base

ENV PYTHONUNBUFFERED = 1 \
  PORT=8501 \
  PATH = "$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

FROM python-base as dev-base

RUN : \
  && apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y \
  --no-install-recommends \
  curl \
  build-essential \
  libsndfile1 \
  libsndfile1-dev

ENV GET_POETRY_IGNORE_DEPRECATION=1

RUN pip install poetry==1.1.14

ENV PATH="${PATH}:/root/.poetry/bin"

COPY poetry.lock pyproject.toml ./

RUN poetry config virtualenvs.create false
RUN poetry update
RUN poetry install -E tensorflow-m1

#Build production image
FROM python-base as production

COPY --from=dev-base $PYSETUP_PATH $PYSETUP_PATH

COPY . /app
WORKDIR /app

EXPOSE 8501
RUN pip install tensorflow
RUN python -c 'import tensorflow as tf'

CMD gunicorn --workers=4 --bind 0.0.0.0:8501 app:app
