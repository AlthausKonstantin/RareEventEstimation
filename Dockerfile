# https://github.com/docker-library/docs/tree/master/python#how-to-use-this-image
# build it: docker build --pull --rm -f "Dockerfile" -t rareeventestimation:latest "."
# enter docker shell, bind ./local_dir to /app/: docker run -it --entrypoint /bin/sh -v  "$(pwd)"/local_dir: /app/ rareeventestimation


FROM python:3

WORKDIR ./

COPY . .


RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt \
    pip install -e ./
RUN ["apt-get", "update"]
RUN ["apt-get", "install", "-y", "vim"]

