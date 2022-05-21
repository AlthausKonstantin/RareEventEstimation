FROM python:3

WORKDIR ./

COPY . .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt \
    pip install ./dist/rareeventestimation-0.1.0.tar.gz



CMD [ "python", "./docs/benchmarking/cbs-toy-problems.py" ]