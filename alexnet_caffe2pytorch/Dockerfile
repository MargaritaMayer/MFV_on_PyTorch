# docker build -t mfv-alexnet-export .
# docker run -it --rm mfv-alexnet-export

FROM mfv

COPY * .

RUN set -ex \
    && pip install --no-cache-dir pytorch
