# docker build -t mfv .
# docker run -it --rm mfv

FROM ubuntu:16.04 as caffe-build-image

ENV DEBIAN_FRONTEND=noninteractive

RUN set -ex \
    && apt-get -qq update \
    && apt-get -qq install wget cmake \
    && apt-get -qq install python-dev python-pip python-numpy \
    # Install dev dependencies of Caffe
    && apt-get -qq install libopenblas-dev libboost-all-dev libopencv-dev libleveldb-dev \
                           libsnappy-dev libhdf5-serial-dev libgflags-dev libgoogle-glog-dev liblmdb-dev \
                           protobuf-compiler libprotobuf-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN set -ex \
    && wget https://github.com/BVLC/caffe/archive/refs/tags/1.0.tar.gz -O caffe-1.0.tar.gz \
    && tar -xf caffe-1.0.tar.gz && rm caffe-1.0.tar.gz && cd caffe-1.0 \
    && mkdir build && cd build \
    && cmake -DCPU_ONLY=1 -DBLAS=open -DCMAKE_INSTALL_PREFIX=/opt/caffe .. \
    && make -j"$(nproc)" && make install



FROM ubuntu:16.04 as caffe-image

COPY --from=caffe-build-image /opt/caffe /opt/caffe

RUN set -ex \
    && apt-get -qq update \
    && apt-get -qq install python python-pip \
    # Install runtime dependencies of Caffe
    && apt-get -qq install libboost-system1.58.0 libboost-thread1.58.0 libboost-python1.58.0 \
                           libgoogle-glog0v5 libopenblas-base libprotobuf9v5 libhdf5-10 liblmdb0 \
                           libleveldb1v5 libopencv-highgui2.4v5 libopencv-imgproc2.4v5 \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    \
    && pip install --no-cache-dir numpy==1.11.0 PyWavelets==0.5.2 decorator==4.2.1 networkx==2.1 Cython==0.19.2 \
                                  pillow==5.1.0 python-dateutil==2.8.2 pytz==2023.3 six==1.16.0 subprocess32==3.5.4 \
                                  backports.functools-lru-cache==1.6.4 cycler==0.10.0 pyparsing==2.2.0 \
    && pip install --no-cache-dir scikit-image==0.13.1 matplotlib==2.2.2 kiwisolver==1.0.1 scipy==0.17.0 protobuf==3.5.2.post1

ENV \
    CAFFE_ROOT=/opt/caffe \
    PYTHONPATH=$PYTHONPATH:/opt/caffe/python

RUN set -ex \
    && echo "$CAFFE_ROOT/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig \
    && mkdir /workspace

WORKDIR /workspace


FROM caffe-image

RUN set -ex \
    && apt-get -qq update \
    && apt-get -qq install wget git \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    \
    && pip install --no-cache-dir backports.shutil-get-terminal-size==1.0.0 enum34==1.1.10 ipython==5.6.0 \
                                  ipython-genutils==0.2.0 pathlib2==2.3.7.post1 pexpect==4.8.0 pickleshare==0.7.5 \
                                  prompt-toolkit==1.0.18 ptyprocess==0.7.0 Pygments==2.2.0 scandir==1.10.0 \
                                  simplegeneric==0.8.1 traitlets==4.3.2 typing==3.10.0.0 wcwidth==0.2.6

RUN set -ex \
    && git clone https://github.com/Evolving-AI-Lab/mfv.git . \
    && wget http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel \
    && echo "model_path = './bvlc_reference_caffenet.caffemodel'\nmodel_definition   = './deploy_alexnet_updated.prototxt'\ncaffe_root = '/opt/caffe/python'\ngpu = False" > settings.py

