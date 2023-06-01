# docker build -t mfv-alexnet-export_temp .
# docker run -it --rm mfv-alexnet-export_temp

#FROM mfv

#RUN set -ex \
#    && pip install --no-cache-dir future torch


FROM mfv-alexnet-export

RUN set -ex \
    && wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt \
    && pip install --no-cache-dir torchvision

COPY alexnet_export.py dog.jpg .

ENTRYPOINT ["python", "alexnet_export.py"]
