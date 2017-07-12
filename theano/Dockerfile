FROM nvidia/cuda:7.5-cudnn5-devel

MAINTAINER Nikki Aldeborgh <nikki.aldeborgh@digitalglobe.com>

RUN apt-get -y update && apt-get -y \
    install python \
    build-essential \
    libopencv-dev \
    python-opencv \
    python-software-properties \
    software-properties-common \
    ipython \
    python-pip \
    python-scipy \
    python-numpy \
    python-dev \
    python-six \
    libatlas-base-dev \
    gfortran \
    libyaml-dev \
    libhdf4-alt-dev \
    libhdf5-serial-dev \
    vim \
    wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN pip install h5py geojson sklearn keras==1.2.2
RUN pip install --upgrade six
RUN pip install Theano==0.9.0

COPY deploy-chip-classifier-theano.py /
COPY gbdx_task_interface.py /
COPY keras.json /root/.keras/keras.json
COPY .theanorc /root/.theanorc

