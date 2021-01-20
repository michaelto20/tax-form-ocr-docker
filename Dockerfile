# Define function directory
ARG FUNCTION_DIR="/app/"

FROM public.ecr.aws/lts/ubuntu:18.04_stable

ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install aws-lambda-cpp build dependencies and openjdk-11-jre
RUN apt-get update && \
  apt-get install -y \
  python-dev \
  libssl-dev \
  wget \
  g++ \
  make \
  unzip \
  libcurl4-openssl-dev \
  openjdk-11-jre \
  yasm \
  pkg-config \
  libswscale-dev \
  libtbb2 \
  libtbb-dev \
  libjpeg-dev \
  libpng-dev \
  libtiff-dev \
  libavformat-dev \
  libpq-dev 

RUN apt-get install -y \
  poppler-utils \
  tesseract-ocr \
  libtesseract-dev \
  libleptonica-dev \
  # ldconfig \
  glibc-source \
  libsm6 \
  libxext6 \
  python-opencv \
  python3-pip \
  nano
  # libzbar0 \

# Include global arg in this stage of the build
ARG FUNCTION_DIR
# Create function directory
RUN mkdir -p ${FUNCTION_DIR}

# Copy function code
COPY app/ ${FUNCTION_DIR}

WORKDIR ${FUNCTION_DIR}
# Copy requirements.txt into docker image
COPY ./requirements.txt ./ 
RUN pip3 install --upgrade pip
RUN pip3 install pillow
RUN pip3 install tesseract-ocr
RUN pip3 install pytesseract
RUN pip3 install opencv-contrib-python
# RUN pip3 install pyzbar
# install dependencies 
RUN pip3 install -r requirements.txt



# Install the runtime interface client
RUN pip3 install awslambdaric

ENTRYPOINT [ "/usr/bin/python3", "-m", "awslambdaric" ]
# ENTRYPOINT [ "/bin/bash" ]
# ENTRYPOINT [ "python3"]
CMD [ "app.handler" ]