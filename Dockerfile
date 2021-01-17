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
  python3-pip

# RUN yum group install -y "Development Tools"

# RUN wget https://cmake.org/files/v3.18/cmake-3.18.0.tar.gz \
# && tar -xvzf cmake-3.18.0.tar.gz \
# && cd cmake-3.18.0 \
# && ./bootstrap \
# && make \
# && make install 

# Install opencv and tesseract stuff
# Include global arg in this stage of the build
ARG FUNCTION_DIR
# Create function directory
RUN mkdir -p ${FUNCTION_DIR}

# Copy function code
COPY app/ ${FUNCTION_DIR}

# WORKDIR /
# ENV OPENCV_VERSION="4.1.1"
# RUN wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip \
# && unzip ${OPENCV_VERSION}.zip \
# && mkdir /opencv-${OPENCV_VERSION}/cmake_binary \
# && cd /opencv-${OPENCV_VERSION}/cmake_binary \
# && cmake -DBUILD_TIFF=ON \
#   -DBUILD_opencv_java=OFF \
#   -DWITH_CUDA=OFF \
#   -DWITH_OPENGL=ON \
#   -DWITH_OPENCL=ON \
#   -DWITH_IPP=ON \
#   -DWITH_TBB=ON \
#   -DWITH_EIGEN=ON \
#   -DWITH_V4L=ON \
#   -DBUILD_TESTS=OFF \
#   -DBUILD_PERF_TESTS=OFF \
#   -DCMAKE_BUILD_TYPE=RELEASE \
#   -DPYBIND11_NOPYTHON=ON \
#   -DCMAKE_INSTALL_PREFIX=$(python3.7 -c "import sys; print(sys.prefix)") \
#   -DPYTHON_EXECUTABLE=$(which python3.7) \
#   -DPYTHON_INCLUDE_DIR=$(python3.7 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
#   -DPYTHON_PACKAGES_PATH=$(python3.7 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
#   -DPYTHON_LIBRARY=$(python3.7 -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
#   .. \
# && make install 
# && rm /${OPENCV_VERSION}.zip \
# && rm -r /opencv-${OPENCV_VERSION}

# create symbolic link for compiled opencv
# RUN ln -s \
#   /usr/local/python/cv2/python-3.7/cv2.cpython-37m-x86_64-linux-gnu.so \
#   /usr/local/lib/python3.7/site-packages/cv2.so

WORKDIR ${FUNCTION_DIR}
# Copy requirements.txt into docker image
COPY ./requirements.txt ./ 
RUN pip3 install --upgrade pip
RUN pip3 install pillow
RUN pip3 install tesseract-ocr
RUN pip3 install pytesseract
RUN pip3 install opencv-contrib-python
# install dependencies 
RUN pip3 install -r requirements.txt


# RUN yum install -y \
#     poppler-utils \
#     tesseract-ocr \
#     libtesseract-dev \
#     libleptonica-dev \
#     ldconfig \
#     libsm6 \
#     libxext6 \
#     python-opencv

# # Include global arg in this stage of the build
# ARG FUNCTION_DIR
# # Create function directory
# RUN mkdir -p ${FUNCTION_DIR}

# # Copy function code
# COPY app/ ${FUNCTION_DIR}

# Install the runtime interface client
RUN pip3 install awslambdaric

# Include global arg in this stage of the build
# ARG FUNCTION_DIR
# Set working directory to function root directory
# WORKDIR ${FUNCTION_DIR}


# Copy in the build image dependencies
# COPY --from=build-image ${FUNCTION_DIR} ${FUNCTION_DIR}

# ENTRYPOINT [ "/var/lang/bin/python", "-m", "awslambdaric" ]
ENTRYPOINT [ "/usr/bin/python3", "-m", "awslambdaric" ]
# ENTRYPOINT [ "/bin/bash" ]
# ENTRYPOINT [ "python3"]
CMD [ "app.handler" ]