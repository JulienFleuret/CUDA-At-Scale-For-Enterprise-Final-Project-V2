# Install any required dependencies and libraries using package manager (apt, yum, etc.)
RUN apt-get update && \
    apt-get install -y libopencv-core-dev libopencv-imgcodecs-dev
