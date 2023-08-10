# Define the image name and tag

# Pull the image
sudo docker pull "jf42673/casv2:latest"

# Change the image tag to a more readable one.
sudo docker tag jf42673/casv2:latest cuda-at-scale-for-enterprise-v2

# Activate the image.
sudo docker run -it --runtime=nvidia --shm-size=1g -e NVIDIA_DRIVER_CAPABILITIES=all --rm -v $PWD:$PWD -w $PWD cuda-at-scale-for-enterprise-v2


