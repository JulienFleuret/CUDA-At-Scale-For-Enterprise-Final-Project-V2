# CUDA At Scale For Enterprise Final Project V2
The second project I did for the module "CUDA at Scale for Enterprise" from Coursera.

## Project Description

This project simply computes the magnitude of a Soble filter. The Directional filters are computed using NPPI, while the magnitude is computed thank to a custom lambda provided to a custom kernel.
It create a command line funtion named ```cudaAtScaleV2.exe```.

### Arguments for ```cudaAtScaleV2.exe```:
- `-h`, `-help`, `-usage`, or `-?`: Print help.
- `-input`,`-i`, or `-if` : specify the input which can be:
  - a video file (e.g. smth.avi, anthg.mp4).
  - a video stream (eg. protocol://host:port/script_name?script_params|auth).
  - an image sequence (eg. img_%02d.jpg)
- `-output`,`-o`, or `-of` : specify the input which can be:
  - a video file (e.g. smth.avi, anthg.mp4).
  - an image sequence (eg. img_%02d.jpg)
- `-batch_size`, or `-bs`: (optional, default: -1) batch size to use for the computations.

## Code Organization

```bin/```
This folder will only exist once the project is compiled, and will contains: ```cudaAtScaleV2.exe``` and ```unit_test.exe``` if the unit test was build.

```data/```
This folder contains few images, which originally comes from the data folder of the sample folder of the legacy modules of OpenCV (i.e. from [here](https://github.com/opencv/opencv/tree/4.x/samples/data) )

```src/```
All the files that this project depends on are here.

```test/```
All the files that related to the build of the unit test.

```INSTALL```
This files gives some advise regarding how to install the dependencies of this project (i.e. CUDA, NPPI, OpenCV).

```CMAkeLists.txt```
Configuration file.

```run.sh```
Run this file in order to compile the project.

```Dockerfile```
Docker image.
