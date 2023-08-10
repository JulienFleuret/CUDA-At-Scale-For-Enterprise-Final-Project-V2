# CUDA At Scale For Enterprise Final Project V2
The second project I did for the module "CUDA at Scale for Enterprise" from Coursera.

## Project Description

This project simply computes the magnitude of a Soble filter. The Directional filters are computed using NPPI, while the magnitude is computed thank to a custom lambda provided to a custom kernel.
It create a command line funtion named ```cudaAtScaleV2.exe```.

### Arguments for ```cudaAtScaleV2.exe```:
- `-h`, `-help`, `-usage`, or `-?`: Print help.
- `-path`, `-folder`, or `-fd`: (optional, default: empty string) Folder containing the file to open.
- `-filename`, `-f`, `-fn`, `-input_filename`, or `-if`: (required) Filename of the file to open.
- `-device`: (optional, default: `-f`) Which GPU to use for the processing? (If multi-GPU available)

## Code Organization

```bin/```
This folder will only exist once the project is compiled, and will contain a single file: ```cudaAtScaleV2.exe```

```data/```
This folder contains a single image, which originally comes from the data folder of the sample folder of the legacy modules of OpenCV (i.e. from [here](https://github.com/opencv/opencv/tree/4.x/samples/data) )

```src/```
All the files that this project depends on are here.


```INSTALL```
This files gives some advise regarding how to install the dependencies of this project (i.e. CUDA, NPPI, OpenCV).

```CMAkeLists.txt```
Configuration file.

```run.sh```
Run this file in order to compile the project.

```Dockerfile```
Docker image.
