#include <iostream>

#include <npp.h>
#include <cuda_runtime.h>

#include <memory>
#include <stdexcept>
#include <type_traits>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/highgui.hpp>

#include "types.cuh"
#include "utils.cuh"

using namespace std;
using namespace cv;
using namespace cas;


// Usefull variables.
__constant__ int d_rows;
__constant__ int d_cols;
__constant__ int d_dstStep;
__constant__ int d_dxStep;
__constant__ int d_dyStep;

// Kernel that allows to play with lambda expressions.
///
/// \brief k_mag : kernel magnitude ... but it could compute anything else depending on the first argumnent.
/// \param fun : lambda expression to execute on the kernel.
/// \param dX : source address of the first element of the horizontal derivative.
/// \param dY : source address of the first element of the vertical derivative.
/// \param dst : address of the first element of the destination.
///
template<class Fun_t>
__global__ void k_mag(Fun_t fun, const Npp8u* __restrict__ dX, const Npp8u*  __restrict__ dY, Npp8u*  __restrict__ dst)
{
    // Current location.
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int x = blockDim.x * blockIdx.x + threadIdx.x;

    // Is the current kernel in the expected grid?
    if((y >= d_rows) || (x >= d_cols) )
        return;

    // If Yes...

    // Retrieve the addressess of the inputs and outputs
    // for the current location.
    const Npp8u* dX_current = dX + y * d_dxStep + x;
    const Npp8u* dY_current = dY + y * d_dyStep + x;
    Npp8u* dst_current = (dst + y * d_dstStep + x);

    // Process.
    *dst_current = fun(*dX_current, *dY_current);
}

int main(int argc, char* argv[])
{
    try
    {
        // Step -2) Parse the inputs. Let the user giving a filename.
        String keys = "{help h usage ? | | print this message}{path | | folder to where to select the images}{filename | | image to process (by default an image will be randomly selected)}{device |0| which gpu to use (in case of multi-gpu computer)}";

        CommandLineParser parser(argc, argv, keys);

        String filename;
        String folder;
        int gpuid(0);

        //I case help is required.
        if(parser.has("help"))
        {
            parser.printMessage();
            return EXIT_SUCCESS;
        }

        // Get the gpuid if specify.
        if(parser.has("device"))
            gpuid = parser.get<int>("device");

        // If the gpuid is set.
        if(gpuid)
        {
            // Step 1) Check the number of GPUs available.
            int nb_devices(0);

            check_cuda_error_or_npp_status(cudaGetDeviceCount(&nb_devices));

            // Step 2) If GPUID in the range of the possible ids, then set the device.
            if((nb_devices > 1) && (gpuid <= nb_devices - 1) )
                cudaSetDevice(gpuid);
        }


        // Get the image folder path, is there is any.
        if(parser.has("path"))
            folder = parser.get<String>("path");

        // Get the filename.
        if(parser.has("filename"))
            filename = parser.get<String>("filename");


        // If the filename is empty, then end the program normaly.
        if(filename.empty())
        {
            std::cout<<"The Filename Is Missing! Please Specify A Filename!"<<std::endl;
            return EXIT_SUCCESS; //Why EXIT_SUCCESS? Because a missing or wrong argument is not a faillure but a misusage.
        }

        // If the folder was set, then added it to the filename.
        if(!folder.empty())
            filename = utils::fs::join(folder, filename);

        // Check if the filename exists, otherwise end the program normaly.
        if(!utils::fs::exists(filename))
        {
            std::cout<<"The Specified Filename: '"<<filename<<"' Does Not Exists!"<<std::endl;
            return EXIT_SUCCESS;
        }

        // Start the program.

        // Step -1) Create the streams and events.
        safe_stream stream1, stream2;
        safe_event event1, event2;

        stream1.create();
        stream2.create();

        event1.create();
        event2.create();

        // Step 0) Read The Image.
        Mat host_image = imread(filename, IMREAD_GRAYSCALE);

        const int rows = host_image.rows;
        const int cols = host_image.cols;

        cv::imshow("source", host_image);

        // Step 1) Upload The Image On The Device.
        nppiMatrix_t<Npp8u> device_image(rows, cols), dX, dY, Mag;

        check_cuda_error_or_npp_status(cudaMemcpy2D(device_image.ptr(), device_image.pitch(), host_image.ptr(), host_image.step, host_image.cols * host_image.elemSize(), host_image.rows, cudaMemcpyHostToDevice));

        NppiPoint anchor = {0,0};
        NppiPoint offset = {0,0};

        NppStreamContext context;

        // Step 2.a) Compute dx (Stream1).

        dX.create(rows, cols);

        nppSetStream(stream1);

        nppGetStreamContext(&context);

        check_cuda_error_or_npp_status(nppiFilterSobelHorizBorder_8u_C1R_Ctx(device_image.ptr(), device_image.pitch(), device_image.size(), offset, dX.ptr(), dX.pitch(), dX.size(), NPP_BORDER_REPLICATE, context));

        event1.record();


        // Step 2.b) Compute dy (Stream2).

        dY.create(rows, cols);

        nppSetStream(stream2);

        nppGetStreamContext(&context);

        check_cuda_error_or_npp_status(nppiFilterSobelVertBorder_8u_C1R_Ctx(device_image.ptr(), device_image.pitch(), device_image.size(), offset, dY.ptr(), dY.pitch(), dX.size(), NPP_BORDER_REPLICATE, context));

        event2.record();


        // Step 3 Compute Magnitude.

        Mag.create(rows, cols);

        // But first wait that both derivatives have been computed.
        stream1.waitEvent(event1);
        stream1.waitEvent(event2);

        nppSetStream(stream1);

        nppGetStreamContext(&context);

        Npp32s mag_step = Mag.pitch();
        Npp32s dX_step = dX.pitch();
        Npp32s dY_step = dY.pitch();

        // Set the constant memory variables.
        check_cuda_error_or_npp_status(cudaMemcpyToSymbol(d_rows, &rows, sizeof(int)));
        check_cuda_error_or_npp_status(cudaMemcpyToSymbol(d_cols, &cols, sizeof(int)));
        check_cuda_error_or_npp_status(cudaMemcpyToSymbol(d_dstStep, &mag_step, sizeof(int)));
        check_cuda_error_or_npp_status(cudaMemcpyToSymbol(d_dxStep, &dX_step, sizeof(int)));
        check_cuda_error_or_npp_status(cudaMemcpyToSymbol(d_dyStep, &dY_step, sizeof(int)));

        // Lambda to process on the kernel.
        auto mag = [=] __device__ (const unsigned char& dx, const unsigned char& dy)->unsigned char
        {
            float dxf = static_cast<float>(dx);
            float dyf = static_cast<float>(dy);

            float ret = std::clamp(std::hypotf(dxf, dyf), 0.f, 255.f);

            return static_cast<unsigned char>(ret);
        };

        // Prepare the grid and blocks.
        dim3 block(32,8);
        dim3 grid(div_up(cols, block.x), div_up(rows, block.y) );

        // Execute the kernel.
        check_cuda_error_or_npp_status(cudaFuncSetCacheConfig(k_mag<__decltype(mag)>, cudaFuncCachePreferL1));
        k_mag<<<grid, block, 0, stream1>>>(mag, dX.ptr(), dY.ptr(), Mag.ptr());
        check_cuda_error_or_npp_status(cudaGetLastError());

        device_image.release();
        dX.release();
        dY.release();

        // Step 4) Visualization.

        Mat1b host_mag(rows, cols);

        check_cuda_error_or_npp_status(cudaMemcpy2D(host_mag.ptr(), host_mag.step, Mag.ptr(), Mag.pitch(), cols, rows, cudaMemcpyDeviceToHost));


        cv::imshow("Magnitude", host_mag);

        cv::waitKey(-1);

    }
    catch(std::exception& err)
    {
        std::clog<<"An Error Has Occured: "<<err.what()<<std::endl;
        return EXIT_FAILURE;
    }

    cout << "Hello World!" << endl;
    return EXIT_SUCCESS;
}