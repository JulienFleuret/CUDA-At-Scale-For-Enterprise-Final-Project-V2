#include "magnitude.h"
#include "utils.h"

#include <npp.h>
#include <cuda_runtime.h>

#include <iostream>

// Convinient because some arithmetic functions on half type
// are only available for achitecture 530 and above.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
#define SUPPORTED_HALF_ARITHM 1
#include <cuda_fp16.h>
#else
#define SUPPORTED_HALF_ARITHM 0
#endif


using namespace cv;

namespace cas
{

namespace
{

// Constant memory variables
__constant__ int d_rows;
__constant__ int d_cols;
__constant__ int d_channels;
__constant__ int d_batchSize;
__constant__ int d_dXStep[3];
__constant__ int d_dYStep[3];
__constant__ int d_MagStep[3];


///
/// \brief convert_u8f32_ac4 : convert the input from unsigned char to float.
/// \param src : variable to convert.
/// \return conversion of the input.
///
template<class DstType, class SrcType>
__device__ __forceinline__ DstType convert_to(const SrcType& src)
{
    static_assert(vectorTraits<SrcType>::channels == vectorTraits<DstType>::channels);

    if constexpr (vectorTraits<SrcType>::channels == 1)
    {
#if SUPPORTED_HALF_ARITHM
        if constexpr (std::is_same<DstType, half>())
        {
            return static_cast<DstType>(static_cast<unsigned short>(src));
        }
        else
        {
#endif
            return static_cast<DstType>(src);
#if SUPPORTED_HALF_ARITHM
        }
#endif
    }
    else
    {
        using lane_type = typename vectorTraits<DstType>::lane_type;

        DstType ret;

        ret.x = static_cast<lane_type>(src.x);
        ret.y = static_cast<lane_type>(src.y);
        ret.z = static_cast<lane_type>(src.z);

        if constexpr (vectorTraits<DstType>::channels == 4)
        {
            ret.w = static_cast<lane_type>(src.w);
        }

        return ret;
    }
}



///
/// \brief magnitude : Computes the square root of the sum of the squares of x and y
/// \param x : Sobel horizontal derivative value.
/// \param y : Sobel vertictal derivative value.
/// \return sqrt(x^2 + y^2).
///
template<class T>
__device__ __forceinline__ T magnitude(const T& x, const T& y)
{
    if constexpr (std::is_same<T, float>())
    {
        return __fsqrt_rn(__fmaf_rn(x, x, __fmul_rn(y, y)));
    }
    else if constexpr (std::is_same<T, float3>() || std::is_same<T, float4>())
    {
        T ret;

        ret.x = __fsqrt_rn(__fmaf_rn(x.x, x.x, __fmul_rn(y.x, y.x)));
        ret.y = __fsqrt_rn(__fmaf_rn(x.y, x.y, __fmul_rn(y.y, y.y)));
        ret.z = __fsqrt_rn(__fmaf_rn(x.z, x.z, __fmul_rn(y.z, y.z)));

        if constexpr(std::is_same<T, float4>())
        {
            ret.w = __fsqrt_rn(__fmaf_rn(x.w, x.w, __fmul_rn(y.w, y.w)));
        }

        return ret;
    }
#if SUPPORTED_HALF_ARITHM
    else if constexpr (std::is_same<T, half>())
    {
        return hsqrt(__hfma(x, x, __hmul(y, y)));
    }
#endif
}

///
/// \brief apply_saturation_u8 : contraint the range of the input to be between [0-255]
/// \param src : variable to process.
/// \return processed variable.
///
template<class T>
__device__ __forceinline__ T apply_saturation_u8(const T& src)
{
    if constexpr (std::is_same<T, float>())
    {
        return std::clamp(src, 0.f, 255.f);
    }
    else
    {
        T ret;

        ret.x = std::clamp(src.x, 0.f, 255.f);
        ret.y = std::clamp(src.y, 0.f, 255.f);
        ret.z = std::clamp(src.z, 0.f, 255.f);

        if constexpr(std::is_same<T, float4>())
        {
            ret.w = std::clamp(src.w, 0.f, 255.f);
        }

        return ret;
    }
}



///
/// \brief cvtToGray : convert the vectorized variable given as input to
/// float, using the BGR 2 gray, psychovisual formula.
/// \param src : variable to process.
/// \return processed variable.
///
__device__ __forceinline__ float cvtToGray(const float3& src)
{
    // x -> b
    // y -> g
    // z -> r

    // gray = 0.21 r + 0.72 g + 0.07 b

    return __fma_rn(0.21f, src.z, __fma_rn(0.75f, src.y, __fmul_rn(0.07f, src.x) ) );
}

///
/// \brief load : load a element from a pointer.
/// \param src : pointer
/// \return fundamental or vector type to process.
///
template<int cn, bool single_outout, class DstType = std::conditional_t<single_outout && (cn == 4), uchar3, std::conditional_t<cn == 1, Npp8u, make_vector_type_t<Npp8u, cn> > > >
__device__ __forceinline__ DstType load(const Npp8u* src)
{
    using vector_type = DstType;
    using const_pointer_type = const vector_type*;

    if constexpr (single_outout && cn == 4)
    {
        vector_type tmp = *reinterpret_cast<const_pointer_type>(src);

        return make_uchar3(tmp.x, tmp.y, tmp.z);
    }
    else
    {
        return *reinterpret_cast<const_pointer_type>(src);
    }

}

///
/// \brief store : store the value of the magnitude into the destination pointer.
/// \param dst : address of the first element of the destination pointer.
/// \param value : element to store.
///
template<int cn, bool single_output, class T>
__device__ __forceinline__ void store(Npp8u* dst, const T& value)
{

    using type = std::conditional_t<single_output, Npp8u, make_vector_type_t<Npp8u, cn> >;
    using pointer = type*;

    type tmp;

    if constexpr( (cn>=3) && single_output)
    {
        tmp = convert_to<type>(apply_saturation_u8(cvtToGray(value) ) );
    }
    else
    {
        tmp = convert_to<type>( apply_saturation_u8(value) );
    }


    *reinterpret_cast<pointer>(dst) = tmp;
}

///
/// \brief compute_magnitude : Computes the square root of the sum of the squares of dX and dY and store it in Mag.
/// \param dX : address of the first element of the result of the Sobel horizontal derivative.
/// \param dY : address of the first element of the result of the Sobel vertical derivative.
/// \param Mag : sqrt(x^2 + y^2).
///
template<int cn, bool single_output>
__device__ __forceinline__ void compute_magnitude(const Npp8u* __restrict__ dX, const Npp8u* __restrict__ dY, Npp8u* __restrict__ Mag)
{
    using pointer = std::conditional_t<single_output && cn == 4, uchar3*, std::conditional_t<cn == 1, Npp8u*, make_vector_type_t<Npp8u, cn>* > >;
    using working_type = std::conditional_t<cn == 1, float, make_vector_type_t<float, single_output ? 3 : cn> >;
    using type = typename std::pointer_traits<pointer>::element_type;

    working_type dXf = convert_to<working_type>(load<cn, single_output>(dX));
    working_type dYf = convert_to<working_type>(load<cn, single_output>(dY));

    working_type mag = magnitude(dXf, dYf);

    store<cn, single_output>(Mag, mag);
}

///
/// \brief k_mag : kernel to compute the magnitude (aka hypothenus) of the elements of a batch.
/// \param dX : Sobel horizontal derivative.
/// \param dY : Sobel vertical derivative.
/// \param Mag : magnitude
///
template<int cn, bool single_output> // cn -> channels.
__global__ void k_mag(const Npp8u* __restrict__ dX, const Npp8u* __restrict__ dY, Npp8u* __restrict__ Mag)
{
    // Current location.
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int z = blockIdx.z;

    if((x<d_cols) && (y<d_rows) && (z<d_batchSize) )
    {
        // Set the pointer to the current element.
        dX += z * d_dXStep[0] + y * d_dXStep[1] + x * d_dXStep[2];
        dY += z * d_dYStep[0] + y * d_dYStep[1] + x * d_dYStep[2];
        Mag += z * d_MagStep[0] + y * d_MagStep[1] + x * d_MagStep[2];


        compute_magnitude<cn, single_output>(dX, dY, Mag);
    }
}

} // anonymous

///
/// \brief compute_magnitude_by_batch_t : default constructor
/// Initialize the attributes to their default values.
///
compute_magnitude_by_batch_t::compute_magnitude_by_batch_t():
    was_init(false),
    rows(0),
    cols(0),
    channels(0),
    nb_frames(0),
    force_vectorize(false),
    batch_size(0),
    nb_batches(0),
    device(),
    dX(),
    dY(),
    Mag(),
    offset({0,0}),
    blocks(32, 8, 1),
    grid(),
    stream1(),
    stream2(),
    stream3(),
    sobel_vert(nullptr),
    sobel_horz(nullptr)
{}


///
/// \brief init : initialize the attributes of the current object.
/// \param input : input video file, video stream, of image folder formating (i.e. img%02d.png)
/// \param output : output filename, or image folder formating (i.e. img%02d.jpg)
/// \param _batch_size : batch size to use during the computation, -1 means use default calculation.
///
void compute_magnitude_by_batch_t::init(const String &input, const String &output, const int &_batch_size)
{
    if(!this->was_init)
    {
        // Initialization of the input streaming object.
        this->load.open(input);

        // Initialization of the attributes related to the image and stream.
        this->nb_frames = static_cast<int>(this->load.get(CAP_PROP_FRAME_COUNT)); // Number of frames.
        this->rows = static_cast<int>(this->load.get(CAP_PROP_FRAME_HEIGHT)); // Number of rows.
        this->cols = static_cast<int>(this->load.get(CAP_PROP_FRAME_WIDTH)); // Number of columns.
        int total = rows * cols; // Total number of pixels.
        double fps = this->load.get(CAP_PROP_FPS); // Frame per second (for the output stream).
        this->channels = 0; // Number of channels.
        bool do_size_safety_check = input.find("%") != cv::String::npos; // Is the input a filestream or a


        // Read the first frame, and reset the frame counter to 0.
        {
            Mat tmp;

            this->load>>tmp;

            this->channels = tmp.channels();

            // If a recursive image formating was requested
            // it is important to ensure that all the images
            // have the same size.
            if(do_size_safety_check)
            {
                Size sz0 = tmp.size();

                for(int i=1;i<this->nb_frames;++i)
                {
                    this->load>>tmp;

                    if(tmp.size() != sz0)
                    {
                        std::clog<<"All the images in the folder must have the same size"<<std::endl;
                        std::exit(EXIT_FAILURE);
                    }
                }
            }

            // reset the frame counter to 0.
            this->load.set(CAP_PROP_POS_FRAMES, 0.);
        }

        // Initialization of the output streaming object.
        if(output.find("%") != String::npos)
        {
            this->save.open(output, 0 , fps, Size(cols, rows), false);
        }
        else
        {
            this->save.open(output, VideoWriter::fourcc('m','p','4','v') , fps, Size(this->cols, this->rows), false);
        }

        // If the images of the input have a single channels
        // and a number of columns that is modulus 3 or 4,
        // is possible to divide the number of columns by
        // the number of channels and process the image like
        // if it has 3 or 4 channels. Doing so is likely to
        // to improve the vectorization for the computation
        // of the Sobel filters.
        this->force_vectorize = (this->channels == 1) && ( (!(this->cols%4)) || (!(this->cols%3)));

        if(this->force_vectorize)
        {
            this->channels = !(this->cols%4) ? 4 : 3;
            this->cols /= this->channels;
        }

        // If no batch size were provided during the initialization
        // a default batch size is computed here.
        this->batch_size = _batch_size;

        if(this->batch_size < 0)
        {

            if(total <= 307200) // VGA -> 640 x 480
            {
                this->batch_size = 1024;
            }
            else if(total <= 2073600) // Full HD -> 1920 x 1080
            {
                this->batch_size = 256;
            }
            else if(total <= 8294400) // 4K -> 3840 x 2160
            {
                this->batch_size = 64;
            }
            else if(total <= 33177600) // 8K -> 7680 x 4320
            {
                this->batch_size = 16;
            }
            else if(total <= 132710400) // 16K -> 15360 x 8640
            {
                this->batch_size = 4;
            }
            else
            {
                this->batch_size = 1;
            }
        }

        this->nb_batches = div_up(this->nb_frames, this->batch_size);


        // Prepare the buffers.
        // Note that: dX, dY and Mag buffers assumes that
        // all the images of the batch are concatenated
        // verticaly.
        this->device.create(rows, cols, channels);
        this->dX.create(rows * batch_size, cols, channels);
        this->dY.create(rows * batch_size, cols, channels);
        this->Mag.create(rows * batch_size, cols);

        // For all the three following variables, the first argument
        // represents the number of bytes to move from an image to the next.
        // The second number is the number of bytes for a given image, to
        // add to pass from a row to the next (aka as pitch or step).
        // The last number is the number of bytes to add for a given row
        // to pass from an element to the next.
        int dXStep[] = {dX.pitch() * this->rows, dX.pitch(), channels};
        int dYStep[] = {dY.pitch() * this->rows, dY.pitch(), channels};
        int MagStep[] = {Mag.pitch() * this->rows, Mag.pitch(), this->force_vectorize ? channels : 1};



        // Prepare the grid and blocks.
        this->grid = dim3(div_up(this->cols, this->blocks.x), div_up(this->rows, this->blocks.y), this->batch_size );

        // Set the constant memory variables.
        check_cuda_error_or_npp_status(cudaMemcpyToSymbol(d_rows, &this->rows, sizeof(int)));
        check_cuda_error_or_npp_status(cudaMemcpyToSymbol(d_cols, &this->cols, sizeof(int)));
        check_cuda_error_or_npp_status(cudaMemcpyToSymbol(d_channels, &this->channels, sizeof(int)));
        check_cuda_error_or_npp_status(cudaMemcpyToSymbol(d_batchSize, &this->batch_size, sizeof(int)));
        check_cuda_error_or_npp_status(cudaMemcpyToSymbol(d_dXStep, dXStep, 3 * sizeof(int)));
        check_cuda_error_or_npp_status(cudaMemcpyToSymbol(d_dYStep, dYStep, 3 * sizeof(int)));
        check_cuda_error_or_npp_status(cudaMemcpyToSymbol(d_MagStep, MagStep, 3 * sizeof(int)));

        // Process the data.
        this->stream1.create();
        this->stream2.create();
        this->stream3.create();

        // Set the Sobel filter functions.
        if(channels == 1)
        {
            this->sobel_horz = nppiFilterSobelHorizBorder_8u_C1R_Ctx;
            this->sobel_vert = nppiFilterSobelVertBorder_8u_C1R_Ctx;
        }
        else if(channels == 3)
        {
            this->sobel_horz = nppiFilterSobelHorizBorder_8u_C3R_Ctx;
            this->sobel_vert = nppiFilterSobelVertBorder_8u_C3R_Ctx;
        }
        else
        {
            if(this->force_vectorize)
            {
                this->sobel_horz = nppiFilterSobelHorizBorder_8u_C4R_Ctx;
                this->sobel_vert = nppiFilterSobelVertBorder_8u_C4R_Ctx;
            }
            else
            {
                this->sobel_horz = nppiFilterSobelHorizBorder_8u_AC4R_Ctx;
                this->sobel_vert = nppiFilterSobelVertBorder_8u_AC4R_Ctx;
            }
        }

        this->was_init = true;
    }
}

///
/// \brief is_init : investigate if the current object was initialized or not.
/// \return true if the initialization was successfull, false otherwise.
///
bool compute_magnitude_by_batch_t::is_init() const
{
    return this->was_init;
}


///
/// \brief run : execute the computation of the magnitude.
///
void compute_magnitude_by_batch_t::run()
{
    if(this->channels == 1)
    {
        this->process<1>();
    }
    else if (this->channels == 3)
    {
        if(this->force_vectorize)
        {
            this->process<3, true>();
        }
        else
        {
            this->process<3>();
        }

    }
    else
    {
        if(this->force_vectorize)
        {
            this->process<4, true>();
        }
        else
        {
            this->process<4>();
        }
    }
}

///
/// \brief compute_magnitude_by_batch_t::process
/// process a video by batch. For a given batch
/// the directional Sobel filter are apply on each
/// image one by one. Once all the images of the batch
/// have been filtered, then the magnitude is compile
/// for all the images of the batch at once.
///
template<int cn, bool force_vectorization>
void compute_magnitude_by_batch_t::process()
{

    for(int i=0, j_start=0, j_end = this->batch_size; i<this->nb_batches; i++, j_start+=this->batch_size, j_end+=this->batch_size)
    {
        // Catch
        for(int j=j_start, k=0; j<std::min(j_end, this->nb_frames); ++j, ++k)
        {
            cv::Mat host;

            // grab an image.
            this->load>>host;

            // upload the image on the device.
            this->device.copyFrom(host.ptr(), host.step, this->rows, this->cols, this->channels);

            // ----------------------------- Apply The Vertical Filter -----------------------------

            nppSetStream(static_cast<cudaStream_t>(this->stream1));

            nppGetStreamContext(&this->context);

            check_cuda_error_or_npp_status(this->sobel_vert(this->device.ptr(), this->device.pitch(), this->device.size(), this->offset, this->dY.ptr(k * this->rows), this->dY.pitch(), this->device.size(), NPP_BORDER_REPLICATE, this->context));


            // ----------------------------- Apply The Vertical Filter -----------------------------

            nppSetStream(static_cast<cudaStream_t>(this->stream2));

            nppGetStreamContext(&this->context);

            check_cuda_error_or_npp_status(this->sobel_horz(this->device.ptr(), this->device.pitch(), this->device.size(), this->offset, this->dX.ptr(k * this->rows), this->dX.pitch(), this->device.size(), NPP_BORDER_REPLICATE, this->context));

        }

        // Synchronize the streams to ensure that all the filtering
        // operations are over before starting to process the magnitude.
        this->stream1.synchronize();
        this->stream2.synchronize();

        // ----------------------------- Process The Magnitude For The Whole Batch At Once -----------------------------
        if constexpr (force_vectorization)
        {
            check_cuda_error_or_npp_status(cudaFuncSetCacheConfig(k_mag<cn, false>, cudaFuncCachePreferL1));
            k_mag<cn, false><<<this->grid, this->blocks, 0, static_cast<cudaStream_t>(this->stream3)>>>(this->dX.ptr(), this->dY.ptr(), this->Mag.ptr());
            check_cuda_error_or_npp_status(cudaGetLastError());
        }
        else
        {
            check_cuda_error_or_npp_status(cudaFuncSetCacheConfig(k_mag<cn, true>, cudaFuncCachePreferL1));
            k_mag<cn, true><<<this->grid, this->blocks, 0, static_cast<cudaStream_t>(this->stream3)>>>(this->dX.ptr(), this->dY.ptr(), this->Mag.ptr());
            check_cuda_error_or_npp_status(cudaGetLastError());
        }

        // ----------------------------- Save Data -----------------------------
        for(int j=0, k=0; j<std::min(this->batch_size, this->nb_frames); ++j, k+=this->rows)
        {
            Mat tmp(this->rows, this->cols, CV_8UC1);

            // The implicit vectorization means
            // that a single channel image was
            // given either 3 or 4 channels, and
            // its width was divided accordingly.
            if constexpr (force_vectorization)
            {
                tmp.create(this->rows, this->cols * this->channels, CV_8UC1);
            }
            else
            {
                tmp.create(this->rows, this->cols, CV_8UC1);
            }

            // Download an image from the device.
            check_cuda_error_or_npp_status(cudaMemcpy2D(tmp.ptr(), tmp.step, Mag.ptr(k), Mag.pitch(), tmp.cols, tmp.rows, cudaMemcpyDeviceToHost));

            // Save the image.
            this->save << tmp;
        }
    }
}

}
