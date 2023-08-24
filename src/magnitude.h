#ifndef MAGNITUDE_H
#define MAGNITUDE_H 1

#pragma once

#include <memory>

#include <opencv2/videoio.hpp>

#include "types.h"

namespace cas
{

///
/// \brief The compute_magnitude_by_batch_t class
/// compute the magnitude of the Sobel filters
/// applied to a set of images. The Sobel filters
/// are applied on each image, while the magnitude
/// for a batch of 1 up to 1024 images as a whole.
///
class compute_magnitude_by_batch_t
{
public:
    using nppiImage_8u_t = nppiImage_t<Npp8u>;
    using self = compute_magnitude_by_batch_t;

    typedef NppStatus(*sobel_function_t)(const Npp8u *, Npp32s, NppiSize, NppiPoint, Npp8u *, Npp32s, NppiSize, NppiBorderType, NppStreamContext);

    ///
    /// \brief create : create an object of the current class.
    /// \return unique pointer on object of this class.
    ///
    __host__ __forceinline__ static std::unique_ptr<self> create()
    {
        return std::unique_ptr<self>(new self);
    }

    compute_magnitude_by_batch_t(const self&) = delete;

    compute_magnitude_by_batch_t(self&&) = default;

    // good practice to always declare it.
    ~compute_magnitude_by_batch_t() = default;

    self& operator=(const self&) = delete;

    self& operator=(self&&) = default;

    ///
    /// \brief init : initialize the attributes of the current object.
    /// \param input : input video file, video stream, of image folder formating (i.e. img%02d.png)
    /// \param output : output filename, or image folder formating (i.e. img%02d.jpg)
    /// \param _batch_size : batch size to use during the computation, -1 means use default calculation.
    ///
    __host__ void init(const cv::String& input, const cv::String& output, const int& _batch_size);

    ///
    /// \brief is_init : investigate if the current object was initialized or not.
    /// \return true if the initialization was successfull, false otherwise.
    ///
    __host__ bool is_init() const;

    ///
    /// \brief run : execute the computation of the magnitude.
    ///
    __host__ void run();


private:

    ///
    /// \brief compute_magnitude_by_batch_t : default constructor
    /// Initialize the attributes to their default values.
    ///
    __host__ compute_magnitude_by_batch_t();


    // Usefull to assess if the object was initialized or not.
    bool was_init;

    // Input Stream, or folder parsing parameters.
    int rows;
    int cols;
    int channels;
    int nb_frames;

    // If the images of the input have a single channels
    // and a number of columns that is modulus 3 or 4,
    // is possible to divide the number of columns by
    // the number of channels and process the image like
    // if it has 3 or 4 channels. Doing so is likely to
    // to improve the vectorization for the computation
    // of the Sobel filters.
    bool force_vectorize;

    // Batch parameters.
    int batch_size;
    int nb_batches;

    // Internal Variables.
    nppiImage_8u_t device;
    nppiImage_8u_t dX;
    nppiImage_8u_t dY;
    nppiImage_8u_t Mag;

    // This variable is used for
    // Sobel filter computation
    // and is set to {0,0}.
    NppiPoint offset;

    // Kernel parameters.
    dim3 blocks;
    dim3 grid;

    // Stream (or image folder) input and output.
    cv::VideoCapture load;
    cv::VideoWriter save;

    // stream1 and stream 2 are used for the
    // computation of the Sobel filter.
    // They are synchronized before calling
    // the computation of the magnitude, which
    // is done on stream3.
    safe_stream stream1, stream2, stream3;

    // Npp context.
    NppStreamContext context;

    // Pointer on the function to call for the
    // computation of the Sobel filters.
    // Note: I used function pointers because
    // it help to keep the code clear.
    sobel_function_t sobel_horz, sobel_vert;


    // Internal computation function.
    template<int cn, bool force_vectorization=false>
    __host__ void process();
};


} // cas

#endif // MAGNITUDE_H
