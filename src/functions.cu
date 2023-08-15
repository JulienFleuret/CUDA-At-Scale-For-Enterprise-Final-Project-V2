#include "functions.cuh"
#include "utils.cuh"

#include <stdexcept>
#include <limits>

namespace cas
{

///
/// \brief spatial_gradient:
/// compute the horizontal and vertical
/// Sobel filters.
/// \param src : image to process.
/// \return a tuple containing the outputs
/// of the horizontal and vertical filtering in that order.
///
template<class T>
std::tuple<nppiMatrix_t<T>, nppiMatrix_t<T> > spatial_gradient(const nppiMatrix_t<T>& src)
{
    static_assert(std::is_same<T, Npp8u>() || std::is_same<T, Npp16s>() || std::is_same<T, Npp32f>(), "Only Npp8u, Npp16s and Npp32f are supported!");

    nppiMatrix_t<T> dX, dY;

    const int rows = src.height();
    const int cols = src.width();

    //Step 1) Create the streams and events.
    safe_stream stream1, stream2;
    safe_event event1, event2;

    stream1.create();
    stream2.create();

    event1.create();
    event2.create();


    NppiPoint offset = {0,0};

    NppStreamContext context;

    // Step 2.a) Compute dx (Stream1).

    dX.create(rows, cols);

    nppSetStream(stream1);

    nppGetStreamContext(&context);

    if constexpr (std::is_same<T, Npp8u>())
    {
        check_cuda_error_or_npp_status(nppiFilterSobelHorizBorder_8u_C1R_Ctx(src.ptr(), src.pitch(), src.size(), offset, dX.ptr(), dX.pitch(), dX.size(), NPP_BORDER_REPLICATE, context));
    }
    else if constexpr(std::is_same<T, Npp16s>())
    {
        check_cuda_error_or_npp_status(nppiFilterSobelHorizBorder_8s_C1R_Ctx(src.ptr(), src.pitch(), src.size(), offset, dX.ptr(), dX.pitch(), dX.size(), NPP_BORDER_REPLICATE, context));
    }
    else
    {
        check_cuda_error_or_npp_status(nppiFilterSobelHorizBorder_32f_C1R_Ctx(src.ptr(), src.pitch(), src.size(), offset, dX.ptr(), dX.pitch(), dX.size(), NPP_BORDER_REPLICATE, context));
    }

    event1.record();


    // Step 2.b) Compute dy (Stream2).

    dY.create(rows, cols);

    nppSetStream(stream2);

    nppGetStreamContext(&context);

    if constexpr (std::is_same<T, Npp8u>())
    {
        check_cuda_error_or_npp_status(nppiFilterSobelVertBorder_8u_C1R_Ctx(src.ptr(), src.pitch(), src.size(), offset, dY.ptr(), dY.pitch(), dX.size(), NPP_BORDER_REPLICATE, context));
    }
    else if constexpr (std::is_same<T, Npp16s>())
    {
        check_cuda_error_or_npp_status(nppiFilterSobelVertBorder_16s_C1R_Ctx(src.ptr(), src.pitch(), src.size(), offset, dY.ptr(), dY.pitch(), dX.size(), NPP_BORDER_REPLICATE, context));
    }
    else
    {
        check_cuda_error_or_npp_status(nppiFilterSobelVertBorder_32f_C1R_Ctx(src.ptr(), src.pitch(), src.size(), offset, dY.ptr(), dY.pitch(), dX.size(), NPP_BORDER_REPLICATE, context));
    }


    event2.record();

    // Wait that both derivatives have been computed, before returning the processed data..
    stream1.waitEvent(event1);
    stream1.waitEvent(event2);

    return std::make_tuple(dX, dY);
}

///
/// \brief magnitude : compute
/// the magnitude (aka hypotenus) of the arguments.
/// \param src : tuple containing the arguments to process.
/// \return magnitude of the arguments, with respect of the saturation.
///
template<class T>
nppiMatrix_t<T> magnitude(const std::tuple<nppiMatrix_t<T>, nppiMatrix_t<T> >& src)
{
    auto& [h, v] = src;

    return magnitude(h, v);
}


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
    const Npp8u* __restrict__ dX_current = dX + y * d_dxStep + x;
    const Npp8u* __restrict__ dY_current = dY + y * d_dyStep + x;
    Npp8u* __restrict__ dst_current = (dst + y * d_dstStep + x);

    // Process.
    *dst_current = fun(*dX_current, *dY_current);
}

///
/// \brief magnitude : compute
/// the magnitude (aka hypotenus) of the arguments.
/// \param horz : result of an horizontal filtering.
/// \param vert : result of an vertical filtering.
/// \return magnitude of the arguments, with respect of the saturation.
///
template<class T>
nppiMatrix_t<T> magnitude(const nppiMatrix_t<T>& horz, const nppiMatrix_t<T>& vert)
{
    assert(horz.width() == vert.width() && horz.height() == vert.height());

    // Create the destination.
    nppiMatrix_t<T> Mag(horz.height(), horz.width());

    // Set the constant memory variables.
    check_cuda_error_or_npp_status(cudaMemcpyToSymbol(d_rows, &rows, sizeof(int)));
    check_cuda_error_or_npp_status(cudaMemcpyToSymbol(d_cols, &cols, sizeof(int)));
    check_cuda_error_or_npp_status(cudaMemcpyToSymbol(d_dstStep, &mag_step, sizeof(int)));
    check_cuda_error_or_npp_status(cudaMemcpyToSymbol(d_dxStep, &dX_step, sizeof(int)));
    check_cuda_error_or_npp_status(cudaMemcpyToSymbol(d_dyStep, &dY_step, sizeof(int)));

    // Lambda to process on the kernel.
    auto mag = [] __device__ (const T& dx, const T& dy)->T
    {
        float dxf = static_cast<float>(dx);
        float dyf = static_cast<float>(dy);

        float ret = std::clamp(std::hypotf(dxf, dyf), std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

        return static_cast<T>(ret);
    };

    // Prepare the grid and blocks.
    dim3 block(32,8);
    dim3 grid(div_up(cols, block.x), div_up(rows, block.y) );

    // Execute the kernel.
    check_cuda_error_or_npp_status(cudaFuncSetCacheConfig(k_mag<__decltype(mag)>, cudaFuncCachePreferL1));
    k_mag<<<grid, block, 0, stream1>>>(mag, dX.ptr(), dY.ptr(), Mag.ptr());
    check_cuda_error_or_npp_status(cudaGetLastError());

    return Mag;
}

///
/// \brief resize_bi_cubic : apply a bi-cubic resizing.
/// \param input : input to process.
/// \param dst_size : destination size to reach.
/// \return interpolated image.
///
template<class T>
nppiMatrix_t<T> resize_bi_cubic(const nppiMatrix_t<T>& input, const NppiSize& dst_size)
{
    static_assert(std::is_same<T, Npp8u>() || std::is_same<T, Npp32f>(),"This Function Only Accepts Npp8u And Npp32f Types!");

    // Allocate the output.
    nppiMatrix_t<T> output(dst_size.height, dst_size.width);

    NppiRect srcRect = {0, 0, input.width(), input.height()};
    NppiRect dstRect = {0, 0, dst_size.width, dst_size.height};

    // Process the interpolation.
    if constexpr (std::is_same<T, Npp8u>())
    {
        check_cuda_error_or_npp_status(nppiResize_8u_C1R(input.ptr(), input.pitch(), input.size(), srcRect, output.ptr(), output.pitch(), dst_size, dstRect, NPPI_INTER_CUBIC));
    }
    else if constexpr (std::is_same<T, Npp16s>())
    {
        check_cuda_error_or_npp_status(nppiResize_16s_C1R(input.ptr(), input.pitch(), input.size(), srcRect, output.ptr(), output.pitch(), dst_size, dstRect, NPPI_INTER_CUBIC));
    }
    else if constexpr (std::is_same<T, Npp16u>())
    {
        check_cuda_error_or_npp_status(nppiResize_16u_C1R(input.ptr(), input.pitch(), input.size(), srcRect, output.ptr(), output.pitch(), dst_size, dstRect, NPPI_INTER_CUBIC));
    }
    else // Npp32f
    {
        check_cuda_error_or_npp_status(nppiResize_32f_C1R(input.ptr(), input.pitch(), input.size(), srcRect, output.ptr(), output.pitch(), dst_size, dstRect, NPPI_INTER_CUBIC));
    }

    return output;
}


///
/// \brief spatial_gradient:
/// compute the horizontal and vertical
/// Sobel filters.
/// \param src : image to process.
/// \return a tuple containing the outputs
/// of the horizontal and vertical filtering in that order.
///
template<class T>
std::tuple<nppiTensor_t<T>, nppiTensor_t<T> > spatial_gradient(const nppiTensor_t<T>& src, const bool& channels_first)
{
    return std::tuple<nppiTensor_t<T>, nppiTensor_t<T> >();
}

///
/// \brief magnitude : compute
/// the magnitude (aka hypotenus) of the arguments.
/// \param src : tuple containing the arguments to process.
/// \return magnitude of the arguments, with respect of the saturation.
///
template<class T>
nppiTensor_t<T> magnitude(const std::tuple<nppiTensor_t<T>, nppiTensor_t<T> >& src, const bool& channels_first)
{
    auto& [h, v] = src;

    return magnitude(h, v);
}

///
/// \brief magnitude : compute
/// the magnitude (aka hypotenus) of the arguments.
/// \param horz : result of an horizontal filtering.
/// \param vert : result of an vertical filtering.
/// \return magnitude of the arguments, with respect of the saturation.
///
template<class T>
nppiTensor_t<T> magnitude(const nppiTensor_t<T>& horz, const nppiTensor_t<T>& vert, const bool& channels_first)
{

}

///
/// \brief resize_bi_cubic : apply a bi-cubic resizing.
/// \param input : input to process.
/// \param dst_size : destination size to reach.
/// \return interpolated image.
///
template<class T>
nppiTensor_t<T> resize_bi_cubic(const nppiTensor_t<T>& input, const NppiSize& dst_size, const bool& channels_first)
{

}


///
/// \brief merge : merge a list of matrices into a tensor.
/// \param matrices : list of matrices to merge.
/// \param channels_first : should the channels dimension be first?
/// \return a tensor
///
template<class T>
nppiTensor_t<T> merge(const std::vector<nppiMatrix_t<T> >& matrices, const mergeOptions &option, const bool& channels_first)
{
    nppiTensor_t<T> ret;

    if(option == mergeOptions::ASSERT_IF_SIZE_NOT_EQUAL)
    {
        // Step 1) check if every matrix has the same size.
        NppiSize sz0 = mat.size();

        for(int i=1; i<matrices.size(); ++i)
        {
            NppiSize szi = matrices.at(i);

            if((sz0.width != szi.width) || (sz0.height != szi.height))
            {
                throw std::runtime_error("An Error Has Occured: The Size Of All The Images Should Be The Same!");
            }
        }

        // Step 2) prepare the outputs.
        ret.create(static_cast<int>(matrices.size()), sz0.height, sz0.width);

        // Step 3) do the copy.
        auto it_mat = matrices.begin();
        int rpitch = ret.dimension(0);
        for(size_t i=0; i<matrices.size(); ++i, ++it_mat)
            check_cuda_error_or_npp_status(cudaMemcpy2D(ret.ptr(i), rpitch, it_mat->ptr(), it_mat->pitch(), sz0.width, sz0.height, cudaMemcpyDeviceToDevice));

    }
    else
    {
        // Find the smalles and larges images.
        NppiSize smallestSrc, smallestDst;

        NppiSize dst_size;

        if (option == mergeOptions::ASSIGN_SMALLEST_SIZE_TO_ALL)
        {
            NppiSize lowest_src = {std::numeric_limits<int>::max(), std::numeric_limits<int>::max()};

            for(const auto& matrix : matrices)
            {
                NppiSize current_size = matrix.size();

                lowest_src.width = std::min(lowest_src.width, current_size.width);
                lowest_src.height = std::min(lowest_src.height, current_size.height);
            }

            dst_size = smallestSrc = lowest_src;
        }
        else
        {
            NppiSize highest_src = {std::numeric_limits<int>::min(), std::numeric_limits<int>::min()};

            smallestSrc = {std::numeric_limits<int>::max(), std::numeric_limits<int>::max()};

            for(const auto& matrix : matrices)
            {
                NppiSize current_size = matrix.size();

                highest_src.width = std::max(lowest_src.width, current_size.width);
                highest_src.height = std::max(lowest_src.height, current_size.height);

                smallestSrc.width = std::min(smallestSrc.width, current_size.width);
                smallestSrc.height = std::max(smallestSrc.height, current_size.height);
            }

            dst_size = highest_src;
        }

        smallestDst = smallestSrc;

        NppiResizeBatchCXR* batch(nullptr);

        try
        {
            check_cuda_error_or_npp_status(cudaMalloc(&batch, matrices.size()));

            auto it = matrices.begin();
            auto it_batch = batch;

            // Prepare the batch structure. Note destination pointer is a temporary buffer.
            for(size_t i=0; i<matrices.size(); ++i, ++it, ++it_batch)
            {
                it_batch->pSrc = it->ptr();
                it_batch->nSrcStep = it->pitch();

                // Why allocating memory? To avoid having issues with the pitch. Yes it is not optimal and memorivorus.
                it_batch->pDst = nppiMalloc_8u_C1(dst_size.width * sizeof(T), dst_size.height, std::addressof(it_batch->nDstStep))

                        // it_batch->pDst = ret.ptr(i);
                        // it_batch->nDstStep = ret.pitch(1);
            }

            NppiRect srcRect = {0, 0, smallestSrc.width, smallestSrc.height};
            NppiRect dstRect = {0, 0, dst_size.width, dst_size.height};

            // Process the batch.
            if constexpr (std::is_same<T, Npp8u>())
            {
                check_cuda_error_or_npp_status(nppiResizeBatch_8u_C1R(smallestSrc, srcRect, smallestDst, dstRect, NPPI_INTER_CUBIC, batch, matrices.size() ));
            }
            else
            {
                check_cuda_error_or_npp_status(nppiResizeBatch_32f_C1R(smallestSrc, srcRect, smallestDst, dstRect, NPPI_INTER_CUBIC, batch, matrices.size() ));
            }

            // Create the output.
            ret.create(static_cast<int>(matrices.size()), dst_size.height, dst_size.width);

            // Fill the output, and release the buffer memory.
            it_batch = batch;
            for(size_t i=0; i<matrices.size(); ++i, ++it_batch)
            {
                check_cuda_error_or_npp_status(cudaMemcpy2D(ret.ptr(i), ret.pitch(i), it_batch->pDst, it_batch->nDstStep, dst_size.width, dst_size.height, cudaMemcpyDeviceToDevice));

                check_cuda_error_or_npp_status(nppiFree(it_batch->pDst));
            }

            check_cuda_error_or_npp_status(cudaFree(batch));
        }
        catch(...)
        {
            // In case of thrown exception, properly
            // deallocate all the memory that was allocated.
            if(batch)
            {
                auto it_batch = batch;
                for(size_t i=0; i<matrices.size(); ++i, ++it_batch)
                {
                    if(it_batch->pDst)
                        check_cuda_error_or_npp_status(nppiFree(it_batch->pDst));
                }

                check_cuda_error_or_npp_status(cudaFree(batch));
            }
        }
    }

    return ret;
}

///
/// \brief merge : merge a list of matrices into a tensor.
/// \param matrices : list of tensors to merge.
/// \return a tensor
///
template<class T>
nppiTensor_t<T> merge(const std::vector<nppiTensor_t<T> >& matrices, const mergeOptions &option)
{

}


} // cas
