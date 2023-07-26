#include "utils.cuh"

#include <stdexcept>

namespace cas
{

///
/// \brief div_up : round-up divison.
/// \param num : numerator.
/// \param den : denominator.
/// \return the rounded up division.
///
__host__ int div_up(const int& num, const int& den)
{
    float numf = static_cast<float>(num);
    float denf = static_cast<float>(den);

    return static_cast<int>(std::ceil(numf/denf));
}

///
/// \brief nppGetStatusString
/// \param status
/// \return
///
const char* nppGetStatusString(NppStatus status)
{
    switch(static_cast<int>(status))
    {
        case NPP_NOT_SUPPORTED_MODE_ERROR:
            return "Error: Mode Not Supported";

        case NPP_INVALID_HOST_POINTER_ERROR:
            return "Error: Invalid Host Pointer";

        case NPP_INVALID_DEVICE_POINTER_ERROR:
            return "Error: Invalid Device Pointer";

        case NPP_LUT_PALETTE_BITSIZE_ERROR:
            return "Error: LUT Palette Bitsize";

        case NPP_ZC_MODE_NOT_SUPPORTED_ERROR:
            return "Error: ZC Mode Not Supported";

        case NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY:
            return "Error: Not Sufficient Compute Capability";

        case NPP_TEXTURE_BIND_ERROR:
            return "Error: Texture Bind";

        case NPP_WRONG_INTERSECTION_ROI_ERROR:
            return "Error: Wrong Interesection ROI";

        case NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR:
            return "Error: HAAR Classifier: Pixel Match";

        case NPP_MEMFREE_ERROR:
            return "Error: Memory Free";

        case NPP_MEMSET_ERROR:
            return "Error: Memory Set";

        case NPP_MEMCPY_ERROR:
            return "Error: Memory Copy";

        case NPP_ALIGNMENT_ERROR:
            return "Error: Incorrect Alignment";

        case NPP_CUDA_KERNEL_EXECUTION_ERROR:
            return "Error: Cuda Kernel Execution";

        case NPP_ROUND_MODE_NOT_SUPPORTED_ERROR:
            return "Error: Round Mode Not Supported";

        case NPP_QUALITY_INDEX_ERROR:
            return "Error: Quality Index";

        case NPP_RESIZE_NO_OPERATION_ERROR:
            return "Error: Resize No Operation";

        case NPP_OVERFLOW_ERROR:
            return "Error: Overflow";

        case NPP_NOT_EVEN_STEP_ERROR:
            return "Error: Not Even Step";

        case NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR:
            return "Error: Histogram Number Of Levels";

        case NPP_LUT_NUMBER_OF_LEVELS_ERROR:
            return "Error: LUT Number Of Levels";

        case NPP_CORRUPTED_DATA_ERROR:
            return "Error: Corrupted Data";

        case NPP_CHANNEL_ORDER_ERROR:
            return "Error: Channel Order";

        case NPP_ZERO_MASK_VALUE_ERROR:
            return "Error: Zero Mask Value";

        case NPP_QUADRANGLE_ERROR:
            return "Error: Quadrangle";

        case NPP_RECTANGLE_ERROR:
            return "Error: Rectangle";

        case NPP_COEFFICIENT_ERROR:
            return "Error: Coefficient";

        case NPP_NUMBER_OF_CHANNELS_ERROR:
            return "Error: Number Of Channels";

        case NPP_COI_ERROR:
            return "Error: COI";

        case NPP_DIVISOR_ERROR:
            return "Error: Divisor";

        case NPP_CHANNEL_ERROR:
            return "Error: Channel";

        case NPP_STRIDE_ERROR:
            return "Error: Stride";

        case NPP_ANCHOR_ERROR:
            return "Error: Anchor";

        case NPP_MASK_SIZE_ERROR:
            return "Error: Mask Size";

        case NPP_RESIZE_FACTOR_ERROR:
            return "Error: Resize Factor";

        case NPP_INTERPOLATION_ERROR:
            return "Error: Interpolation";

        case NPP_MIRROR_FLIP_ERROR:
            return "Error: Mirror Flip";

        case NPP_MOMENT_00_ZERO_ERROR:
            return "Error: Moment 00 Zero";

        case NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR:
            return "Error: Threshold Negative Level";

        case NPP_THRESHOLD_ERROR:
            return "Error: Threshold";

        case NPP_CONTEXT_MATCH_ERROR:
            return "Error: Context Match";

        case NPP_FFT_FLAG_ERROR:
            return "Error: FFT Flag";

        case NPP_FFT_ORDER_ERROR:
            return "Error: FFT Order";

        case NPP_STEP_ERROR:
            return "Error: Step";

        case NPP_SCALE_RANGE_ERROR:
            return "Error: Scale Range";

        case NPP_DATA_TYPE_ERROR:
            return "Error: Data Type";

        case NPP_OUT_OFF_RANGE_ERROR:
            return "Error: Out Of Range";

        case NPP_DIVIDE_BY_ZERO_ERROR:
            return "Error: Divide By Zero";

        case NPP_MEMORY_ALLOCATION_ERR:
            return "Error: Memory Allocation";

        case NPP_NULL_POINTER_ERROR:
            return "Error: Null Pointer";

        case NPP_RANGE_ERROR:
            return "Error: Range";

        case NPP_SIZE_ERROR:
            return "Error: Size";

        case NPP_BAD_ARGUMENT_ERROR:
            return "Error: Bad Argument";

        case NPP_NO_MEMORY_ERROR:
            return "Error: No Memory";

        case NPP_NOT_IMPLEMENTED_ERROR:
            return "Error: Not Implemented";

        case NPP_ERROR:
            return  "Error";

        case NPP_ERROR_RESERVED:
            return "Error: Reserved";

        case NPP_NO_ERROR:
            return "Success";

        case NPP_NO_OPERATION_WARNING:
            return "Warning: No Operation";

        case NPP_DIVIDE_BY_ZERO_WARNING:
            return "Warning: Divide By Zero";

        case NPP_AFFINE_QUAD_INCORRECT_WARNING:
            return "Warning: Affine Quad Incorrect";

        case NPP_WRONG_INTERSECTION_ROI_WARNING:
            return "Warning: Wrong Intersection ROI";

        case NPP_WRONG_INTERSECTION_QUAD_WARNING:
            return "Warning: Wrong Intersection Quad";

        case NPP_DOUBLE_SIZE_WARNING:
            return "Warning: Double Size";

        case NPP_MISALIGNED_DST_ROI_WARNING:
            return "Warning: Misaligned Destination ROI";

        default:
            break;
    }

    return "Error: Unknown (The Translation Of This Flag To Text Is not Yet Supported)";
}

///
/// \brief check_cuda_error_or_npp_status
/// \param error_or_status
///
template<class Type>
__host__ void check_cuda_error_or_npp_status(const Type& error_or_status)
{
    if(error_or_status)
    {

        if constexpr (std::is_same<Type, cudaError_t>())
        {
            throw std::runtime_error(cudaGetErrorString(error_or_status));
        }
        else
        {
            throw std::runtime_error(nppGetStatusString(error_or_status));
        }
    }
}

template void check_cuda_error_or_npp_status<cudaError_t>(const cudaError_t&);
template void check_cuda_error_or_npp_status<NppStatus>(const NppStatus&);

} // cas
