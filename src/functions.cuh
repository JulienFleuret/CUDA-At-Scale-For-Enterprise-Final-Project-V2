#ifndef FUNCTIONS_CUH
#define FUNCTIONS_CUH 1

#pragma once

#include "types.cuh"

#include <tuple>
#include <vector>

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
std::tuple<nppiMatrix_t<T>, nppiMatrix_t<T> > spatial_gradient(const nppiMatrix_t<T>& src);

///
/// \brief magnitude : compute
/// the magnitude (aka hypotenus) of the arguments.
/// \param src : tuple containing the arguments to process.
/// \return magnitude of the arguments, with respect of the saturation.
///
template<class T>
nppiMatrix_t<T> magnitude(const std::tuple<nppiMatrix_t<T>, nppiMatrix_t<T> >& src);

///
/// \brief magnitude : compute
/// the magnitude (aka hypotenus) of the arguments.
/// \param horz : result of an horizontal filtering.
/// \param vert : result of an vertical filtering.
/// \return magnitude of the arguments, with respect of the saturation.
///
template<class T>
nppiMatrix_t<T> magnitude(const nppiMatrix_t<T>& horz, const nppiMatrix_t<T>& vert);

///
/// \brief resize_bi_cubic : apply a bi-cubic resizing.
/// \param input : input to process.
/// \param dst_size : destination size to reach.
/// \return interpolated image.
///
template<class T>
nppiMatrix_t<T> resize_bi_cubic(const nppiMatrix_t<T>& input, const NppiSize& dst_size);


///
/// \brief seprable_filter
/// \param src : tensor to process.
/// \param k_horz : horizontal kernels to apply to the tensors.
/// \param k_vert : vertical kernels to apply to the tensors.
/// \return processed tensor.
///
//template<class T>
//nppiTensor_t<T> seprable_filter(const nppiTensor_t<T>& src, const nppiTensor_t<T>& k_horz, const nppiTensor_t<T>& k_vert);

class SeparableFilter
{
public:

    using tensor_type_f = nppiTensor_t<Npp32f>;

    ~SeparableFilter() = default;

    __host__ bool try_init();

    __host__ void clear();

    template<class... Args>
    __host__ __forceinline__ void update_kernels(const std::vector<Npp32s>& src_dimensions, const Args&... kernels)
    {
        this->update_kernels(src_dimensions, {kernels...});
    }

    __host__ void update_kernels(const std::vector<Npp32s>& src_dimensions, const std::vector<tensor_type_f >& _kernels);

    template<class T>
    __host__ void apply(const nppiTensor_t<T>& src, nppiTensor_t<T>& dst);



    template<class... Args>
    static __host__ __forceinline__ std::unique_ptr<SeparableFilter> create(const std::vector<Npp32s>& src_dimensions, const Args&... kernels)
    {
        return create(src_dimensions, {kernels...});
    }

    static __host__ std::unique_ptr<SeparableFilter> create(const std::vector<Npp32s>& src_dimensions, const std::vector<tensor_type_f>& kernels);

private:

    // Transaction in the sense of a single unit of work, i.e. the application of a kernel.
    struct transaction_t;

    __host__ SeparableFilter();

    __host__ SeparableFilter(const std::vector<Npp32s>& src_dimensions, const std::vector<nppiTensor_t<Npp32f> >& kernels);

    __host__ nppiTensor_t<Npp32f> applyOneFilter(transaction_t& obj);

    bool init;
    std::vector<Npp32s> dimensions;
    std::vector<tensor_type_f > kernels;
    std::vector<tensor_type_f > buffers;
    std::vector<std::vector<int> > src_offsets;
    std::vector<std::vector<int> > dst_offsets;
};



///
/// \brief spatial_gradient:
/// compute the horizontal and vertical
/// Sobel filters.
/// \param src : image to process.
/// \return a tuple containing the outputs
/// of the horizontal and vertical filtering in that order.
///
template<class T>
std::tuple<nppiTensor_t<T>, nppiTensor_t<T> > spatial_gradient(const nppiTensor_t<T>& src, const bool& channels_first=true);

///
/// \brief magnitude : compute
/// the magnitude (aka hypotenus) of the arguments.
/// \param src : tuple containing the arguments to process.
/// \return magnitude of the arguments, with respect of the saturation.
///
template<class T>
nppiTensor_t<T> magnitude(const std::tuple<nppiTensor_t<T>, nppiTensor_t<T> >& src, const bool& channels_first=true);

///
/// \brief magnitude : compute
/// the magnitude (aka hypotenus) of the arguments.
/// \param horz : result of an horizontal filtering.
/// \param vert : result of an vertical filtering.
/// \return magnitude of the arguments, with respect of the saturation.
///
template<class T>
nppiTensor_t<T> magnitude(const nppiTensor_t<T>& horz, const nppiTensor_t<T>& vert, const bool& channels_first=true);

///
/// \brief resize_bi_cubic : apply a bi-cubic resizing.
/// \param input : input to process.
/// \param dst_size : destination size to reach.
/// \return interpolated image.
///
template<class T>
nppiTensor_t<T> resize_bi_cubic(const nppiTensor_t<T>& input, const NppiSize& dst_size, const bool& channels_first=true);


enum class mergeOptions
{
    ASSIGN_SMALLEST_SIZE_TO_ALL,
    ASSIGN_BIGEST_SIZE_TO_ALL,
    ASSERT_IF_SIZE_NOT_EQUAL
};

///
/// \brief merge : merge a list of matrices into a tensor.
/// \param matrices : list of matrices to merge.
/// \param option : what to do if the size of the inputs are not the same.
/// \param channels_first : should the channels dimension be first?
/// \return a tensor
///
template<class T>
nppiTensor_t<T> merge(const std::vector<nppiMatrix_t<T> >& matrices, const mergeOptions& option, const bool& channels_first=true);

///
/// \brief merge : merge a list of matrices into a tensor.
/// \param tensors : list of tensors to merge.
/// \param option : what to do if the size of the inputs are not the same.
/// \return a tensor
///
template<class T>
nppiTensor_t<T> merge(const std::vector<nppiTensor_t<T> >& matrices, const mergeOptions& option);


} // cas

#endif // FUNCTIONS_CUH
