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
template<class T>
nppiTensor_t<T> seprable_filter(const nppiTensor_t<T>& src, const nppiTensor_t<T>& k_horz, const nppiTensor_t<T>& k_vert);

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
