#ifndef UTILS_CUH
#define UTILS_CUH 1

#pragma once


#include <npp.h>
#include <vector_types.h>
#include <cuda_fp16.h>


namespace cas
{

///
/// \brief div_up : round-up divison.
/// \param num : numerator.
/// \param den : denominator.
/// \return the rounded up division.
///
__host__ int div_up(const int& num, const int& den);

///
/// \brief nppGetStatusString : return a string
/// correspond to the provided status.
/// \param status
/// \return message corresponding to the string.
///
const char* nppGetStatusString(NppStatus status);


///
/// \brief check_cuda_error_or_npp_status :
/// as its name indicate, check if either
/// a cudaError_t or an NppStatus have are
/// notify a success. If not an std::runtime_error
/// with a message indicating the issue is thrown.
/// \param error_or_status : error or status to assess.
///
template<class Type>
__host__ void check_cuda_error_or_npp_status(const Type& error_or_status);

///
/// \brief isDevicePointer : check if an address is located on the host or on the device.
/// \param ptr : address to access.
/// \return true if the address is located on the host, false otherwise.
///
__host__ bool isDevicePointer(const void* ptr);

///
/// \brief traits class for cuda's vector types.
/// Provide informations such as the number of channels,
/// or the type of a single element.
///
template<class T>
struct vectorTraits;

#define IMPL_VECTOR_TRAITS(type, ftype, cn)\
    template<>\
    struct vectorTraits<type>\
{ \
    constexpr static int channels = cn; \
    using lane_type = ftype; \
};

///
/// \brief convinient macro for specialization declaration and definition.
///
#define IMPL_VECTOR_TRAITS_CN(ftype, vtype)\
    IMPL_VECTOR_TRAITS(ftype     , ftype, 1)\
    IMPL_VECTOR_TRAITS(vtype ## 1, ftype, 1)\
    IMPL_VECTOR_TRAITS(vtype ## 2, ftype, 2)\
    IMPL_VECTOR_TRAITS(vtype ## 3, ftype, 3)\
    IMPL_VECTOR_TRAITS(vtype ## 4, ftype, 4)

IMPL_VECTOR_TRAITS_CN(unsigned char, uchar)
IMPL_VECTOR_TRAITS_CN(signed char, char)
IMPL_VECTOR_TRAITS_CN(unsigned short, ushort)
IMPL_VECTOR_TRAITS_CN(short, short)
IMPL_VECTOR_TRAITS_CN(unsigned int, uint)
IMPL_VECTOR_TRAITS_CN(int, int)
IMPL_VECTOR_TRAITS_CN(unsigned long, ulong)
IMPL_VECTOR_TRAITS_CN(long, long)
IMPL_VECTOR_TRAITS_CN(unsigned long long, ulonglong)
IMPL_VECTOR_TRAITS_CN(long long, longlong)
IMPL_VECTOR_TRAITS_CN(float, float)
IMPL_VECTOR_TRAITS_CN(double, double)

IMPL_VECTOR_TRAITS(half, half, 1)
IMPL_VECTOR_TRAITS(half2, half2, 2)

#undef IMPL_VECTOR_TRAITS_CN
#undef IMPL_VECTOR_TRAITS

///
/// \brief convinient class to call cuda's vector types
/// based on the fundamental type and the number of channels.
///
template<class T, int cn>
struct make_vector_type;

///
/// \brief convinient macro for specialization declaration and definition.
///
#define IMPL_MAKE_VECTOR_TYPE(ftype, cn, vtype)\
    template<> struct make_vector_type<ftype, cn>{ using type=vtype; };

#define IMPL_MAKE_VECTOR_TYPE_CN(ftype, vtype)\
    IMPL_MAKE_VECTOR_TYPE(ftype, 1, vtype ## 1) \
    IMPL_MAKE_VECTOR_TYPE(ftype, 2, vtype ## 2) \
    IMPL_MAKE_VECTOR_TYPE(ftype, 3, vtype ## 3) \
    IMPL_MAKE_VECTOR_TYPE(ftype, 4, vtype ## 4)

IMPL_MAKE_VECTOR_TYPE_CN(unsigned char, uchar)
IMPL_MAKE_VECTOR_TYPE_CN(signed char, char)
IMPL_MAKE_VECTOR_TYPE_CN(unsigned short, ushort)
IMPL_MAKE_VECTOR_TYPE_CN(short, short)
IMPL_MAKE_VECTOR_TYPE_CN(unsigned int, uint)
IMPL_MAKE_VECTOR_TYPE_CN(int, int)
IMPL_MAKE_VECTOR_TYPE_CN(unsigned long, ulong)
IMPL_MAKE_VECTOR_TYPE_CN(long, long)
IMPL_MAKE_VECTOR_TYPE_CN(unsigned long long, ulonglong)
IMPL_MAKE_VECTOR_TYPE_CN(long long, longlong)
IMPL_MAKE_VECTOR_TYPE_CN(float, float)
IMPL_MAKE_VECTOR_TYPE_CN(double, double)

#undef IMPL_MAKE_VECTOR_TYPE_CN
#undef IMPL_MAKE_VECTOR_TYPE


template<class T, int cn>
using make_vector_type_t = typename make_vector_type<T, cn>::type;

} // namespace cas




#endif
