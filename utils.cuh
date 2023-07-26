#ifndef UTILS_CUH
#define UTILS_CUH 1

#pragma once


#include <npp.h>

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
__host__ __forceinline__ bool isDevicePointer(void* ptr);


}

#endif
