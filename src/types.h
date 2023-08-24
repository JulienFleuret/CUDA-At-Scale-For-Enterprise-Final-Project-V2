#ifndef TYPES_CUH
#define TYPES_CUH 1

#pragma once

#include <npp.h>

#include <memory>


namespace cas // cuda at scale
{

/////////////////////////////////////////////////////
/// SAFE STREAM
////////////////////////////////////////////////////

///
/// \brief The safe_event class
/// The goal of this class is to
/// manage a cudaStream_t. Why
/// do one need this class?
/// Simply it allows to properly
/// deallocate the GPU resoureces
/// in case of an unexpected event.
/// Yes the GPU resource can be reset
/// at the begining of a program, or
/// using nividia-smi. But isn't
/// better to manage them in a safeway?
///
class safe_stream
{
private:
    cudaStream_t stream;

public:

    ///
    /// \brief safe_stream : default constructor.
    /// Initialize the attribute "stream" to nullptr.
    ///
    __host__ safe_stream();

    ///
    /// \brief safe_stream : parametric constructor.
    /// \param _stream : stream to own.
    /// The stream provided as argument is own
    /// by the object, and will be destroy either
    /// by calling the method "destroy" of by
    /// the destructor.
    ///
    __host__ safe_stream(const cudaStream_t& _stream);

    ///
    /// \note For the copy constructor to make
    /// sense it would require a reference counter.
    ///
    safe_stream(const safe_stream&) = delete;

    ///
    /// \brief safe_stream : move constructor.
    /// \param : object to move.
    ///
    safe_stream(safe_stream&&) = default; //no __host__ required, see warning #20012-D "__host__ annotation is ignored on a function ... that is explicitly defaulted on its first declaration"

    ///
    /// \brief ~safe_stream : destructor
    /// If a stream was created, it will be
    /// destroy.
    ///
    __host__ ~safe_stream();

    ///
    /// \brief For the assignation operator to make sense
    /// it would require a reference counter.
    ///
    safe_stream& operator=(const safe_stream&) = delete;

    ///
    /// \brief move operator : swap the attribute
    /// between the current and the argument object.
    /// \return the current object
    ///
    safe_stream& operator=(safe_stream&&) = default;

    ///
    /// \brief create : create a new stream.
    /// \param flags : attributes of the stream to set.
    /// \param priority : priority to set for the stream.
    ///
    __host__ void create(const unsigned int& flags = cudaStreamDefault, const int& priority=-1);

    ///
    /// \brief destroy : destroy the current
    /// stream, if it was created. Otherwise
    /// do nothing.
    ///
    __host__ void destroy();

    ///
    /// \brief synchronize : wait for the current stream to have finish its task.
    ///
    __host__ void synchronize();

    ///
    /// \brief operator cudaStream_t : implicit conversion operator.
    /// Convinient to maintain compatibility with the rest of the
    /// CUDA API.
    ///
    __host__ explicit operator cudaStream_t() const;

};

/////////////////////////////////////////////////////
/// Image Type
////////////////////////////////////////////////////

///
/// \brief The nppiImage_t class
/// Image class, represent an image.
/// Inspired by OpenCV's Mat, UMat and GpuMat
/// object this class does not make copy by assignation
/// neither by the copy constructor. The least methods
/// for the purpose of this project were implemented.
///
template<class T>
class nppiImage_t
{
public:


    using pointer = T*;
    using const_pointer = const T*;

    ///
    /// \brief nppiImage_t : default contructor.
    /// initialize the attributes to their default
    /// value.
    ///
    __host__ nppiImage_t();

    ///
    /// \brief nppiImage_t : copy constructor.
    /// This constructor does not make a deep copy.
    /// The attributes of the current objet are set
    /// to those of the provided object, and the reference
    /// counter is incremented.
    /// \param obj : object to reference.
    ///
    __host__ nppiImage_t(const nppiImage_t& obj);


    nppiImage_t(nppiImage_t&&) = default;

    ///
    /// \brief destructor : deallocate any allocated memory.
    ///
    ~nppiImage_t();

    ///
    /// \brief operator = assignation operator.
    /// If the current object is initialize the method
    /// release is called before assigning the attributes
    /// to those of object provided as argument.
    /// The reference counter is then incremented.
    /// \note this operator does not do a deep copy
    /// \return a reference on the current object.
    ///
    __host__ nppiImage_t& operator = (const nppiImage_t&);

    nppiImage_t& operator = (nppiImage_t&&) = default;

    ///
    /// \brief channels
    /// \return number of channels.
    ///
    __host__ int channels() const;


    ///
    /// \brief create : allocate memory
    /// \param rows : number of rows.
    /// \param cols : number of columns.
    /// \param channels : number of channels.
    ///
    __host__ void create(const Npp32s &rows, const Npp32s &cols, const Npp32s &channels=1);

    ///
    /// \brief create : allocate memory, convinient overload of the previous method.
    /// \param size : width and height to allocate.
    /// \param channels : number of channels.
    ///
    __host__ void create(const NppiSize& size, const Npp32s& channels=1);

    ///
    /// \brief release : deallocate memory if
    /// memory was allocated and the decrement of the reference
    /// counter reach 0. The attributes will be reset to their
    /// default value, after each call.
    ///
    __host__ void release();

    ///
    /// \brief copyTo : copy to an external address.
    /// \param pDst : address of the first destination element.
    /// \param nDstStep : number of bytes to move from one row to another, from the destinaiton address.
    /// \param _rows : number of rows of the destination.
    /// \param _cols : number of columns of the destination.
    /// \note the external pointer can be on the device or on the host.
    ///
    __host__ void copyTo(pointer pDst, const Npp32s& nDstStep, const Npp32s& _rows, const Npp32s& _cols, const Npp32s& _channels = 1) const;


    ///
    /// \brief copyFrom : copy the data from an external address.
    /// \param pSrc : source pointer to copy from.
    /// \param nSrcStep : number of bytes to move from one row to another, from the source address.
    /// \param _rows : number of rows of the source.
    /// \param _cols : number of column of the source.
    /// \note the external pointer can be on the device or on the host.
    ///
    __host__ void copyFrom(const_pointer pSrc, const Npp32s nSrcStep, const Npp32s& _rows, const Npp32s& _cols, const Npp32s& _channels = 1);

    ///
    /// \brief size : accessor.
    /// Return a NppiSize object.
    /// \return the width and height of the current matrix
    /// allocation.
    ///
    __host__ NppiSize size() const;

    ///
    /// \brief width : accessor.
    /// \return the width of the current matrix.
    ///
    __host__ Npp32s width() const;

    ///
    /// \brief height : accessor.
    /// \return the height of the current matrix.
    ///
    __host__ Npp32s height() const;

    ///
    /// \brief pitch : accessor.
    /// Return the line step of the current memory allocation
    /// \note The line step ensure that the memory is aligned.
    /// \return the line step of the current matrix.
    ///
    __host__ Npp32s pitch() const;

    ///
    /// \brief ptr : accessor.
    /// return the address of the first element of the specified row.
    /// \param y : index of the row to return the address of the first element of.
    /// \return address of the first element of the specified specified by the row.
    ///
    __host__ pointer ptr(const Npp32s& y=0);

    ///
    /// \brief ptr : accessor.
    /// return the address of the first element of the specified row.
    /// \param y : index of the row to return the address of the first element of.
    /// \return address of the first element of the specified specified by the row.
    ///
    __host__ const_pointer ptr(const Npp32s& y=0) const;

    ///
    /// \brief ptr : accessor.
    /// return the address of the element of the specified row and column.
    /// \param y : index of the row.
    /// \param x : index of the columns.
    /// \return address of the element located that the y^{th} rows and x^{th} rows..
    ///
    __host__ pointer ptr(const Npp32s& y, const Npp32s& x);

    ///
    /// \brief ptr : accessor.
    /// return the address of the element of the specified row and column.
    /// \param y : index of the row.
    /// \param x : index of the columns.
    /// \return address of the element located that the y^{th} rows and x^{th} rows..
    ///
    __host__ const_pointer ptr(const Npp32s& y, const Npp32s& x) const;

private:

    Npp8u* data;
    Npp32s rows, cols, cns, step;
    std::shared_ptr<Npp32s> counter;



};






} // cas

#endif // TYPES_CUH
