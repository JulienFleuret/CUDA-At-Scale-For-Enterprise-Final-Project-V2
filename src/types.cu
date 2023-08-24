#include "types.h"
#include "utils.h"

#include <iostream>

#define DEF_CLASS_SPEC(name)\
    template class name<Npp8u>;\
    template class name<Npp8s>;\
    template class name<Npp16u>;\
    template class name<Npp16s>;\
    template class name<Npp32u>;\
    template class name<Npp32s>;\
    template class name<Npp64u>;\
    template class name<Npp64s>;\
    template class name<Npp32f>;\
    template class name<Npp64f>;

namespace cas
{

/////////////////////////////////////////////////////
/// SAFE STREAM
////////////////////////////////////////////////////

///
/// \brief safe_stream : default constructor.
/// Initialize the attribute "stream" to nullptr.
///
safe_stream::safe_stream():
    stream(nullptr)
{}

///
/// \brief safe_stream : parametric constructor.
/// \param _stream : stream to own.
/// The stream provided as argument is own
/// by the object, and will be destroy either
/// by calling the method "destroy" of by
/// the destructor.
///
safe_stream::safe_stream(const cudaStream_t& _stream):
    stream(_stream)
{}

///
/// \brief ~safe_stream : destructor
/// If a stream was created, it will be
/// destroy.
///
safe_stream::~safe_stream()
{
    this->destroy();

}

///
/// \brief create : create a new stream.
/// \param flags : attributes of the stream to set.
/// \param priority : priority to set for the stream.
///
void safe_stream::create(const unsigned int& flags, const int& priority)
{
    // If a stream already exists
    // it must be destroied before
    // continuing.
    if(this->stream)
        this->destroy();

    if(flags == cudaStreamDefault && priority<0)
    {
        check_cuda_error_or_npp_status(cudaStreamCreate(std::addressof(this->stream)));
    }
    else if(flags != cudaStreamDefault && priority<0)
    {
        check_cuda_error_or_npp_status(cudaStreamCreateWithFlags(std::addressof(this->stream), flags));
    }
    else if(flags == cudaStreamDefault && priority>=0)
    {
        check_cuda_error_or_npp_status(cudaStreamCreateWithPriority(std::addressof(this->stream), cudaStreamDefault, priority));
    }
    else
    {
        check_cuda_error_or_npp_status(cudaStreamCreateWithPriority(std::addressof(this->stream), flags, priority));
    }
}

///
/// \brief destroy : destroy the current
/// stream, if it was created. Otherwise
/// do nothing.
///
void safe_stream::destroy()
{
    if(this->stream)
    {
        check_cuda_error_or_npp_status(cudaStreamDestroy(this->stream));
        this->stream = nullptr;
    }
}

///
/// \brief waitEvent : wait for an event to finish.
/// \param event : event to monitor.
/// \param flags : attributes to set.
///
safe_stream::operator cudaStream_t() const
{
    return this->stream;
}



///
/// \brief synchronize : wait for the current stream to have finish its task.
///
void safe_stream::synchronize()
{
    check_cuda_error_or_npp_status(cudaStreamSynchronize(this->stream));
}


/////////////////////////////////////////////////////
/// SIMPLE IMAGE TYPE
////////////////////////////////////////////////////



///
/// \brief nppiImage_t : default contructor.
/// initialize the attributes to their default
/// value.
///
template<class T>
nppiImage_t<T>::nppiImage_t():
    data(nullptr),
    rows(0),
    cols(0),
    cns(0),
    step(0)
{}

///
/// \brief nppiImage_t : copy constructor.
/// This constructor does not make a deep copy.
/// The attributes of the current objet are set
/// to those of the provided object, and the reference
/// counter is incremented.
/// \param obj : object to reference.
///
template<class T>
nppiImage_t<T>::nppiImage_t(const nppiImage_t& obj):
    data(obj.data),
    rows(obj.rows),
    cols(obj.cols),
    cns(obj.cns),
    step(obj.step),
    counter(obj.counter)
{
    if(this->counter)
        ++(*this->counter);
}

template<class T>
nppiImage_t<T>::~nppiImage_t()
{
    this->release();
}

///
/// \brief operator = assignation operator.
/// If the current object is initialize the method
/// release is called before assigning the attributes
/// to those of object provided as argument.
/// The reference counter is then incremented.
/// \note this operator does not do a deep copy
/// \return a reference on the current object.
///
template<class T>
nppiImage_t<T>& nppiImage_t<T>::operator=(const nppiImage_t& obj)
{
    if(this != std::addressof(obj))
    {
        this->release();

        this->data    = obj.data;
        this->rows    = obj.rows;
        this->cols    = obj.cols;
        this->cns     = obj.cns;
        this->step    = obj.step;
        this->counter = obj.counter;

        if(this->counter)
            ++(*this->counter);
    }

    return (*this);
}

///
/// \brief channels
/// \return number of channels.
///
template<class T>
int nppiImage_t<T>::channels() const
{
    return this->cns;
}

///
/// \brief create : allocate memory
/// \param rows : number of rows.
/// \param cols : number of columns.
/// \param channels : number of channels.
///
template<class T>
void nppiImage_t<T>::create(const Npp32s& _rows, const Npp32s& _cols, const Npp32s& channels)
{
    this->release();

    this->rows = _rows;
    this->cols = _cols;
    this->cns = channels;
    this->data = nppiMalloc_8u_C1(this->cols * channels * sizeof(T), this->rows, &this->step);
    this->counter.reset(new int (1));
}

///
/// \brief create : allocate memory, convinient overload of the previous method.
/// \param size : width and height to allocate.
/// \param channels : number of channels.
///
template<class T>
void nppiImage_t<T>::create(const NppiSize &size, const Npp32s &channels)
{
    this->create(size.height, size.width, channels);
}

///
/// \brief release : deallocate memory if
/// memory was allocated and the decrement of the reference
/// counter reach 0.
///
template<class T>
void nppiImage_t<T>::release()
{
    // If the counter was initialized and its decrement reach 0,
    // it is time to deallocate the memory.
    if(this->counter && !(--(*this->counter)))
        nppiFree(this->data);

    // reset attribute to their default values.
    this->data = nullptr;
    this->rows = this->cols = this->cns = this->step = 0;
    this->counter.reset();
}

///
/// \brief copyTo : copy to an external address.
/// \param pDst : address of the first destination element.
/// \param nDstStep : number of bytes to move from one row to another, from the destinaiton address.
/// \param _rows : number of rows of the destination.
/// \param _cols : number of columns of the destination.
/// \note the external pointer can be on the device or on the host.
///
template<class T>
void nppiImage_t<T>::copyTo(pointer pDst, const Npp32s& nDstStep, const Npp32s& _rows, const Npp32s& _cols, const Npp32s& _channels) const
{
    check_cuda_error_or_npp_status(cudaMemcpy2D(pDst, nDstStep, this->data, this->step, std::min(_cols, this->cols) * std::min(_channels, this->cns) * sizeof(T), std::min(_rows, this->rows), isDevicePointer(pDst) ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice));
}

///
/// \brief copyFrom : copy the data from an external address.
/// \param pSrc : source pointer to copy from.
/// \param nSrcStep : number of bytes to move from one row to another, from the source address.
/// \param _rows : number of rows of the source.
/// \param _cols : number of column of the source.
/// \note the external pointer can be on the device or on the host.
///
template<class T>
void nppiImage_t<T>::copyFrom(const_pointer pSrc, const Npp32s nSrcStep, const Npp32s& _rows, const Npp32s& _cols, const Npp32s& _channels)
{
    check_cuda_error_or_npp_status(cudaMemcpy2D(this->data, this->step, pSrc, nSrcStep, std::min(_cols, this->cols) * std::min(_channels, this->cns) * sizeof(T), std::min(_rows, this->rows), isDevicePointer(pSrc) ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));
}



///
/// \brief size : accessor.
/// Return a NppiSize object.
/// \return the width and height of the current matrix
/// allocation.
///
template<class T>
NppiSize nppiImage_t<T>::size() const
{
    return {this->cols, this->rows};
}

///
/// \brief width : accessor.
/// \return the width of the current matrix.
///
template<class T>
Npp32s nppiImage_t<T>::width() const
{
    return this->cols;
}

///
/// \brief height : accessor.
/// \return the height of the current matrix.
///
template<class T>
Npp32s nppiImage_t<T>::height() const
{
    return this->rows;
}

///
/// \brief pitch : accessor.
/// Return the line step of the current memory allocation
/// \note The line step ensure that the memory is aligned.
/// \return the line step of the current matrix.
///
template<class T>
Npp32s nppiImage_t<T>::pitch() const
{
    return this->step;
}

///
/// \brief ptr : accessor.
/// return the address of the first element of the specified row.
/// \param y : index of the row to return the address of the first element of.
/// \return address of the first element of the specified specified by the row.
///
template<class T>
typename nppiImage_t<T>::pointer nppiImage_t<T>::ptr(const Npp32s& y)
{
    return reinterpret_cast<pointer>(this->data + y * this->step);
}

///
/// \brief ptr : accessor.
/// return the address of the first element of the specified row.
/// \param y : index of the row to return the address of the first element of.
/// \return address of the first element of the specified specified by the row.
///
template<class T>
typename nppiImage_t<T>::const_pointer nppiImage_t<T>::ptr(const Npp32s& y) const
{
    return reinterpret_cast<const_pointer>(this->data + y * this->step);
}

///
/// \brief ptr : accessor.
/// return the address of the element of the specified row and column.
/// \param y : index of the row.
/// \param x : index of the columns.
/// \return address of the element located that the y^{th} rows and x^{th} rows..
///
template<class T>
typename nppiImage_t<T>::pointer nppiImage_t<T>::ptr(const Npp32s& y, const Npp32s& x)
{
    return this->ptr(y) + (x * this->cns);
}

///
/// \brief ptr : accessor.
/// return the address of the element of the specified row and column.
/// \param y : index of the row.
/// \param x : index of the columns.
/// \return address of the element located that the y^{th} rows and x^{th} rows..
///
template<class T>
typename nppiImage_t<T>::const_pointer nppiImage_t<T>::ptr(const Npp32s& y, const Npp32s& x) const
{
    return this->ptr(y) + (x * this->cns);
}

DEF_CLASS_SPEC(nppiImage_t)

} // cas
