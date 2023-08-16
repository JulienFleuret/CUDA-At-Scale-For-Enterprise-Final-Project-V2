#include "types.cuh"
#include "utils.cuh"

#include <numeric>
#include <functional>
#include <cassert>

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
__host__ safe_stream::safe_stream():
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
__host__ safe_stream::safe_stream(const cudaStream_t& _stream):
    stream(_stream)
{}

///
/// \brief ~safe_stream : destructor
/// If a stream was created, it will be
/// destroy.
///
__host__ safe_stream::~safe_stream()
{
    this->destroy();

}

///
/// \brief create : create a new stream.
/// \param flags : attributes of the stream to set.
/// \param priority : priority to set for the stream.
///
__host__ void safe_stream::create(const unsigned int& flags, const int& priority)
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
__host__ void safe_stream::destroy()
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
__host__ safe_stream::operator cudaStream_t() const
{
    return this->stream;
}

///
/// \brief operator cudaStream_t : implicit conversion operator.
/// Convinient to maintain compatibility with the rest of the
/// CUDA API.
///
__host__ void safe_stream::waitEvent(const safe_event& event, const unsigned int& flags)
{
    check_cuda_error_or_npp_status(cudaStreamWaitEvent(this->stream, event, flags));
}


/////////////////////////////////////////////////////
/// SAFE EVENT
////////////////////////////////////////////////////

///
/// \brief safe_event : parametric constructor.
/// \param event : event to own.
/// The stream provided as argument is own
/// by the object, and will be destroy either
/// by calling the method "destroy" of by
/// the destructor.
///
__host__ safe_event::safe_event():
    event(nullptr)
{}


///
/// \brief ~safe_event : destructor
/// If an event was created, it will be
/// destroy.
///
__host__ safe_event::~safe_event()
{
    if(this->event)
        this->destroy();
}

///
/// \brief create : create the event.
/// \param flags : attriutes to apply on the event.
///
__host__ void safe_event::create(const unsigned int& flags)
{
    if(this->event)
        this->destroy();

    if(flags!=cudaEventDefault)
    {
        check_cuda_error_or_npp_status(cudaEventCreate(std::addressof(this->event)));
    }
    else
    {
        check_cuda_error_or_npp_status(cudaEventCreateWithFlags(std::addressof(this->event), flags));
    }
}

///
/// \brief destroy : destroy the current
/// event, if it was created. Otherwise
/// do nothing.
///
__host__ void safe_event::destroy()
{
    if(this->event)
    {
        check_cuda_error_or_npp_status(cudaEventDestroy(this->event));
        this->event = nullptr;
    }
}

///
/// \brief record : record an event
/// \param _stream : stream to record.
/// \param _flags : attributes to apply on the recording.
///
__host__ void safe_event::record(const safe_stream& _stream, const unsigned int& _flags)
{
    if(_flags!=cudaEventDefault)
    {
        check_cuda_error_or_npp_status(cudaEventRecord(this->event, _stream));
    }
    else
    {
        check_cuda_error_or_npp_status(cudaEventRecordWithFlags(this->event, _stream, _flags));
    }
}

///
/// \brief synchonize : waits for an event to complete.
///
__host__ void safe_event::synchonize()
{
    check_cuda_error_or_npp_status(cudaEventSynchronize(this->event));
}

///
/// \brief operator cudaEvent_t : implicit conversion operator.
/// Convinient to maintain compatibility with the rest of the
/// CUDA API.
///
__host__ safe_event::operator cudaEvent_t() const
{
    return this->event;
}


/////////////////////////////////////////////////////
/// SIMPLE VECTOR
////////////////////////////////////////////////////

///
/// \brief nppiVector_t : default constructor.
/// Initialize the data pointer and the counter
/// to null, and the dimensionality attributes to 0
///
template<class T>
__host__ nppiVector_t<T>::nppiVector_t():
    data(nullptr),
    len(0)
{}

///
/// \brief nppiVector_t
/// \param size
///
template<class T>
__host__ nppiVector_t<T>::nppiVector_t(const Npp32s& size):
    nppiVector_t()
{
    this->create(size);
}

///
/// \brief nppiVector_t :
/// \param _data
/// \param size
/// \param own
///
template<class T>
__host__ nppiVector_t<T>::nppiVector_t(pointer _data, const Npp32s size, const bool& own):
    data(_data),
    len(size),
    counter(own ? new int(1) : nullptr)
{}

///
/// \brief nppiVector_t
/// \param obj
///
template<class T>
__host__ nppiVector_t<T>::nppiVector_t(const nppiVector_t& obj):
    data(obj.data),
    len(obj.len),
    counter(obj.counter)
{
    if(this->counter)
        ++(*this->counter);
}


///
/// \brief ~nppiVector_t
///
template<class T>
__host__ nppiVector_t<T>::~nppiVector_t()
{
    this->release();
}

///
/// \brief operator =
/// \param obj
/// \return
///
template<class T>
__host__ nppiVector_t<T>& nppiVector_t<T>::operator=(const nppiVector_t& obj)
{

    if(this != std::addressof(obj))
    {
        this->release();

        this->data = obj.data;
        this->len = obj.len;
        this->counter = obj.counter;

        if(this->counter)
            ++(*this->counter);
    }

    return (*this);
}



///
/// \brief size
/// \return
///
template<class T>
__host__ Npp32s nppiVector_t<T>::size() const
{
    return this->len;
}

///
/// \brief ptr
/// \param i
/// \return
///
template<class T>
__host__ typename nppiVector_t<T>::pointer nppiVector_t<T>::ptr(const Npp32s& i)
{
    return reinterpret_cast<pointer>(this->data + i * sizeof(value_type));
}

///
/// \brief ptr
/// \param i
/// \return
///
template<class T>
__host__ typename nppiVector_t<T>::const_pointer nppiVector_t<T>::ptr(const Npp32s& i)const
{
    return reinterpret_cast<const_pointer>(this->data + i * sizeof(value_type));
}

///
/// \brief create : memory allocation method
/// \param size : size to allocate or reallocate.
///
template<class T>
__host__ void nppiVector_t<T>::create(const Npp32s& _size)
{
    if(this->len != _size)
    {
        this->release();

        this->len = _size;

        this->data = nppsMalloc_8u(this->len * sizeof(value_type));
    }
}

///
/// \brief release : memory release method.
/// If the memory is own and the counter
/// after decrementation has reach 0,
/// then the memory is deallocated.
/// In any cases the attrobutes are reset
/// to null for the address and the counter
/// and 0 for the dimensionality attributes.
///
template<class T>
__host__ void nppiVector_t<T>::release()
{
    if(this->data && this->counter && !(--(*this->counter)) )
        nppsFree(this->data);
    this->data = nullptr;
    this->len = 0;
    this->counter.reset();
}


/////////////////////////////////////////////////////
/// SIMPLE MATRIX
////////////////////////////////////////////////////

///
/// \brief nppiMatrix_t : default constructor.
/// Initialize the data pointer and the counter
/// to null, and the dimensionality attributes to 0
///
template<class T>
__host__ nppiMatrix_t<T>::nppiMatrix_t():
    data(nullptr),
    rows(0),
    cols(0),
    step(0)
{}

///
/// \brief nppiMatrix_t : parametrict constructor.
/// Allocate memory in order to at least host a matrix
/// which dimensions are specify by the inputs.
/// \param _rows : number of rows of the matrix to create.
/// \param _cols : number of colmuns of the matrix to create.
///
template<class T>
__host__ nppiMatrix_t<T>::nppiMatrix_t(const Npp32s& _rows, const Npp32s& _cols)
{
    this->create(_rows, _cols);
}

///
/// \brief nppiMatrix_t : parametric constructor.
/// This constructor is an interface with memory
/// allocation outside of the class. If memory
/// can be own it will be deallocated by the current
/// object, otherwise it will not be deallocate
/// by the current object.
/// \param data :
/// \param _step
/// \param _rows
/// \param _cols
/// \param _own
///
template<class T>
__host__ nppiMatrix_t<T>::nppiMatrix_t(pointer _data, const Npp32s& _step, const Npp32s& _rows, const Npp32s& _cols, const bool& _own):
    data(_data),
    rows(_rows),
    cols(_cols),
    step(_step),
    counter(_own ? new int(1) : nullptr)
{}

///
/// \brief nppiMatrix_t : copy constructor,
/// Initialize the current object to the same
/// values as those of the provided object.
/// If the counter is initialize, it is incremented.
/// This constructor DOES NOT perform any copy.
/// \param obj : object to initialize the attributes on.
///
template<class T>
__host__ nppiMatrix_t<T>::nppiMatrix_t(const nppiMatrix_t &obj):
    data(obj.data),
    rows(obj.rows),
    cols(obj.cols),
    step(obj.step),
    counter(obj.counter)
{}


///
/// \brief ~nppiMatrix_t : destructor.
/// If the memory is own and the counter
/// after decrementation has reach 0,
/// then the memory is deallocated.
/// In any cases the attrobutes are reset
/// to null for the address and the counter
/// and 0 for the dimensionality attributes.
///
template<class T>
__host__ nppiMatrix_t<T>::~nppiMatrix_t()
{
    this->release();
}


///
/// \brief Assignation operator :
/// Initialize the current object to the same
/// values as those of the provided object.
/// If the counter is initialize, it is incremented.
/// This operator DOES NOT perform any copy.
/// \param obj : object to initialize the attributes on.
/// \return current object.
///
template<class T>
__host__ nppiMatrix_t<T>& nppiMatrix_t<T>::operator=(const nppiMatrix_t& obj)
{
    if(std::addressof(obj) != this)
    {
        this->data = obj.data;
        this->rows = obj.rows;
        this->cols = obj.cols;
        this->step = obj.step;
        this->counter = obj.counter;

        if(this->counter)
            ++(*this->counter);
    }

    return (*this);
}

///
/// \brief release : memory release method.
/// If the memory is own and the counter
/// after decrementation has reach 0,
/// then the memory is deallocated.
/// In any cases the attrobutes are reset
/// to null for the address and the counter
/// and 0 for the dimensionality attributes.
///
template<class T>
__host__ void nppiMatrix_t<T>::release()
{
    if(this->counter && this->counter && !(--(*this->counter)))
        nppiFree(this->data);

    this->data = nullptr;
    this->rows = this->cols = this->step = 0;
    this->counter.reset();
}


///
/// \brief create : memory allocation method.
/// Allocate memory in order to at least host a matrix
/// which dimensions are specify by the inputs.
/// \param _rows : number of rows of the matrix to create.
/// \param _cols : number of colmuns of the matrix to create.
///
template<class T>
__host__ void nppiMatrix_t<T>::create(const Npp32s& _rows, const Npp32s& _cols)
{
    if(_rows != this->rows || _cols != this->cols)
        this->release();

    this->rows = _rows;
    this->cols = _cols;
    this->data = reinterpret_cast<T*>(nppiMalloc_8u_C1(this->cols * sizeof(T), this->rows, &this->step));
    this->counter.reset(new int (1));
}

///
/// \brief size : accessor.
/// Return a NppiSize object.
/// \return the width and height of the current matrix
/// allocation.
///
template<class T>
__host__ NppiSize nppiMatrix_t<T>::size() const
{
    return {this->cols, this->rows};
}


///
/// \brief width : accessor.
/// \return the width of the current matrix.
///
template<class T>
__host__ Npp32s nppiMatrix_t<T>::width() const
{
    return this->cols;
}


///
/// \brief height : accessor.
/// \return the height of the current matrix.
///
template<class T>
__host__ Npp32s nppiMatrix_t<T>::height() const
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
__host__ Npp32s nppiMatrix_t<T>::pitch() const
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
__host__ typename nppiMatrix_t<T>::pointer nppiMatrix_t<T>::ptr(const Npp32s& y)
{
    return reinterpret_cast<pointer>(reinterpret_cast<unsigned char*>(this->data) + y * this->step);
}


///
/// \brief ptr : accessor.
/// return the address of the first element of the specified row.
/// \param y : index of the row to return the address of the first element of.
/// \return address of the first element of the specified specified by the row.
///
template<class T>
__host__ typename nppiMatrix_t<T>::const_pointer nppiMatrix_t<T>::ptr(const Npp32s& y) const
{
    return reinterpret_cast<const_pointer>(reinterpret_cast<const unsigned char*>(this->data) + y * this->step);
}


///
/// \brief ptr : accessor.
/// return the address of the element of the specified row and column.
/// \param y : index of the row.
/// \param x : index of the columns.
/// \return address of the element located that the y^{th} rows and x^{th} rows..
///
template<class T>
__host__ typename nppiMatrix_t<T>::pointer nppiMatrix_t<T>::ptr(const Npp32s& y, const Npp32s& x)
{
    return this->ptr(y) + x;
}


///
/// \brief ptr : accessor.
/// return the address of the element of the specified row and column.
/// \param y : index of the row.
/// \param x : index of the columns.
/// \return address of the element located that the y^{th} rows and x^{th} rows..
///
template<class T>
__host__ typename nppiMatrix_t<T>::const_pointer nppiMatrix_t<T>::ptr(const Npp32s& y, const Npp32s& x) const
{
    return this->ptr(y) + x;
}

DEF_CLASS_SPEC(nppiMatrix_t)



/////////////////////////////////////////////////////
/// SIMPLE TENSOR
////////////////////////////////////////////////////


///
/// \brief nppiTensor_t : parametric constructor.
/// This constructor is an interface with memory
/// allocation outside of the class. If memory
/// can be own it will be deallocated by the current
/// object, otherwise it will not be deallocate
/// by the current object.
/// \param data :
/// \param _step
/// \param _rows
/// \param _cols
/// \param _own
///
template<class T>
__host__ nppiTensor_t<T>::nppiTensor_t(pointer _data, const std::vector<Npp32s>& _steps, const std::vector<Npp32s>& _dimensions, const bool& _own):
    data(reinterpret_cast<unsigned char*>(_data)),
    steps(_steps),
    dims(_dimensions),
    counter(_own ? new int(1) : nullptr)
{}

///
/// \brief nppiTensor_t : copy constructor,
/// Initialize the current object to the same
/// values as those of the provided object.
/// If the counter is initialize, it is incremented.
/// This constructor DOES NOT perform any copy.
/// \param obj : object to initialize the attributes on.
///
template<class T>
__host__ nppiTensor_t<T>::nppiTensor_t(const nppiTensor_t& obj):
    data(obj.data),
    steps(obj.steps),
    dims(obj.dims),
    counter(obj.counter)
{
    if(this->counter)
        ++(*this->counter);
}



///
/// \brief ~nppiTensor_t : destructor.
/// If the memory is own and the counter
/// after decrementation has reach 0,
/// then the memory is deallocated.
/// In any cases the attrobutes are reset
/// to null for the address and the counter
/// and 0 for the dimensionality attributes.
///
template<class T>
__host__ nppiTensor_t<T>::~nppiTensor_t()
{
    this->release();
}

///
/// \brief Assignation operator :
/// Initialize the current object to the same
/// values as those of the provided object.
/// If the counter is initialize, it is incremented.
/// This operator DOES NOT perform any copy.
/// \param obj : object to initialize the attributes on.
/// \return current object.
///
template<class T>
__host__ nppiTensor_t<T>& nppiTensor_t<T>::operator=(const nppiTensor_t& obj)
{

    if(std::addressof(obj) != this)
    {
        this->release();

        this->data = obj.data;
        this->dims = obj.dims;
        this->steps = obj.steps;
        this->counter = obj.counter;

        if(this->counter)
            ++(*this->counter);
    }

    return (*this);

}



///
/// \brief order :
/// return the tensor order. (0 if it is a scalar, 1 for a vector, ...)
/// \return order of the current tensor.
///
template<class T>
__host__ Npp32s nppiTensor_t<T>::order() const
{
    return !this->dims.front() ? 0 : static_cast<Npp32s>(this->steps.size());
}


///
/// \brief dimensions :
/// return the dimensions of the current array
/// \return
///
template<class T>
__host__ std::vector<Npp32s> nppiTensor_t<T>::dimensions() const
{
    return this->dims;
}

///
/// \brief dimension :
/// return the value of the specified dimension.
/// \param idx : dimension to know about.
/// \return value of the specified dimension.
///
template<class T>
__host__ Npp32s nppiTensor_t<T>::dimension(const Npp32s& idx) const
{
    assert(idx<static_cast<Npp32s>(this->dims.size()));
    return this->dims.at(idx);
}

///
/// \brief pitchs :
/// return the value of the pitchs for all the dimensions as number of bytes.
/// \return value of the pitchs for all the dimensions as number of bytes.
///
template<class T>
__host__ std::vector<Npp32s> nppiTensor_t<T>::pitchs() const
{
    std::vector<Npp32s> ret = this->steps;

    for(Npp32s& step : ret)
        step *= sizeof(T);

    return ret;
}

///
/// \brief pitch :
/// return the value of pitch for the specified dimension as number of bytes.
/// \param idx : dimension to know about.
/// \return value of pitch for the specified dimension as number of bytes.
///
template<class T>
__host__ Npp32s nppiTensor_t<T>::pitch(const Npp32s& idx) const
{
    return this->steps.at(idx) * sizeof(T);
}

///
/// \brief release : memory release method.
/// If the memory is own and the counter
/// after decrementation has reach 0,
/// then the memory is deallocated.
/// In any cases the attrobutes are reset
/// to null for the address and the counter
/// and 0 for the dimensionality attributes.
///
template<class T>
__host__ void nppiTensor_t<T>::release()
{
    if(this->counter && this->counter && !(--(*this->counter)))
        nppiFree(this->data);

    this->data = nullptr;
    this->dims.clear();
    this->steps.clear();
    this->counter.reset();
}


template<class T>
__host__ void nppiTensor_t<T>::create(const std::vector<int>& dimensions)
{

    // Prepare the dimensions attribute.
    this->dims = std::move(dimensions);

    Npp32s step0 = std::accumulate(this->dims.begin(), this->dims.end(), 1, std::multiplies<Npp32s>()) * sizeof(value_type);

    this->data = nppsMalloc_8u(step0);

    // Step 0 correspond to the product of all the dimensions starting from the second.
    // e.g. type: float32 (i.e. sizeof -> 4),  dims : 32 x 3 x 640 x 480 -> steps: (3 x 640 x 480 x sizeof(float32)) x (640 x 480 x sizeof(float32)) x (480 x sizeof(float32)) x sizeof(float32)
    this->steps.push_back(step0);

    for(size_t i=1;i<this->dims.size(); ++i)
        this->steps.push_back(std::accumulate(this->dims.begin() + i, this->dims.end(), 1, std::multiplies<Npp32s>()) * sizeof(value_type));

    // One might observe or know that step0 value might be larger than product of all, but the first, dimensions.
    // From a practical point of view this will results in some bytes not used. This slight over-allocation allocation
    // ensure that the memory is always aligned.

}



///
/// \brief ptr : accessor.
/// return the address of the element of the specified row and column.
/// \param y : index of the first element on the dimension.
/// \param indices : index of the elements of all the other dimensions, but the first.
/// \return address of the element located that the y^{th} rows and x^{th} rows..
///
template<class T>
__host__ typename nppiTensor_t<T>::pointer nppiTensor_t<T>::ptr(const std::initializer_list<int>& indices)
{
    return reinterpret_cast<pointer>(this->data + this->get_index(indices));
}

///
/// \brief ptr : accessor.
/// return the address of the element of the specified row and column.
/// \param y : index of the first element on the dimension.
/// \param indices : index of the elements of all the other dimensions, but the first.
/// \return address of the element located that the y^{th} rows and x^{th} rows..
///
template<class T>
__host__ typename nppiTensor_t<T>::const_pointer nppiTensor_t<T>::ptr(const std::initializer_list<int>& indices) const
{
    return reinterpret_cast<const_pointer>(this->data + this->get_index(indices));
}


///
/// \brief get_index :
/// compute the index of the pointer
/// corresponding to a set of coordinates.
/// \param indices
/// \return address of the element corresponding to the provided coordinates.
///
template<class T>
Npp32s nppiTensor_t<T>::get_index(const std::initializer_list<int>& _indices)const
{
    std::vector<int> indices = _indices;

    Npp32s ret(0);

    for(size_t i=0; i<std::min(this->steps.size(), indices.size()); ++i)
        ret += indices.at(i) * this->steps.at(i);

    return ret;
}


DEF_CLASS_SPEC(nppiTensor_t)

} // cas
