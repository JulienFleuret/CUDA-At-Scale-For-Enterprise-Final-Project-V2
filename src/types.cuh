#ifndef TYPES_CUH
#define TYPES_CUH 1

#pragma once

#include <npp.h>

#include <memory>
#include <vector>

namespace cas // cuda at scale
{

/////////////////////////////////////////////////////
/// SAFE STREAM
////////////////////////////////////////////////////

class safe_event;

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
    /// \brief waitEvent : wait for an event to finish.
    /// \param event : event to monitor.
    /// \param flags : attributes to set.
    ///
    __host__ void waitEvent(const safe_event& event, const unsigned int& flags = cudaEventWaitDefault);

    ///
    /// \brief operator cudaStream_t : implicit conversion operator.
    /// Convinient to maintain compatibility with the rest of the
    /// CUDA API.
    ///
    __host__ operator cudaStream_t() const;

};

/////////////////////////////////////////////////////
/// SAFE EVENT
////////////////////////////////////////////////////

///
/// \brief The safe_event class
/// The goal of this class is to
/// manage a cudaEvent_t. Why
/// do one need this class?
/// Simply it allows to properly
/// deallocate the GPU resoureces
/// in case of an unexpected event.
/// Yes the GPU resource can be reset
/// at the begining of a program, or
/// using nividia-smi. But isn't
/// better to manage them in a safeway?
///
class safe_event
{
private:

    cudaEvent_t event;

public:

    ///
    /// \brief safe_event : default constructor.
    /// Initialize the attribute "event" to nullptr.
    ///
    __host__ safe_event();

    ///
    /// \brief safe_event : parametric constructor.
    /// \param event : event to own.
    /// The stream provided as argument is own
    /// by the object, and will be destroy either
    /// by calling the method "destroy" of by
    /// the destructor.
    ///
    __host__ safe_event(const cudaEvent_t& event);

    ///
    /// \note For the copy constructor to make
    /// sense it would require a reference counter.
    ///
    safe_event(const safe_event&) = delete;

    ///
    /// \brief safe_event : move constructor.
    /// \param : object to move.
    ///
    safe_event(safe_event&&) = default;

    ///
    /// \brief ~safe_event : destructor
    /// If an event was created, it will be
    /// destroy.
    ///
    __host__ ~safe_event();

    ///
    /// \brief For the assignation operator to make sense
    /// it would require a reference counter.
    ///
    safe_event& operator=(const safe_event&) = delete;

    ///
    /// \brief move operator : swap the attribute
    /// between the current and the argument object.
    /// \return the current object
    ///
    safe_event& operator=(safe_event&&) = default;

    ///
    /// \brief create : create the event.
    /// \param flags : attriutes to apply on the event.
    ///
    __host__ void create(const unsigned int& flags = cudaEventDefault);

    ///
    /// \brief destroy : destroy the current
    /// event, if it was created. Otherwise
    /// do nothing.
    ///
    __host__ void destroy();

    ///
    /// \brief record : record an event
    /// \param _stream : stream to record.
    /// \param _flags : attributes to apply on the recording.
    ///
    __host__ void record(const safe_stream& _stream = safe_stream(0), const unsigned int& _flags = cudaEventDefault);

    ///
    /// \brief synchonize : waits for an event to complete.
    ///
    __host__ void synchonize();

    ///
    /// \brief operator cudaEvent_t : implicit conversion operator.
    /// Convinient to maintain compatibility with the rest of the
    /// CUDA API.
    ///
    __host__ operator cudaEvent_t() const;
};


/////////////////////////////////////////////////////
/// SIMPLE MATRIX
////////////////////////////////////////////////////

///
/// \brief The nppiMatrix_t class
/// Simple matrix class, for device
/// memory management. Use a reference
/// counter approach in order to reduce
/// the need for copy.
///
template<class T>
class nppiMatrix_t
{
public:

    using pointer = T*;
    using const_pointer = const T*;

    ///
    /// \brief nppiMatrix_t : default constructor.
    /// Initialize the data pointer and the counter
    /// to null, and the dimensionality attributes to 0
    ///
    __host__ nppiMatrix_t();

    ///
    /// \brief nppiMatrix_t : parametrict constructor.
    /// Allocate memory in order to at least host a matrix
    /// which dimensions are specify by the inputs.
    /// \param _rows : number of rows of the matrix to create.
    /// \param _cols : number of colmuns of the matrix to create.
    ///
    __host__ nppiMatrix_t(const Npp32s& _rows, const Npp32s& _cols);

    ///
    /// \brief nppiMatrix_t : parametric constructor.
    /// This constructor is an interface with memory
    /// allocation outside of the class. If memory
    /// can be own it will be deallocated by the current
    /// object, otherwise it will not be deallocate
    /// by the current object.
    /// \param data : pointer on the data to own.
    /// \param _step : number of bytes for move from a row to another.
    /// \param _rows : number of rows.
    /// \param _cols : number of columns.
    /// \param _own : should the memory be deallocated by the current object.
    ///
    __host__ nppiMatrix_t(pointer data, const Npp32s& _step, const Npp32s& _rows, const Npp32s& _cols, const bool& _own = false);

    ///
    /// \brief nppiMatrix_t : copy constructor,
    /// Initialize the current object to the same
    /// values as those of the provided object.
    /// If the counter is initialize, it is incremented.
    /// This constructor DOES NOT perform any copy.
    /// \param obj : object to initialize the attributes on.
    ///
    __host__ nppiMatrix_t(const nppiMatrix_t& obj);

    ///
    /// \brief nppiMatrix_t : move constructor.
    /// swap the attributes of the current and
    /// provided objects.
    /// \param : object to move.
    ///
    nppiMatrix_t(nppiMatrix_t&&) = default;

    ///
    /// \brief ~nppiMatrix_t : destructor.
    /// If the memory is own and the counter
    /// after decrementation has reach 0,
    /// then the memory is deallocated.
    /// In any cases the attrobutes are reset
    /// to null for the address and the counter
    /// and 0 for the dimensionality attributes.
    ///
    __host__ ~nppiMatrix_t();

    ///
    /// \brief Assignation operator :
    /// Initialize the current object to the same
    /// values as those of the provided object.
    /// If the counter is initialize, it is incremented.
    /// This operator DOES NOT perform any copy.
    /// \param obj : object to initialize the attributes on.
    /// \return current object.
    ///
    __host__ nppiMatrix_t& operator=(const nppiMatrix_t& obj);

    ///
    /// \brief Move operator :
    /// Swap the attributes of the current and
    /// provided objects.
    /// \param : object to move.
    /// \return current object.
    ///
    nppiMatrix_t& operator=(nppiMatrix_t&&) = default;

    ///
    /// \brief release : memory release method.
    /// If the memory is own and the counter
    /// after decrementation has reach 0,
    /// then the memory is deallocated.
    /// In any cases the attrobutes are reset
    /// to null for the address and the counter
    /// and 0 for the dimensionality attributes.
    ///
    __host__ void release();

    ///
    /// \brief create : memory allocation method.
    /// Allocate memory in order to at least host a matrix
    /// which dimensions are specify by the inputs.
    /// \param _rows : number of rows of the matrix to create.
    /// \param _cols : number of colmuns of the matrix to create.
    ///
    __host__ void create(const Npp32s& _rows, const Npp32s& _cols);

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

    pointer data;
    Npp32s rows, cols, step;
    std::shared_ptr<int> counter;
};


/////////////////////////////////////////////////////
/// SIMPLE TENSOR
////////////////////////////////////////////////////

///
/// \brief The nppiTensor_t class
/// Simple tensor class, for device
/// memory management. Use a reference
/// counter approach in order to reduce
/// the need for copy.
///
template<class T>
class nppiTensor_t
{
public:

    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;

    ///
    /// \brief nppiTensor_t : default or parametrict constructor.
    /// Allocate memory in order to at least host a matrix
    /// which dimensions are specify by the inputs.
    /// \param _rows : number of rows of the matrix to create.
    /// \param _cols : number of colmuns of the matrix to create.
    ///
    template<class... Args>
    __host__ nppiTensor_t(const Args&... dimensions);

    ///
    /// \brief nppiTensor_t : parametric constructor.
    /// This constructor is an interface with memory
    /// allocation outside of the class. If memory
    /// can be own it will be deallocated by the current
    /// object, otherwise it will not be deallocate
    /// by the current object.
    /// \param data : pointer on the data.
    /// \param _steps : steps for each dimensions.
    /// \param _dimensions : dimensions.
    /// \param _own : should the memory be deallocated by the current object or not.
    ///
    __host__ nppiTensor_t(pointer data, const std::vector<Npp32s>& steps, const std::vector<Npp32s>& dimensions, const bool& _own = false);

    ///
    /// \brief nppiTensor_t : copy constructor,
    /// Initialize the current object to the same
    /// values as those of the provided object.
    /// If the counter is initialize, it is incremented.
    /// This constructor DOES NOT perform any copy.
    /// \param obj : object to initialize the attributes on.
    ///
    __host__ nppiTensor_t(const nppiTensor_t& obj);

    ///
    /// \brief nppiTensor_t : move constructor.
    /// swap the attributes of the current and
    /// provided objects.
    /// \param : object to move.
    ///
    nppiTensor_t(nppiTensor_t&&) = default;

    ///
    /// \brief ~nppiTensor_t : destructor.
    /// If the memory is own and the counter
    /// after decrementation has reach 0,
    /// then the memory is deallocated.
    /// In any cases the attrobutes are reset
    /// to null for the address and the counter
    /// and 0 for the dimensionality attributes.
    ///
    __host__ ~nppiTensor_t();

    ///
    /// \brief Assignation operator :
    /// Initialize the current object to the same
    /// values as those of the provided object.
    /// If the counter is initialize, it is incremented.
    /// This operator DOES NOT perform any copy.
    /// \param obj : object to initialize the attributes on.
    /// \return current object.
    ///
    __host__ nppiTensor_t& operator=(const nppiTensor_t& obj);

    ///
    /// \brief Move operator :
    /// Swap the attributes of the current and
    /// provided objects.
    /// \param : object to move.
    /// \return current object.
    ///
    nppiTensor_t& operator=(nppiTensor_t&&) = default;

    ///
    /// \brief order :
    /// return the tensor order. (0 if it is a scalar, 1 for a vector, ...)
    /// \return order of the current tensor.
    ///
    __host__ Npp32s order() const;


    ///
    /// \brief dimensions :
    /// return the dimensions of the current array
    /// \return
    ///
    __host__ std::vector<Npp32s> dimensions() const;

    ///
    /// \brief dimension :
    /// return the value of the specified dimension.
    /// \param idx : dimension to know about.
    /// \return value of the specified dimension.
    ///
    __host__ Npp32s dimension(const Npp32s& idx) const;

    ///
    /// \brief pitchs :
    /// return the value of the pitchs for all the dimensions as number of bytes.
    /// \return value of the pitchs for all the dimensions as number of bytes.
    ///
    __host__ std::vector<Npp32s> pitchs() const;

    ///
    /// \brief pitch :
    /// return the value of pitch for the specified dimension as number of bytes.
    /// \param idx : dimension to know about.
    /// \return value of pitch for the specified dimension as number of bytes.
    ///
    __host__ Npp32s pitch(const Npp32s& idx) const;

    ///
    /// \brief release : memory release method.
    /// If the memory is own and the counter
    /// after decrementation has reach 0,
    /// then the memory is deallocated.
    /// In any cases the attrobutes are reset
    /// to null for the address and the counter
    /// and 0 for the dimensionality attributes.
    ///
    __host__ void release();

    ///
    /// \brief create
    /// \param dimensions : dimensions of the current object.
    ///
    template<class... Args>
    void create(const Args&... dimensions);


    ///
    /// \brief ptr : accessor.
    /// return the address of the element of the specified row and column.
    /// \param y : index of the first element on the dimension.
    /// \param indices : index of the elements of all the other dimensions, but the first.
    /// \return address of the element located that the y^{th} rows and x^{th} rows..
    ///
    template<class... Args>
    __host__ pointer ptr(const Args&... indices);

    ///
    /// \brief ptr : accessor.
    /// return the address of the element of the specified row and column.
    /// \param y : index of the first element on the dimension.
    /// \param indices : index of the elements of all the other dimensions, but the first.
    /// \return address of the element located that the y^{th} rows and x^{th} rows..
    ///
    template<class... Args>
    __host__ const_pointer ptr(const Args&... indices) const;


private:

    ///
    /// \brief get_index :
    /// compute the index of the pointer
    /// corresponding to a set of coordinates.
    /// \param indices
    /// \return address of the element corresponding to the provided coordinates.
    ///
    template<class... Args>
    Npp32s get_index(const Args&... indices)const;

    unsigned char* data;
    std::vector<Npp32s> dims, steps;
    std::shared_ptr<int> counter;

};

} // cas

#endif // TYPES_CUH
