module StrideArraysCore

using ArrayInterface, ThreadingUtilities, ManualMemory
using Static: StaticInt, Zero, One, StaticBool, True, False
using ArrayInterface: 
  OptionallyStaticUnitRange, size, strides, offsets, indices,
  static_length, static_first, static_last, axes,
  dense_dims, stride_rank, offset1, StrideIndex,
  contiguous_axis, contiguous_batch_size
using ManualMemory: MemoryBuffer, offsetsize, preserve_buffer, load, store!


export PtrArray, StrideArray, StaticInt

include("closeopen.jl")
include("ptr_array.jl")
include("stridearray.jl")
include("thread_compatible.jl")
include("views.jl")
include("adjoints.jl")

function __init__()
    # @require LoopVectorization="bdcacae8-1622-11e9-2a5c-532679323890" @eval using StrideArrays
end

end
