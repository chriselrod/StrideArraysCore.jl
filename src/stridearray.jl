
@inline undef_memory_buffer(::Type{T}, ::StaticInt{L}) where {T,L} = MemoryBuffer{L,T}(undef)
@inline undef_memory_buffer(::Type{T}, L) where {T} = Vector{T}(undef, L)

struct StrideArray{R,C,D,B,S,T,N,X,O,A} <: AbstractStrideArray{R,C,D,B,S,T,N,X,O}
  ptr::PtrArray{R,C,D,B,S,T,N,X,O}
  data::A
end

const StrideVector{R,C,D,B,S,T,X,O} = StrideArray{R,C,D,B,S,T,1,X,O}
const StrideMatrix{R,C,D,B,S,T,X,O} = StrideArray{R,C,D,B,S,T,2,X,O}

@inline StrideArray(A::AbstractArray) = StrideArray(PtrArray(A), ManualMemory.preserve_buffer(A))
@inline ManualMemory.preserve_buffer(A::StrideArray) = getfield(A, :data)

@inline function StrideArray{T}(::UndefInitializer, s::Tuple{Vararg{Integer,N}}) where {N,T}
  x = rank_order_prod(s)
  L = getfield(x, N, false) * getfield(s, N, false)
  b = undef_memory_buffer(T, L)
  StrideArray(pointer(b), s, x, b, all_dense(Val{N}()))
end
@inline function StrideArray(ptr::Ptr{T}, s::S, x::X, b, ::Val{D}) where {S,X,T,D}
    StrideArray(PtrArray(ptr, s, x, Val{D}()), b)
end
@inline StrideArray(::UndefInitializer, s::Vararg{Integer,N}) where {N} = StrideArray{Float64}(undef, s)
@inline StrideArray(::UndefInitializer, s::Tuple{Vararg{Integer,N}}) where {N} = StrideArray{Float64}(undef, s)
@inline StrideArray(::UndefInitializer, ::Type{T}, s::Vararg{Integer,N}) where {T,N} = StrideArray{T}(undef, s)
# @inline function StrideArray(A::PtrArray{S,D,T,N}, s::Tuple{Vararg{Integer,N}}) where {S,D,T,N}
#   PtrArray(stridedpointer(A), s, val_dense_dims(A))
# end
# @inline StrideArray(A::AbstractArray{T,N}, s::Tuple{Vararg{Integer,N}}) where {T,N} = StrideArray(PtrArray(A), preserve_buffer(A))
@inline Base.pointer(A::StrideArray) = getfield(getfield(A,:ptr),:ptr)

@generated function to_static_tuple(::Val{S}) where {S}
  t = Expr(:tuple)
  for s ∈ S.parameters
    push!(t.args, Expr(:new, s))
  end
  t
end
mutable struct StaticStrideArray{R,C,D,B,S,T,N,X,O,L} <: AbstractStrideArray{R,C,D,B,S,T,N,X,O}
  data::NTuple{L,T}
  @inline StaticStrideArray{R,C,D,B,S,T,N,X,O}(::UndefInitializer) where {R,C,D,B,S,T,N,X,O} = new{R,C,D,B,S,T,N,X,O,Int(prod(to_static_tuple(Val(S))))}()
  @inline StaticStrideArray{R,C,D,B,S,T,N,X,O,L}(::UndefInitializer) where {R,C,D,B,S,T,N,X,O,L} = new{R,C,D,B,S,T,N,X,O,L}()
end

@inline ArrayInterface.size(::StaticStrideArray{R,C,D,B,S}) where {R,C,D,B,S} = to_static_tuple(Val(S))
@inline ArrayInterface.strides(::StaticStrideArray{R,C,D,B,S,T,N,X}) where {R,C,D,B,S,T,N,X} = to_static_tuple(Val(X))
@inline ArrayInterface.offsets(::StaticStrideArray{R,C,D,B,S,T,N,X,O}) where {R,C,D,B,S,T,N,X,O} = to_static_tuple(Val(O))
@inline Base.unsafe_convert(::Type{Ptr{T}}, A::StaticStrideArray) where {T} = Base.unsafe_convert(Ptr{T}, pointer_from_objref(A))
@inline Base.pointer(A::StaticStrideArray{R,C,D,B,S,T}) where {R,C,D,B,S,T} = Base.unsafe_convert(Ptr{T}, pointer_from_objref(A))
@inline function StaticStrideArray{T}(::UndefInitializer, s::Tuple{Vararg{StaticInt,N}}) where {N,T}
  x = rank_order_prod(s)
  L = getfield(x,N,false) * getfield(s,N,false)
  R = ntuple(Int, Val(N))
  O = ntuple(_ -> One(), Val(N))
  StaticStrideArray{R,1,ntuple(_->true,Val(N)),0,typeof(s),T,N,typeof(x),typeof(O),L}(undef)
end
@inline StrideArray{T}(::UndefInitializer, s::Tuple{Vararg{StaticInt,N}}) where {N,T} = StaticStrideArray{T}(undef, s)

function dense_quote(N::Int, b::Bool)
  d = Expr(:tuple)
  for n in 1:N
    push!(d.args, b)
  end
  Expr(:call, Expr(:curly, :Val, d))
end
@generated all_dense(::Val{N}) where {N} = dense_quote(N, true)


@inline maybe_ptr_array(A) = A
@inline maybe_ptr_array(A::AbstractArray) = maybe_ptr_array(ArrayInterface.device(A), A)
@inline maybe_ptr_array(::ArrayInterface.CPUPointer, A::AbstractArray) = PtrArray(A)
@inline maybe_ptr_array(_, A::AbstractArray) = A

@inline ArrayInterface.size(A::StrideArray) = getfield(getfield(A, :ptr), :size)

@inline ArrayInterface.strides(A::StrideArray) = strides(getfield(A, :ptr))
@inline ArrayInterface.offsets(A::StrideArray) = getfield(getfield(A, :ptr), :offsets)

@inline zeroindex(r::ArrayInterface.OptionallyStaticUnitRange{One}) = CloseOpen(Zero(), last(r))
@inline zeroindex(r::Base.OneTo) = CloseOpen(Zero(), last(r))
@inline zeroindex(r::AbstractUnitRange) = Zero():length(r)

@inline zeroindex(r::CloseOpen{Zero}) = r
@inline zeroindex(r::ArrayInterface.OptionallyStaticUnitRange{Zero}) = r
@inline zeroindex(A::PtrArray{R,C,D,B,S,T,N}) where {R,C,D,B,S,T,N} = PtrArray{R,C,D,B}(pointer(A), size(A), strides(A), ntuple(_->Zero(),Val(N)))
@inline zeroindex(A::StrideArray) = StrideArray(zeroindex(PtrArray(A)), preserve_buffer(A))
@inline zeroindex(A::StaticStrideArray) = StrideArray(zeroindex(PtrArray(A)), A)

@generated rank_to_sortperm_val(::Val{R}) where {R} = :(Val{$(rank_to_sortperm(R))}())
@inline function similar_layout(A::AbstractStrideArray{R}) where {R}
  permutedims(similar(permutedims(A, rank_to_sortperm_val(Val{R}()))), Val{R}())
end
@inline function similar_layout(A::AbstractArray)
  b = preserve_buffer(A)
  GC.@preserve b begin
    similar_layout(PtrArray(A))
  end
end
@inline function Base.similar(A::_AbstractStrideArray{T}) where {T}
  StrideArray{T}(undef, size(A))
end
@inline function Base.similar(A::AbstractStrideArray, ::Type{T}) where {T}
  StrideArray{T}(undef, size(A))
end

@inline function Base.view(A::StrideArray, i::Vararg{Union{Integer,AbstractRange,Colon},K}) where {K}
  StrideArray(view(A.ptr, i...), A.data)
end
@inline function zview(A::StrideArray, i::Vararg{Union{Integer,AbstractRange,Colon},K}) where {K}
  StrideArray(zview(A.ptr, i...), A.data)
end
@inline function Base.permutedims(A::StrideArray, ::Val{P}) where {P}
  StrideArray(permutedims(A.ptr, Val{P}()), A.data)
end
@inline Base.adjoint(a::StrideVector) = StrideArray(adjoint(a.ptr), a.data)


function gc_preserve_call(ex, skip=0)
  q = Expr(:block)
  call = Expr(:call, esc(ex.args[1]))
  gcp = Expr(:gc_preserve, call)
  for i ∈ 2:length(ex.args)
    arg = ex.args[i]
    if i+1 ≤ skip
      push!(call.args, arg)
      continue
    end
    A = gensym(:A); buffer = gensym(:buffer);
    if arg isa Expr && arg.head === :kw
      push!(call.args, Expr(:kw, arg.args[1], Expr(:call, :maybe_ptr_array, A)))
      arg = arg.args[2]
    else
      push!(call.args, Expr(:call, :maybe_ptr_array, A))
    end
    push!(q.args, :($A = $(esc(arg))))
    push!(q.args, Expr(:(=), buffer, Expr(:call, :preserve_buffer, A)))
    push!(gcp.args, buffer)
  end
  push!(q.args, gcp)
  q
end
"""
  @gc_preserve foo(A, B, C)

Apply to a single, non-nested, function call. It will `GC.@preserve` all the arguments, and substitute suitable arrays with `PtrArray`s.
This has the benefit of potentially allowing statically sized mutable arrays to be both stack allocated, and passed through a non-inlined function boundary.
"""
macro gc_preserve(ex)
  @assert ex.head === :call
  gc_preserve_call(ex)
end

