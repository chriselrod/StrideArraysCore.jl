
abstract type _AbstractStrideArray{T,N,S<:Tuple{Vararg{Integer,N}},D,C,B,R,X<:Tuple{Vararg{Integer,N}},O<:Tuple{Vararg{Integer,N}}} <: DenseArray{T,N} end
const AbstractStrideArray{R,C,D,B,S,T,N,X,O} = _AbstractStrideArray{T,N,S,D,C,B,R,X,O}
abstract type AbstractPtrStrideArray{R,C,D,B,S,T,N,X,O} <: AbstractStrideArray{R,C,D,B,S,T,N,X,O} end
const AbstractStrideVector{R,C,D,B,S,T,N,X,O} = AbstractStrideArray{R,C,D,B,S,T,1,X,O}
const AbstractStrideMatrix{R,C,D,B,S,T,N,X,O} = AbstractStrideArray{R,C,D,B,S,T,2,X,O}

struct PtrArray{R,C,D,B,S,T,N,X,O} <: AbstractPtrStrideArray{R,C,D,B,S,T,N,X,O}
  ptr::Ptr{T}
  size::S
  strides::X
  offsets::O
end
const PtrVector{R,C,D,B,S,T,X,O} = PtrArray{R,C,D,B,S,T,1,X,O}
const PtrMatrix{R,C,D,B,S,T,X,O} = PtrArray{R,C,D,B,S,T,2,X,O}
@inline function PtrArray(A::AbstractArray{T,N}) where {T,N}
  x = strides(A)
  o = offsets(A)
  s = size(A)
  D = map(Bool, dense_dims(A))
  R = map(Int, stride_rank(A))
  C = Int(contiguous_axis(A))
  B = Int(contiguous_batch_size(A))
  PtrArray{R,C,D,B,typeof(s),T,N,typeof(x),typeof(o)}(pointer(A), s, x, o)
end
@generated function rank_order_prod(s::Tuple{Vararg{Integer,N}}, ::Val{R}) where {R,N}
  N == 0 && return ()
  q = Expr(:block, Expr(:meta,:inline), :(x_1 = One()))
  t = Expr(:tuple, :x_1)
  r = 1; Rmax = maximum(R)
  lastx = :x_1
  for n ∈ 2:N
    while true
      i = findfirst(==(r), R)
      r += 1
      if i === nothing
        @assert r ≤ Rmax
      else
        xsym = Symbol(:x_,n)
        push!(q.args, :($xsym = $lastx * $getfield(s, $i, false)))
        push!(t.args, xsym)
        lastx = xsym
        break
      end
    end
  end
  push!(q.args, t)
  q
end
@inline rank_order_prod(s::Tuple{Vararg{Integer,N}}) where {N} = rank_order_prod(s, Val(ntuple(identity, Val(N))))
@inline function PtrArray{R,C,D,B}(
  ptr::Ptr{T}, s::S, x::X = rank_order_prod(s, Val(R)), o::O = ntuple(_->One(),Val(N))
) where {N,S<:Tuple{Vararg{Integer,N}},D,T,C,B,R,X<:Tuple{Vararg{Integer,N}},O<:Tuple{Vararg{Integer,N}}}
  PtrArray{R,C,D,B,S,T,N,X,O}(ptr, s, x, o)
end
@inline function PtrArray{R,C,D}(
  ptr::Ptr{T}, s::S, x::X = rank_order_prod(s, Val(R)), o::O = ntuple(_->One(),Val(N))
) where {N,S<:Tuple{Vararg{Integer,N}},D,T,C,R,X<:Tuple{Vararg{Integer,N}},O<:Tuple{Vararg{Integer,N}}}
  PtrArray{R,C,D,0,S,T,N,X,O}(ptr, s, x, o)
end
@inline function PtrArray{R,C}(
  ptr::Ptr{T}, s::S, x::X = rank_order_prod(s, Val(R)), o::O = ntuple(_->One(),Val(N))
) where {N,S<:Tuple{Vararg{Integer,N}},T,C,R,X<:Tuple{Vararg{Integer,N}},O<:Tuple{Vararg{Integer,N}}}
  PtrArray{R,C,ntuple(_->true,Val(N)),0,S,T,N,X,O}(ptr, s, x, o)
end
@generated function rank_1_ind(::Val{R}) where {R}
  ind = findfirst(==(1), R)
  ind === nothing ? -1 : ind
end
@inline function PtrArray{R}(
  ptr::Ptr{T}, s::S, x::X = rank_order_prod(s, Val(R)), o::O = ntuple(_->One(),Val(N))
) where {N,S<:Tuple{Vararg{Integer,N}},T,R,X<:Tuple{Vararg{Integer,N}},O<:Tuple{Vararg{Integer,N}}}
  PtrArray{R,rank_1_ind(Val(R)),ntuple(_->true,Val(N)),0,S,T,N,X,O}(ptr, s, x, o)
end
@inline function PtrArray(
  ptr::Ptr{T}, s::S, x::X = rank_order_prod(s), o::O = ntuple(_->One(),Val(N))
) where {N,S<:Tuple{Vararg{Integer,N}},T,X<:Tuple{Vararg{Integer,N}},O<:Tuple{Vararg{Integer,N}}}
  PtrArray{ntuple(identity,Val(N)),1,ntuple(_->true,Val(N)),0,S,T,N,X,O}(ptr, s, x, o)
end

@inline Base.pointer(A::PtrArray) = getfield(A, :ptr)
@inline Base.unsafe_convert(::Type{Ptr{T}}, A::AbstractStrideArray) where {T} = Base.unsafe_convert(Ptr{T}, pointer(A))
@inline Base.elsize(::_AbstractStrideArray{T}) where {T} = offsetsize(T)

@inline ArrayInterface.size(A::PtrArray) = getfield(A, :size)
@inline ArrayInterface.strides(A::PtrArray) = getfield(A, :strides)
@inline ArrayInterface.offsets(A::PtrArray) = getfield(A, :offsets)

ArrayInterface.device(::AbstractStrideArray) = ArrayInterface.CPUPointer()

ArrayInterface.contiguous_axis(::Type{<:AbstractStrideArray{R,C}}) where {R,C} = StaticInt{C}()
ArrayInterface.contiguous_batch_size(::Type{<:AbstractStrideArray{R,C,D,B}}) where {R,C,D,B} = StaticInt{B}()

static_expr(N::Int) = Expr(:call, Expr(:curly, StaticInt, N))
static_expr(b::Bool) = Expr(:call, b ? :True : :False)
@generated function ArrayInterface.stride_rank(::Type{<:AbstractStrideArray{R}}) where {R}
  t = Expr(:tuple)
  for r ∈ R
    push!(t.args, static_expr(r::Int))
  end
  t
end
@generated function ArrayInterface.dense_dims(::Type{<:AbstractStrideArray{R,C,D}}) where {R,C,D}
  t = Expr(:tuple)
  for d ∈ D
    push!(t.args, static_expr(d::Bool))
  end
  t
end

@inline function ptrarray0(ptr::Ptr{T}, s::Tuple{Vararg{Integer,N}}, x::Tuple{Vararg{Integer,N}} = rank_order_prod(s, Val(ntuple(identity,Val(N)))), ::Val{D} = Val(ntuple(_->true,Val(N)))) where {T,N,D}
  PtrArray{ntuple(identity,Val(N)),1,D}(ptr, s, x, ntuple(_->Zero(),Val(N)))
end
@inline function PtrArray(ptr::Ptr{T}, s::Tuple{Vararg{Integer,N}}, x::Tuple{Vararg{Integer,N}}, ::Val{D}) where {T,N,D}
  PtrArray{ntuple(identity,Val(N)),1,D}(ptr, s, x, ntuple(_->One(),Val(N)))
end

@inline Base.size(A::AbstractStrideArray) = map(Int, size(A))
@inline Base.strides(A::AbstractStrideArray) = map(Int, strides(A))

@inline create_axis(s, ::Zero) = CloseOpen(s)
@inline create_axis(s, ::One) = One():s
@inline create_axis(s, o) = CloseOpen(o, s+o)

@inline ArrayInterface.axes(A::AbstractStrideArray) = map(create_axis, size(A), offsets(A))
@inline Base.axes(A::AbstractStrideArray) = axes(A)

@inline ArrayInterface.static_length(A::AbstractStrideArray) = ArrayInterface.reduce_tup(*, size(A))

# type stable, because index known at compile time
@inline type_stable_select(t::NTuple, ::StaticInt{N}) where {N} = getfield(t, N, false)
@inline type_stable_select(t::Tuple, ::StaticInt{N}) where {N} = getfield(t, N, false)
# type stable, because tuple is homogenous
@inline type_stable_select(t::NTuple, i::Integer) = getfield(t, i, false)
# make the tuple homogenous before indexing
@inline type_stable_select(t::Tuple, i::Integer) = getfield(map(Int, t), i, false)

@inline function ArrayInterface._axes(A::AbstractStrideVector, i::Integer)
  if i == 1
    o = type_stable_select(offsets(A), i)
    s = type_stable_select(size(A), i)
    return create_axis(s, o)
  else
    return One():1
  end
end
@inline function ArrayInterface._axes(A::AbstractStrideArray, i::Integer)
  o = type_stable_select(offsets(A), i)
  s = type_stable_select(size(A), i)
  create_axis(s, o)
end
@inline Base.axes(A::AbstractStrideArray, i::Integer) = axes(A, i)

@inline function ArrayInterface.size(A::AbstractStrideVector, i::Integer)
    d = Int(length(A))
    ifelse(isone(i), d, one(d))
end
@inline ArrayInterface.size(A::AbstractStrideVector, ::StaticInt{N}) where {N} = One()
@inline ArrayInterface.size(A::AbstractStrideVector, ::StaticInt{1}) = length(A)
@inline ArrayInterface.size(A::AbstractStrideArray, ::StaticInt{N}) where {N} = getfield(size(A), N, false)
@inline ArrayInterface.size(A::AbstractStrideArray, i::Integer) = type_stable_select(size(A), i)
@inline Base.size(A::AbstractStrideArray, i::Integer) = size(A, i)


@generated function Base.IndexStyle(::Type{<:AbstractStrideArray{R,C,D,B,S,T,N}}) where {R,C,D,B,S,T,N}
  # if is column major || is a transposed contiguous vector
  if all(D) && ((isone(C) && R === ntuple(identity, Val(N))) || (C === 2 && R === (2,1) && S <: Tuple{One,Integer}))
    IndexLinear()
  else
    IndexCartesian()
  end          
end

@inline ManualMemory.preserve_buffer(A::PtrArray) = nothing

function rank2sortperm(R)
  map(R) do r
    sum(map(≥(r),R))
  end
end
@generated function _offset_ptr(p::Ptr{T}, x::Tuple{Vararg{Integer,N}}, o::Tuple{Vararg{Integer,N}}, ::Val{R}, i::Tuple{Vararg{Integer,NI}}) where {T,N,R,NI}
  N == 0 && return Expr(:block, Expr(:meta,:inline), :p)
  st = offsetsize(T)
  if N ≠ NI
    if (N > NI) & (NI ≠ 1)
      throw(ArgumentError("If the dimension of the array exceeds the dimension of the index, then the index should be linear/one dimensional."))
    end
    # use only the first index. Supports, for example `x[i,1,1,1,1]` when `x` is a vector, or `A[i]` where `A` is an array with dim > 1.
    return Expr(:block, Expr(:meta,:inline), :(p + (first(i)-1)*$st))
  end
  sp = rank2sortperm(R)
  q = Expr(:block, Expr(:meta,:inline))
  for n ∈ 1:N
    j = findfirst(==(n),sp)::Int
    index = Expr(:call, getfield, :i, j, false)
    offst = Expr(:call, getfield, :o, j, false)
    strid = Expr(:call, getfield, :x, j, false)
    push!(q.args, :(p += ($index - $offst)*$strid*$st))
  end
  q
end
@inline _offset_ptr(A::AbstractStrideArray{R}, i) where {R} = _offset_ptr(pointer(A), strides(A), offsets(A), Val{R}(), i)

@inline function Base.getindex(A::PtrArray, i::Vararg{Integer})
  @boundscheck checkbounds(A, i...)
  load(_offset_ptr(A, i))
end
@inline function Base.getindex(A::AbstractStrideArray, i::Vararg{Integer,K}) where {K}
  b = preserve_buffer(A)
  GC.@preserve b begin
    @boundscheck checkbounds(PtrArray(A), i...)
    load(_offset_ptr(A, i))
  end
end
@inline function Base.setindex!(A::PtrArray, v, i::Vararg{Integer,K}) where {K}
  @boundscheck checkbounds(A, i...)
  store!(_offset_ptr(A, i), v)
  v
end
@inline function Base.setindex!(A::AbstractStrideArray, v, i::Vararg{Integer,K}) where {K}
  b = preserve_buffer(A)
  GC.@preserve b begin
    @boundscheck checkbounds(PtrArray(A), i...)
    store!(_offset_ptr(A, i), v)
  end
  v
end
@inline function Base.getindex(A::PtrArray, i::Integer)
  @boundscheck checkbounds(A, i)
  load(pointer(A) + (i-oneunit(i))*offsetsize(eltype(A)))
end
@inline function Base.getindex(A::AbstractStrideArray, i::Integer)
  b = preserve_buffer(A)
  GC.@preserve b begin
    @boundscheck checkbounds(PtrArray(A), i)
    load(pointer(A) + (i-oneunit(i))*offsetsize(eltype(A)))
  end
end
@inline function Base.setindex!(A::PtrArray, v, i::Integer)
  @boundscheck checkbounds(A, i)
  store!(pointer(A) + (i-oneunit(i))*offsetsize(eltype(A)), v)
  v
end
@inline function Base.setindex!(A::AbstractStrideArray, v, i::Integer)
  b = preserve_buffer(A)
  GC.@preserve b begin
    @boundscheck checkbounds(PtrArray(A), i)
    store!(pointer(A) + (i-oneunit(i))*offsetsize(eltype(A)), v)
  end
  v
end


