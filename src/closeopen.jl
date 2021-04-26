

struct CloseOpen{L <: Union{Int,StaticInt}, S <: Union{Int,StaticInt}, U <: Union{Int,StaticInt}} <: AbstractRange{Int}
  lower::L
  step::S
  upper::U
end
@inline _tofield(::StaticInt{L}) where {L} = StaticInt{L}()
@inline _tofield(i::Integer) = i % Int
@inline CloseOpen(l,s,u) = CloseOpen(_tofield(l),_tofield(s),_tofield(u))
@inline CloseOpen(l,u) = CloseOpen(_tofield(l),One(),_tofield(u))
@inline CloseOpen(u) = CloseOpen(Zero(),One(),_tofield(u))


@inline Base.first(r::CloseOpen) = getfield(r, :lower)
@inline Base.step(r::CloseOpen) = getfield(r, :step)
@inline Base.last(r::CloseOpen) = getfield(r, :upper) - One()
@inline Base.length(r::CloseOpen{L,One}) where {L} = getfield(r, :upper) - getfield(r, :lower)
@inline Base.length(r::CloseOpen{Zero,One}) = getfield(r, :upper)
@inline function Base.length(r::CloseOpen)
  s = Int(getfield(r, :step))
  Base.udiv_int(getfield(r, :upper) - getfield(r, :lower) + s - 1, s)
end

@inline Base.iterate(r::CloseOpen) = Base.iterate(r, Int(getfield(r,:lower)))
@inline function Base.iterate(r::CloseOpen, i::Int)
  i ≥ getfield(r,:upper) ? nothing : i + getfield(r,:step)
end

ArrayInterface.known_first(::Type{<:CloseOpen{StaticInt{F}}}) where {F} = F
ArrayInterface.known_step(::Type{<:CloseOpen{<:Any,StaticInt{S}}}) where {S} = S
ArrayInterface.known_last(::Type{<:CloseOpen{<:Any,<:Any,StaticInt{L}}}) where {L} = L - 1
ArrayInterface.known_length(::Type{CloseOpen{StaticInt{F},StaticInt{S},StaticInt{L}}}) where {F,S,L} = Base.udiv_int(L-F+S-1,S)

Base.IteratorSize(::Type{<:CloseOpen}) = Base.HasShape{1}()
Base.IteratorEltype(::Type{<:CloseOpen}) = Base.HasEltype()
@inline Base.size(r::CloseOpen) = (length(r),)
Base.eltype(::CloseOpen) = Int


