
# For vectors
function permute_dims_expr(perm, D, C, B, R)
  s = Expr(:tuple) # size
  x = Expr(:tuple) # stride
  o = Expr(:tuple) # offsets
  Rnew = Expr(:tuple) # rank
  Dnew = Expr(:tuple) # dense
  Cnew = -1
  Bnew = -1
  N = length(perm)
  for n ∈ 1:N
    p = perm[n]
    push!(s.args, :($getfield(s, $p, false)))
    push!(x.args, :($getfield(x, $p, false)))
    push!(o.args, :($getfield(o, $p, false)))
    push!(Rnew.args, R[p])
    push!(Dnew.args, D[p])
    if C == p
      Cnew = n
    end
    if B == p
      Bnew = n
    end
  end
  Dnew, Cnew, Bnew, Rnew, s, x, o
end
@generated function Base.permutedims(A::PtrArray{R,C,D,B}, ::Val{P}) where {R,C,D,B,P}
  Dnew, Cnew, Bnew, Rnew, s, x, o = permute_dims_expr(P, D, C, B, R)
  quote
    $(Expr(:meta,:inline))
    p = pointer(A)
    s = size(A)
    x = strides(A)
    o = offsets(A)
    PtrArray{$Rnew,$Cnew,$Dnew,$Bnew}(p, $s, $x, $o)
  end
end

@inline Base.adjoint(A::AbstractStrideMatrix) = permutedims(A, Val{(2,1)}())
@inline Base.transpose(A::AbstractStrideMatrix) = permutedims(A, Val{(2,1)}())

@generated function Base.adjoint(a::PtrArray{R,C,D,B,S,T,1,X,O}) where {S,D,T,C,B,R,X,O}
  s = Expr(:tuple, :(One()), Expr(:call, getfield, :s, 1, false))
  x₁ = Expr(:call, getfield, :x, 1, false)
  x = Expr(:tuple, x₁, x₁)
  o = Expr(:tuple, :(One()), Expr(:call, getfield, :o, 1, false))
  R1 = R[1]
  Rnew = Expr(:tuple, R1+1, R1)
  Dnew = Expr(:tuple, true, D[1])
  Cnew = C == 1 ? 2 : C
  quote
    $(Expr(:meta,:inline))
    s = size(a)
    p = pointer(a)
    x = strides(a)
    o = offsets(a)
    PtrArray{$Rnew,$Cnew,$Dnew,$B}(p, $s, $x, $o)
  end
end

@inline Base.transpose(a::AbstractStrideVector) = adjoint(a)



