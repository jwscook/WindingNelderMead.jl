struct Vertex{T, U<:Number, V<:AbstractVector{T}}
  position::V
  value::U
end
Vertex(x::AbstractVector, f::F) where {F} = Vertex(x, f(x))

value(v::Vertex) = v.value
position(v::Vertex) = v.position
newposition(a, ϵ, b) = a + ϵ .* (a - b)

function vertexpositions(ic::T, initial_steps::AbstractVector{V}
    ) where {U, T<:AbstractVector{U}, V<:Number}
  if any(iszero, initial_steps)
    throw(ArgumentError("initial_steps, $initial_steps  must not have any zero
                        values"))
  end
  if length(ic) != length(initial_steps)
    throw(ArgumentError("ic, $ic must be same length as initial_steps
                        $initial_steps"))
  end
  dim = length(ic)
  _fx(i, ::Type{<:T}) where {T} = T(_fx(i)) # for static arrays etc
  _fx(i, ::Type{<:Vector}) = collect(_fx(i)) # standard vectors
  _fx(i) = ((@inbounds ic[j] + (j == i) * initial_steps[j]) for j ∈ 1:dim)
  positions = map(i->_fx(i, T), 1:dim+1)
  return positions
end

# must explicitly use <= and >= because == can't overridden and will
# be used in conjunction with < to create a <=
import Base: isless, +, -, <=, >=, isequal, isnan, hash
Base.isless(a::Vertex, b::Vertex) = abs(value(a)) < abs(value(b))
Base.:<=(a::Vertex, b::Vertex) = abs(value(a)) <= abs(value(b))
Base.:>=(a::Vertex, b::Vertex) = abs(value(a)) >= abs(value(b))
Base.:-(a::Vertex, b::Vertex) = position(a) - position(b)
Base.:-(a::AbstractVector, b::Vertex) = a - position(b)
Base.:+(a::Vertex, b::AbstractVector) = position(a) + b
function Base.isequal(a::Vertex, b::Vertex)
  values_equal = value(a) == value(b) || (isnan(a) && isnan(b))
  positions_equal = position(a) == position(b)
  return values_equal && positions_equal
end
Base.hash(v::Vertex) = hash(hash(v.value), hash(v.position))

Base.isnan(a::Vertex) = isnan(value(a))
Base.isfinite(a::Vertex) = isfinite(value(a))

