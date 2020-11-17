struct Vertex{T, U}
  position::AbstractVector{T}
  value::U
end
Vertex(x::AbstractVector{T}, f::F) where {T, F<:Function} = Vertex(x, f(x))

value(v::Vertex) = v.value
position(v::Vertex) = v.position
newposition(a, ϵ, b) = a + ϵ .* (a - b)

# must explicitly use <= and >= because == can't overridden and will
# be used in conjunction with < to create a <=
import Base: isless, +, -, <=, >=, isequal, isnan, hash
Base.isless(a::Vertex, b::Vertex) = abs(value(a)) < abs(value(b))
Base.:<=(a::Vertex, b::Vertex) = abs(value(a)) <= abs(value(b))
Base.:>=(a::Vertex, b::Vertex) = abs(value(a)) >= abs(value(b))
Base.:+(a::Vertex, b) = position(a) .+ b
Base.:-(a::Vertex, b::Vertex) = position(a) .- position(b)
Base.:-(a, b::Vertex) = a .- position(b)
function Base.isequal(a::Vertex, b::Vertex)
  values_equal = value(a) == value(b) || (isnan(a) && isnan(b))
  positions_equal = all(position(a) .== position(b))
  return values_equal && positions_equal
end
Base.isequal(a::Vertex, b::AbstractVector) = all(position(a) .== b)

Base.isnan(a::Vertex) = isnan(value(a))
Base.hash(v::Vertex) = hash(v, hash(:Vertex))
Base.hash(v::Vertex, h::UInt64) = hash(hash.(v.position), hash(v.value, h))

