using StaticArrays

#permabsarray(x) = SizedVector((UInt32(0) for _ in eachindex(x))...)
struct Simplex{D,T<:Number,U<:Complex,V,W<:AbstractVector{Vertex{T,U,V}},
    X<:AbstractVector{<:Integer}}
  vertices::W
  permabs::X
  function Simplex(vertices::W) where
      {T,U<:Complex,V,W<:AbstractVector{Vertex{T,U,V}}}
    D = length(vertices) - 1
#    permabs = permabsarray(vertices)
    permabs = SizedVector((UInt32(0) for _ in eachindex(vertices))...)
    X = typeof(permabs)
    output = new{D,T,U,V,W,X}(vertices, permabs)
    sort!(output)
    return output
  end
end

function Simplex(f::T, ic::AbstractVector{<:Number}, initial_step::Number
    ) where {T}
  initial_steps = [initial_step for i ∈ eachindex(ic)] # opportunity for SizedArray
  return Simplex(f, ic, initial_steps)
end

function Simplex(f::T, ic::AbstractVector{U}, initial_steps::AbstractVector{V}
    ) where {T, U<:Number, V<:Number}
  return Simplex(f, vertexpositions(ic, initial_steps))
end
function Simplex(f::T, positions::U
    ) where {T, W<:Number, V<:AbstractVector{W}, U<:AbstractVector{V}}
  if length(unique(length.(positions))) != 1
    throw(ArgumentError("All entries in positions $positions must be the same
                        length"))
  end
  vertices = SizedVector((Vertex(p, f(p)) for p in positions)...)
  return Simplex(vertices)
end


import Base: length, iterate, iterate, getindex
import Base: eachindex, sort!, hash, extrema
Base.length(s::Simplex{D}) where {D} = D + 1
Base.iterate(s::Simplex) = iterate(s.vertices)
Base.iterate(s::Simplex, counter) = iterate(s.vertices, counter)
Base.getindex(s::Simplex, index) = s.vertices[index]

sortby(s::Simplex) = v->angle(value(v))

function sortby(s::Simplex{2})
  return function output(v)
    c = centre(s)
    pv = position(v)
    return atan(pv[2] - c[2], pv[1] - c[1])
  end
end

function Base.sort!(s::Simplex)
  sort!(s.vertices, by=sortby(s))
  sortperm!(s.permabs, s.vertices, by=x->abs(value(x)))
  return nothing
end

issortedbyangle(s::Simplex) = issorted(s, by=sortby(s))

function Base.extrema(s::Simplex)
  return [extrema(position(v)[i] for v in s) for i in 1:dimensionality(s)]
end

dimensionality(s::Simplex{D}) where {D} = D

selectabs(s, index) = s.vertices[s.permabs[index]]

bestvertex(s::Simplex) = selectabs(s, 1)
worstvertex(s::Simplex) = selectabs(s, length(s))
secondworstvertex(s::Simplex) = selectabs(s, length(s) - 1)

function centroidposition(s::Simplex, ignoredvertex=worstvertex(s))
  g(v) = isequal(v, ignoredvertex) ? zero(position(v)) : position(v)
  return mapreduce(g, +, s) / (length(s) - 1)
end

centre(s::Simplex) = mapreduce(position, +, s) / length(s)

function hypervolume(s::Simplex)
  m = hcat((vcat(position(v), 1) for v in s)...)
  d = dimensionality(s)
  return abs(det(m)) / factorial(d)
end

function swap!(s::Simplex, this::Vertex, forthat::Vertex)
  @assert this ∈ s.vertices
  s.vertices[findfirst(x -> isequal(x, this), s.vertices)] = forthat
  sort!(s)
  return nothing
end

swapworst!(s::Simplex, forthis::Vertex) = swap!(s, worstvertex(s), forthis)

function closestomiddlevertex(s::Simplex)
  mid = mapreduce(position, +, s) ./ length(s)
  _, index = findmin(map(v->sum(abs, position(v) - mid), s))
  return s[index]
end

function assessconvergence(simplex, config)

  if abs(value(bestvertex(simplex))) <= config[:stopval]
    return :STOPVAL_REACHED
  end

  toprocess = Set{Int}(1)
  processed = Set{Int}()
  while !isempty(toprocess)
    vi = pop!(toprocess)
    v = simplex.vertices[vi]
    pv = position(v)
    connectedto = Set{Int}()
    for (qi, q) ∈ enumerate(simplex)
      thisxtol = true
      pq = position(q)
      @inbounds for i ∈ 1:dimensionality(simplex)
        thisxtol &= isapprox(pv[i], pq[i],
          rtol=config[:xtol_rel][i], atol=config[:xtol_abs][i])
        thisxtol || break
      end
      thisxtol && push!(connectedto, qi)
      thisxtol && for i in connectedto if i ∉ processed push!(toprocess, i) end end
    end
    push!(processed, vi)
  end
  allxtol = all(i ∈ processed for i ∈ 1:length(simplex))
  allxtol && return :XTOL_REACHED

  allftol = true
  for (vi, v) ∈ enumerate(simplex)
    for qi ∈ vi+1:length(simplex)
      q = simplex.vertices[qi]
      allftol &= all(isapprox(value(v), value(q),
                              rtol=config[:ftol_rel], atol=config[:ftol_abs]))
      position(v) == position(q) && return :XTOL_DEGENERATE_SIMPLEX
    end
  end
  allftol && return :FTOL_REACHED

  return :CONTINUE
end

function _πtoπ(ϕ::T) where {T}
  ϕ < -T(π) && return _πtoπ(ϕ + 2π)
  ϕ >= T(π) && return _πtoπ(ϕ - 2π)
  return ϕ
end

function windingangle(s::Simplex)
  return sum(_πtoπ.(angle.(value.(circshift(s.vertices, -1))) .-
                    angle.(value.(s.vertices))))
end

function windingnumber(s::Simplex)
  radians = windingangle(s)
  return isfinite(radians) ? Int64(round(radians / 2π)) : Int64(0)
end


