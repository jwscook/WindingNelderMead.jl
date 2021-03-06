struct Simplex{D,T<:Number,U<:Complex,V}
  vertices::Vector{Vertex{T,U,V}}
  permabs::Vector{Int}
  function Simplex(vertices::Vector{Vertex{T,U,V}},
      permabs::Vector{Int}=zeros(Int64, length(vertices))
      ) where {T,U<:Complex,V}
    D = length(vertices) - 1
    output = new{D,T,U,V}(vertices, permabs)
    sort!(output)
    return output
  end
end

function Simplex(f::T, ic::U, initial_step::Number
    ) where {T, U<:AbstractVector{<:Number}}
  return Simplex(f, ic, fill!(similar(ic), initial_step))
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
  vertices = [Vertex(p, f(p)) for p in positions]
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
    return @inbounds atan(pv[2] - c[2], pv[1] - c[1])
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

selectabs(s, index) = @inbounds s.vertices[s.permabs[index]]

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

  toprocess = BitSet(1)
  processed = BitSet()
  connectedto = BitSet()
  @inbounds while !isempty(toprocess)
    vi = pop!(toprocess)
    v = simplex.vertices[vi]
    pv = position(v)
    empty!(connectedto)
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
  @inbounds for (vi, v) ∈ enumerate(simplex)
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

function windingangle(s::Simplex{D,T,U}) where {D,T,U}
  θ = zero(real(U))
  @inbounds for i in 1:length(s)
    θ += _πtoπ(angle(value(s.vertices[mod1(i+1, length(s))])) -
               angle(value(s.vertices[i])))
  end
  return θ
end

function windingnumber(s::Simplex)
  radians = windingangle(s)
  return isfinite(radians) ? Int64(round(radians / 2π)) : Int64(0)
end


