struct Simplex{T<:Number, U}
  vertices::Vector{Vertex{T,U}}
  permabs::Vector{Int}
  function Simplex(vertices::Vector{Vertex{T,U}}) where {T<:Number, U}
    output = new{T,U}(vertices, zeros(Int64, length(vertices)))
    sort!(output)
    return output
  end
end

function Simplex(f::T, ic::AbstractVector{U}, initial_step::Number
    ) where {T<:Function, U<:Number}
  return Simplex(f, ic, initial_step .+ zeros(Bool, ic))
end

function Simplex(f::T, ic::AbstractVector{U}, initial_step::AbstractVector{V}
    ) where {T<:Function, U<:Number, V<:Number}
  if any(iszero.(initial_step))
    throw(ArgumentError("initial_step, $initial_step  must not have any zero
                        values"))
  end
  if length(ic) != length(initial_step)
    throw(ArgumentError("ic, $ic must be same length as initial_step
                        $initial_step"))
  end
  dim = length(ic)
  positions = Vector{Vector{promote_type(U,V)}}()
  for i ∈ 1:dim+1
    x = [ic[j] + ((j == i) ? initial_step[j] : zero(V)) for j ∈ 1:dim]
    push!(positions, x)
  end
  return Simplex(f, positions)
end
function Simplex(f::T, positions::U
    ) where {T<:Function, W<:Number, V<:AbstractVector{W}, U<:AbstractVector{V}}
  if length(unique(length.(positions))) != 1
    throw(ArgumentError("All entries in positions $positions must be the same
                        length"))
  end
  dim = length(positions) - 1
  vertex = Vertex(positions[1], f(positions[1]))
  vertices = Vector{typeof(vertex)}()
  push!(vertices, vertex)
  map(i->push!(vertices, Vertex(positions[i], f(positions[i]))), 2:dim+1)
  return Simplex(vertices)
end


import Base: length, iterate, push!, iterate, getindex
import Base: eachindex, sort!, hash
Base.length(s::Simplex) = length(s.vertices)
function Base.push!(s::Simplex, v::Vertex)
  push!(s.vertices, v)
  l = length(s.vertices)
  return nothing
end
Base.iterate(s::Simplex) = iterate(s.vertices)
Base.iterate(s::Simplex, counter) = iterate(s.vertices, counter)
Base.getindex(s::Simplex, index) = s.vertices[index]
function Base.sort!(s::Simplex)
  sort!(s.vertices, by=v->angle(value(v)))
  sortperm!(s.permabs, s.vertices, by=x->abs(value(x)))
  return nothing
end

Base.hash(s::Simplex) = hash(s, hash(:Simplex))
Base.hash(s::Simplex, h::UInt64) = hash(hash.(s), h)

dimensionality(s::Simplex) = length(s) - 1

remove!(s::Simplex, v::Vertex) = filter!(x -> !isequal(x, v), s.vertices)
remove!(s::Simplex, x::Vector) = deleteat!(s.vertices, x)

issortedbyangle(s::Simplex) = issorted(s, by=v->angle(value(v)))

selectabs(s, index) = s.vertices[s.permabs[index]]

bestvertex(s::Simplex) = selectabs(s, 1)
worstvertex(s::Simplex) = selectabs(s, length(s))
secondworstvertex(s::Simplex) = selectabs(s, length(s) - 1)

function centroidposition(s::Simplex, ignoredvertex=worstvertex(s))
  g(v) = isequal(v, ignoredvertex) ? zero(position(v)) : position(v)
  return mapreduce(g, +, s) / (length(s) - 1)
end

centre(s::Simplex) = mapreduce(v->position(v), +, s) / length(s)

function swap!(s::Simplex, this::Vertex, forthat::Vertex)
  before = deepcopy(s)
  @assert this ∈ s.vertices
  lengthbefore = length(s)
  remove!(s, this)
  @assert length(s) == lengthbefore - 1 "$(length(s)), $lengthbefore"
  push!(s, forthat)
  @assert forthat ∈ s.vertices
  sort!(s)
  @assert length(s) == lengthbefore
  return nothing
end

swapworst!(s::Simplex, forthis::Vertex) = swap!(s, worstvertex(s), forthis)

function closestomiddlevertex(s::Simplex)
  mid = mapreduce(position, +, s) ./ length(s)
  _, index = findmin(map(v->sum((position(v) - mid).^2), s))
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
    connectedto = Set{Int}()
    for (qi, q) ∈ enumerate(simplex)
      thisxtol = true
      for (i, (pv, pq)) ∈ enumerate(zip(position(v), position(q)))
        thisxtol &= isapprox(pv, pq, rtol=config[:xtol_rel][i],
                                     atol=config[:xtol_abs][i])
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
      all(position(v) .== position(q)) && return :XTOL_DEGENERATE_SIMPLEX
    end
  end
  allftol && return :FTOL_REACHED

  return :CONTINUE
end

function _πtoπ(ϕ)
  ϕ < -π && return _πtoπ(ϕ + 2π)
  ϕ > π && return _πtoπ(ϕ - 2π)
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


