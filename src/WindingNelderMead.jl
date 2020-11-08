module WindingNelderMead

using LinearAlgebra

include("Vertexes.jl")
include("Simplexes.jl")

"""
    optimise(f, initial_vertex_positions; kwargs...)

Find minimum of function, `f`, first creating a Simplex from vertices at
`initial_vertex_positions`, and options passed in via kwargs.
"""
function optimise(f::T, initial_vertex_positions::U; kwargs...
   ) where {T<:Function, W<:Number, V<:AbstractVector{W}, U<:AbstractVector{V}}
  return optimise(f, Simplex(f, initial_vertex_positions); kwargs...)
end

"""
    optimise(f, initial_positions, initial_step; kwargs...)

Find minimum of function, `f`, first creating a Simplex using a starting
vertex position, `initial_position`, and other vertices `initial_step` away from
that point in all directions, and options passed in via kwargs.
"""
function optimise(f::T, initial_position::AbstractVector{U},
               initial_step::AbstractVector{V}; kwargs...
               ) where {T<:Function, U<:Number, V<:Number}
  return optimise(f, Simplex(f, initial_position, initial_step); kwargs...)
end

function convergenceconfig(dim::Int, T::Type; kwargs...)
  kwargs = Dict(kwargs)
  timelimit = get(kwargs, :timelimit, Inf)
  xtol_abs = get(kwargs, :xtol_abs, zeros(real(T))) .* ones(Bool, dim)
  xtol_rel = get(kwargs, :xtol_rel, eps(real(T))) .* ones(Bool, dim)
  ftol_abs = get(kwargs, :ftol_abs, zero(real(T)))
  ftol_rel = get(kwargs, :ftol_rel, eps(real(T)))
  stopval = get(kwargs, :stopval, eps(real(T)))
  maxiters = get(kwargs, :maxiters, 1000)

  α = get(kwargs, :α, 1)
  β = get(kwargs, :β, 0.5)
  γ = get(kwargs, :γ, 2)
  δ = get(kwargs, :δ, 0.5)
  α >= 0 || error(ArgumentError("$α >= 0"))
  0 <= β < 1 || error(ArgumentError("0 <= $β < 1"))
  γ > 1 || error(ArgumentError("$γ > 1"))
  γ > α || error(ArgumentError("$γ > $α"))
  if any(iszero.(xtol_rel) .& iszero.(xtol_abs))
    throw(ArgumentError("xtol_rel .& xtol_abs must not contain zeros"))
  end
  return (timelimit=timelimit, xtol_abs=xtol_abs, xtol_rel=xtol_rel,
          ftol_abs=ftol_abs, ftol_rel=ftol_rel, stopval=stopval,
          maxiters=maxiters, α=α, β=β, γ=γ, δ=δ)
end


"""
    optimise(f, s; kwargs...)

Find minimum of function, `f`, starting from Simplex, `s`, with options
passed in via kwargs.

# Keyword Arguments
-  stopval (default sqrt(eps())): stopping criterion when function evaluates
equal to or less than stopval
-  xtol_abs (default zeros(T)) .* ones(Bool, dimensionality(s)): stop if
the vertices of simplex get within this absolute tolerance
-  xtol_rel (default eps(T)) .* ones(Bool, dimensionality(s)): stop if
the vertices of simplex get within this relative tolerance
-  ftol_abs (default zero(real(U))): stop if function evaluations at the
vertices are close to one another by this absolute tolerance
-  ftol_rel (default 1000eps(real(U))): stop if function evaluations at the
vertices are close to one another by this relative tolerance
-  maxiters (default 1000): maximum number of iterations of the Nelder Mead
algorithm
-  timelimit (default Inf): stop if it takes longer than this in seconds
-  α (default 1): Reflection factor
-  β (default 0.5): Contraction factor
-  γ (default 2): Expansion factor
-  δ (default 0.5): Shrinkage factor
"""
function optimise(f::F, s::Simplex{T,U}; kwargs...) where {F<:Function, T<:Real, U}

  config = convergenceconfig(dimensionality(s), T; kwargs...)

  reflect(this, other) = Vertex(newposition(this, config[:α], other), f)
  expand(this, other) = Vertex(newposition(this, -config[:γ], other), f)
  contract(this, other) = Vertex(newposition(this, -config[:β], other), f)
  shrink(this, other) = Vertex(newposition(this, config[:δ], other), f)

  function shrink!(s::Simplex)
    lengthbefore = length(s)
    best = bestvertex(s)
    newvertices = [shrink(best, v) for v ∈ s if !isequal(v, best)]
    remove!(s, findall(v->!isequal(v, best), s.vertices))
    map(nv->push!(s, nv), newvertices)
    sort!(s, by=v->norm(value(v)))
    @assert length(s) == lengthbefore
    return nothing
  end

  iters, totaltime = 0, 0.0
  returncode = assessconvergence(s, config)
  history = deepcopy(s.vertices)
  while returncode == :CONTINUE && totaltime < config[:timelimit]
    totaltime += @elapsed begin
      (iters += 1) < config[:maxiters] || break

      if windingnumber(s) == 0
        best = bestvertex(s)
        worst = worstvertex(s)
        secondworst = secondworstvertex(s)
        centroid = Vertex(centroidposition(s), f)
        reflected = reflect(centroid, worst)

        if any(h->isequal(h, reflected), history)
          returncode = :ENDLESS_NELDERMEAD_LOOP
          break
        end
        history .= circshift(history, 1)
        history[1] = reflected

        if best <= reflected < secondworst
          swapworst!(s, reflected)
        elseif reflected < best
          expanded = expand(centroid, reflected)
          expanded < reflected && swapworst!(s, expanded)
          expanded >= reflected && swapworst!(s, reflected)
        elseif secondworst <= reflected < worst
          contracted = contract(centroid, reflected)
          contracted <= reflected ? swapworst!(s, contracted) : shrink!(s)
        elseif reflected >= worst
          contracted = contract(centroid, worst)
          contracted < worst ? swapworst!(s, contracted) : shrink!(s)
        end
      else
        keeper = closestomiddlevertex(s)
        centroid = Vertex(centroidposition(s, keeper), f)
        any(isequal(centroid, v) for v in s) && continue
        for vertex ∈ s
          isequal(vertex, keeper) && continue
          swap!(s, vertex, centroid)
          windingnumber(s) == 0 || break
          swap!(s, centroid, vertex)
        end
      end
      returncode = assessconvergence(s, config)
    end
  end

  iters == config[:maxiters] && (returncode = :MAXITERS_REACHED)
  return s, windingnumber(s), returncode, iters
end # optimise

end
