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
   ) where {T, W<:Number, V<:AbstractVector{W}, U<:AbstractVector{V}}
  return optimise!(Simplex(f, initial_vertex_positions), f; kwargs...)
end

"""
    optimise(f, initial_positions, initial_step; kwargs...)

Find minimum of function, `f`, first creating a Simplex using a starting
vertex position, `initial_position`, and other vertices `initial_step` away from
that point in all directions, and options passed in via kwargs.
"""
function optimise(f::T, initial_position::AbstractVector{U}, initial_step;
                  kwargs...) where {T, U<:Number}
  return optimise(f, vertexpositions(initial_position, initial_step); kwargs...)
end

function convergenceconfig(dim::Int, T::Type; kwargs...)
  kwargs = Dict(kwargs)
  timelimit = get(kwargs, :timelimit, Inf)
  xtol_abs = get(kwargs, :xtol_abs, zero(real(T)))
  xtol_rel = get(kwargs, :xtol_rel, eps(real(T)))
  ftol_abs = get(kwargs, :ftol_abs, zero(real(T)))
  ftol_rel = get(kwargs, :ftol_rel, eps(real(T)))
  stopval = get(kwargs, :stopval, eps(real(T)))
  maxiters = get(kwargs, :maxiters, 1000)

  α = get(kwargs, :α, 1.0)
  β = get(kwargs, :β, 0.5)
  γ = get(kwargs, :γ, 2.0)
  δ = get(kwargs, :δ, 0.5)

  (α >= 0) || throw(ArgumentError("$α >= 0"))
  (0 <= β < 1) || throw(ArgumentError("0 <= $β < 1"))
  (γ > 1) || throw(ArgumentError("$γ > 1"))
  (γ > β) || throw(ArgumentError("$γ > $α"))

  vectoriser(x) = length(x) == 1 ? [x for _ ∈ 1:dim] : Vector(x)
  xtol_abs = vectoriser(xtol_abs)
  xtol_rel = vectoriser(xtol_rel)

  return (timelimit=timelimit, xtol_abs=xtol_abs, xtol_rel=xtol_rel,
          ftol_abs=ftol_abs, ftol_rel=ftol_rel, stopval=stopval,
          maxiters=maxiters, α=α, β=β, γ=γ, δ=δ)
end

function updatehistory!(history, newentry)
  returncode = any(h->isequal(h, newentry), history) ? :ENDLESS_LOOP : :CONTINUE
  for i in length(history):-1:2 # history .= circshift(history, 1)
    history[i] = history[i-1]
  end
  history[1] = newentry
  return returncode
end

"""
    optimise!(s, f; kwargs...)

Find minimum of function, `f`, starting from Simplex, `s`, with options
passed in via kwargs.

# Keyword Arguments
-  stopval (default sqrt(eps())): stopping criterion when function evaluates
equal to or less than stopval
-  xtol_abs (default zero(T) scalar or with length dimensionality(s)): stop if
the vertices of simplex get within this absolute tolerance
-  xtol_rel (default eps(T) scalar or with length dimensionality(s)): stop if
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
function optimise!(s::Simplex{D,T}, f::F; kwargs...) where {F,D,T<:Real}

  config = convergenceconfig(dimensionality(s), T; kwargs...)

  reflect(this, other) = Vertex(newposition(this, config[:α], other), f)
  expand(this, other) = Vertex(newposition(this, -config[:γ], other), f)
  contract(this, other) = Vertex(newposition(this, -config[:β], other), f)
  shrink(this, other) = Vertex(newposition(this, config[:δ], other), f)

  function shrink!(s::Simplex)
    ignore = bestvertex(s)
    for v ∈ s
      isequal(v, ignore) || swap!(s, v, shrink(ignore, v))
    end
    return nothing
  end

  iters, totaltime = 0, 0.0
  asg = AssessConvergenceGraph()
  returncode = assessconvergence(s, config, asg)
  history = deepcopy(s.vertices)

  windings = windingnumber(s)

  while returncode == :CONTINUE && totaltime < config[:timelimit]
    totaltime += @elapsed begin
      (iters += 1) < config[:maxiters] || break

      if windings == 0
        best = bestvertex(s)
        worst = worstvertex(s)
        secondworst = secondworstvertex(s, worst)
        reflected = reflect(centroidposition(s), worst)

        returncode = updatehistory!(history, reflected)
        returncode == :ENDLESS_LOOP && break

        if best <= reflected < secondworst
          swap!(s, worst, reflected)
        else
          centroid = Vertex(centroidposition(s), f)
          if reflected < best
            expanded = expand(centroid, reflected)
            expanded < reflected && swap!(s, worst, expanded)
            expanded >= reflected && swap!(s, worst, reflected)
          elseif secondworst <= reflected < worst
            contracted = contract(centroid, reflected)
            contracted <= reflected ? swap!(s, worst, contracted) : shrink!(s)
          elseif reflected >= worst
            contracted = contract(centroid, worst)
            contracted < worst ? swap!(s, worst, contracted) : shrink!(s)
          end
        end

        windings = windingnumber(s)
      else
        keeper = closestomiddlevertex(s)
        newnodeposition = centroidposition(s, keeper)
        any(isequal(newnodeposition, position(v)) for v in s) && continue
        centroid = Vertex(newnodeposition, f)
        rootlostandsimplexunchanged = true
        for vertex ∈ s
          isequal(vertex, keeper) && continue
          swap!(s, vertex, centroid)
          windingnumber(s) == 0 || (rootlostandsimplexunchanged = false; break)
          swap!(s, centroid, vertex)
        end

        returncode = updatehistory!(history, centroid)
        returncode == :ENDLESS_LOOP && break

        windings = if rootlostandsimplexunchanged
          worst = worstvertex(s)
          centroid < worst && swap!(s, worst, centroid)
          0
        else
          windingnumber(s)
        end
      end
      returncode = assessconvergence(s, config, asg)
    end
  end

  iters == config[:maxiters] && (returncode = :MAXITERS_REACHED)
  totaltime >= config[:timelimit] && (returncode = :TIMELIMIT_REACHED)

  return s, windingnumber(s), returncode, iters
end # optimise!

end
