![CI](https://github.com/jwscook/WindingNelderMead.jl/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/jwscook/WindingNelderMead.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/jwscook/WindingNelderMead.jl)
[![DOI](https://zenodo.org/badge/312883506.svg)](https://zenodo.org/badge/latestdoi/312883506)


# WindingNelderMead.jl

Find the roots of a complex function, or even a function that returns two numbers converted into one that returns a complex number.

The following code

```julia
using WindingNelderMead, Plots
using WindingNelderMead: Vertex, Simplex, windingnumber, windingangle
using WindingNelderMead: centre, assessconvergence, position, value
using WindingNelderMead: bestvertex, issortedbyangle, hypervolume

function run()
  defaults = WindingNelderMead.convergenceconfig(2, Float64)
  positions = Vector{Float64}[]
  values = Vector{Float64}[]
  windingnumbers = Int[]
  initial_vertex_positions = [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]]
  objective(x) = (x[1] + im * x[2]) - (exp(1) + im * sqrt(2))
  simplex = Simplex(objective, initial_vertex_positions)
  simplices = [deepcopy(simplex)]
  function wrappedobjective(x)
    push!(positions, x)
    push!(windingnumbers, windingnumber(simplex))
    output = objective(x)
    push!(values, [reim(output)...])
    push!(simplices, simplex)
    return output
  end
  ics = rand(2)
  sizes = rand(2)
  solution = WindingNelderMead.optimise!(simplex, wrappedobjective)
  (s, n, returncode, its) = solution
  return positions, values, windingnumbers
end

positions, values, windingnumbers = run()

h = plot(layout=@layout [a b; c])
title!(h[1], "Input: complex x")
xlabel!(h[1], "Real x")
ylabel!(h[1], "Imag x")
title!(h[2], "Output: complex v")
xlabel!(h[2], "Real v")
ylabel!(h[2], "Imag v")
xlabel!(h[3], "Iteration")
ylabel!(h[3], "Log10 |v|")
anim = @animate for (i, w) in enumerate(windingnumbers)
  x, y = positions[i]
  rv, iv = values[i]
  c = iszero(w) ? :blue : :red
  scatter!(h[1], [x], [y], mc=c, label=nothing,
           xlims=(-0.2, 4.2), ylims=(-0.2,2.2))
  scatter!(h[2], [rv], [iv], mc=c, label=nothing,
           xlims=(-3.2, 2.2), ylims=(-2.2,1.2))
  scatter!(h[3], [i], [log10(abs(rv + im * iv))], mc=c, label=nothing,
           xlims=(0, length(values) + 1), ylims=(-18,2))
  if i == 1
    scatter!(h[3], [-10], [0], mc=:blue, label="Nelder-Mead", legend=true)
    scatter!(h[3], [-10], [0], mc=:red, label="Winding", legend=true)
  else
    x_, y_ = positions[i-1]
    rv_, iv_ = values[i-1]
    plot!(h[1], [x_, x], [y_, y], lc=c, label=nothing)
    plot!(h[2], [rv_, rv], [iv_, iv], lc=c, label=nothing)
  end
end

gif(anim, "WindingNelderMead.gif", fps = 10)
```
results in

![WindingNelderMead](https://github.com/jwscook/WindingNelderMead.jl/assets/15519866/b6b1c252-8dac-4a76-a14f-5267d9fb1597)

Also, see the tests for more examples.
