using Random, Test, WindingNelderMead
using WindingNelderMead: Vertex, Simplex, windingnumber, windingangle
using WindingNelderMead: centre, assessconvergence, position, value
using WindingNelderMead: bestvertex, issortedbyangle, hypervolume

@testset "WindingNelderMead tests" begin

  Random.seed!(0)

  @testset "Vertices" begin
    v1 = Vertex([0.0, 1.0], 2.0 - 3.0im)
    v2 = Vertex([0.0, 1.0], 2.0 - 3.0im)
    @test !(v1 == v2)
    @test isequal(v1, v2)
    @test hash(v1) == hash(v2)
  end

  @testset "Simplex tests" begin

    irrelevant = [0.0, 0.0]
    @testset "Simplex encloses zero" begin
      v1 = Vertex(irrelevant, 1.0 - im)
      v2 = Vertex(irrelevant, 0.0 + im)
      v3 = Vertex(irrelevant, -1.0 - im)
      encloseszero = Simplex([v1, v2, v3])
      @test isapprox(1, abs(windingangle(encloseszero)) / (2π))
      @test windingnumber(encloseszero) == 1
    end

    @testset "Simplex doesn't enclose zero" begin
      v1 = Vertex(irrelevant, 1.0 + im)
      v2 = Vertex(irrelevant, 2.0 + im)
      v3 = Vertex(irrelevant, 1.0 + im * 2)
      doesntenclosezero = Simplex([v1, v2, v3])
      @test abs(windingangle(doesntenclosezero)) / (2π) < 1.0e-3
      @test windingnumber(doesntenclosezero) == 0
      v1 = Vertex(irrelevant, 1.0 + im)
      v2 = Vertex(irrelevant, 0.0 + 2*im)
      v3 = Vertex(irrelevant, -1.0 + im)
      doesntenclosezero = Simplex([v1, v2, v3])
      @test abs(windingangle(doesntenclosezero)) / (2π) < 1.0e-3
      @test windingnumber(doesntenclosezero) == 0
      v1 = Vertex(irrelevant, 1.0 + 0im)
      v2 = Vertex(irrelevant, 0.0 + im)
      v3 = Vertex(irrelevant, 1.0 + 0im)
      doesntenclosezero = Simplex([v1, v2, v3])
      @test abs(windingangle(doesntenclosezero)) / (2π) < 1.0e-3
      @test windingnumber(doesntenclosezero) == 0
    end

    @testset "Simplices with identical vertices is converged" begin
      dim = 2
      T = Float64
      U = ComplexF64
      pos = T[1.0, 1.0]
      val = rand(U)
      v1 = Vertex(pos, val)
      v2 = Vertex(pos, val)
      v3 = Vertex(pos, val)
      s = Simplex([v1, v2, v3])

      defaults = WindingNelderMead.convergenceconfig(dim, T)
      returncode = assessconvergence(s, defaults)
      @test returncode == :XTOL_REACHED
    end

    @testset "Vertices in a simplex eps apart are converged" begin
      dim = 2
      T = Float64
      U = ComplexF64
      v1 = Vertex([one(T), one(T)], rand(U))
      v2 = Vertex([one(T), one(T) + eps(T)], rand(U))
      v3 = Vertex([one(T) + eps(T), one(T)], rand(U))
      s = Simplex([v1, v2, v3])
      defaults = WindingNelderMead.convergenceconfig(dim, T)
      returncode = assessconvergence(s, defaults)
      @test returncode == :XTOL_REACHED
    end

    @testset "Vertices in a chain eps apart are converged" begin
      dim = 2
      T = Float64
      U = ComplexF64
      v1 = Vertex([one(T), one(T)], rand(U))
      v2 = Vertex([one(T), one(T) + eps(T)], rand(U))
      v3 = Vertex([one(T), one(T) + eps(T) + eps(T)], rand(U))
      s = Simplex([v1, v2, v3])
      defaults = WindingNelderMead.convergenceconfig(dim, T)
      returncode = assessconvergence(s, defaults)
      @test returncode == :XTOL_REACHED
      @assert all(isapprox(position(v1)[d], position(v2)[d],
                        atol=defaults[:xtol_abs][d],
                        rtol=defaults[:xtol_rel][d]) for d ∈ 1:dim)
      @assert all(isapprox(position(v2)[d], position(v3)[d],
                        atol=defaults[:xtol_abs][d],
                        rtol=defaults[:xtol_rel][d]) for d ∈ 1:dim)
      @assert !all(isapprox(position(v1)[d], position(v3)[d],
                        atol=defaults[:xtol_abs][d],
                        rtol=defaults[:xtol_rel][d]) for d ∈ 1:dim)
      @test returncode == :XTOL_REACHED
    end

    @testset "Simplex vertices are in order" begin
      for _ ∈ 1:10
        dim = 2
        T = Float64
        U = ComplexF64
        v1 = Vertex(rand(dim), rand(U))
        v2 = Vertex(rand(dim), rand(U))
        v3 = Vertex(rand(dim), rand(U))
        s = Simplex([v1, v2, v3])
        p = s.permabs
        @test abs(value(s[p[1]])) < abs(value(s[p[2]]))
        @test abs(value(s[p[2]])) < abs(value(s[p[3]]))
        @test issortedbyangle(s)
      end
    end

    @testset "Simplex centres" begin
      v1 = Vertex([0.0, 0.0], one(ComplexF64))
      v2 = Vertex([1.0, 0.0], one(ComplexF64))
      v3 = Vertex([0.0, 1.0], one(ComplexF64))
      s = Simplex([v1, v2, v3])
      @test all(centre(s) .== [1/3, 1/3])
    end

    @testset "Simplex extremas" begin
      s = Simplex(x->im, [1.0, 3.0], 1.0)
      exs = WindingNelderMead.extrema(s)
      @test exs[1] == (1.0, 2.0)
      @test exs[2] == (3.0, 4.0)
    end

    @testset "Simplex hypervolumes" begin
      x0 = rand(2)
      a, b = rand(2)
      v1 = Vertex(x0 .+ [0.0, 0.0], one(ComplexF64))
      v2 = Vertex(x0 .+ [a, 0.0], one(ComplexF64))
      v3 = Vertex(x0 .+ [0.0, b], one(ComplexF64))
      s = Simplex([v1, v2, v3])
      @test hypervolume(s) ≈ (a * b) / 2
      x0 = rand(3)
      a, b, c = rand(3)
      v1 = Vertex(x0 .+ [0, 0, 0], one(ComplexF64))
      v2 = Vertex(x0 .+ [a, 0, 0], one(ComplexF64))
      v3 = Vertex(x0 .+ [0, b, 0], one(ComplexF64))
      v4 = Vertex(x0 .+ [0, 0, c], one(ComplexF64))
      s = Simplex([v1, v2, v3, v4])
      @test hypervolume(s) ≈ (a * b * c) / 6
    end

    @testset "atan gives same as angle for sortby" begin
      v1 = Vertex([0.0, 0.0], rand(ComplexF64))
      v2 = Vertex([1.0, 0.0], rand(ComplexF64))
      v3 = Vertex([0.0, 1.0], rand(ComplexF64))
      s = Simplex([v1, v2, v3])
      c = centre(s)
      for v in s
        pv = position(v)
        a = angle(Complex(pv[1] - c[1], pv[2] - c[2]))
        @test a == WindingNelderMead.sortby(s)(v)
      end
    end

    @testset "root (pole) has winding number 1 (-1) in 2D" begin
      objective(x) = (x[1] + im * x[2])
      ps = ([-1.0, -1.0], [1.0, -1.0], [0.0, 1.0])
      s = Simplex([Vertex(p, objective(p)) for p in ps])
      @test windingnumber(s) == 1
      invobjective(x) = 1 / objective(x)
      s = Simplex([Vertex(p, invobjective(p)) for p in ps])
      @test windingnumber(s) == -1
    end

  end

  @testset "End-to-end tests roots" begin
    defaults = WindingNelderMead.convergenceconfig(2, Float64)
    @testset "Single root" begin
      for _ in 1:10
        stopval=10.0^(-rand(3:14))
        root = rand(ComplexF64)
        objective(x) = (x[1] + im * x[2]) - root
        ics = rand(2)
        sizes = rand(2)
        solution = WindingNelderMead.optimise(objective,
          ics, sizes, stopval=stopval, maxiters=10_000)
        (s, n, returncode, its) = solution
        @test n == 1
        if returncode == :STOPVAL_REACHED
          @test abs(value(bestvertex(s))) <= stopval #defaults[:stopval]
        elseif returncode == :XTOL_REACHED && n != 0
          @test isapprox(bestvertex(s).position[1], real(root),
                         rtol=defaults[:xtol_rel][1], atol=defualts[:xtol_abs][1])
          @test isapprox(bestvertex(s).position[2], imag(root),
                         rtol=defaults[:xtol_rel][2], atol=defaults[:xtol_abs][1])
        else
          @show stopval, root, ics, sizes, returncode
          @test false
        end
      end
    end
  end

  @testset "check errors are caught" begin
    @test_throws ArgumentError WindingNelderMead.optimise(x->im, [0.0], [0.0])
    @test_throws ArgumentError WindingNelderMead.optimise(x->im, [0.0], [1.0,
                                                                         2.0])
    @test_throws ArgumentError WindingNelderMead.optimise(x->im, [[0],[1, 2]])
  end

end

