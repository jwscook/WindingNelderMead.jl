using Random, Test, WindingNelderMead
using WindingNelderMead: Vertex, Simplex, windingnumber, windingangle
using WindingNelderMead: centre, assessconvergence, position, value
using WindingNelderMead: bestvertex, issortedbyangle

@testset "WindingNelderMead tests" begin

  Random.seed!(0)

  @testset "Vertices" begin
    v1 = Vertex([0.0, 1.0], 2.0 - 3.0im)
    v2 = Vertex([0.0, 1.0], 2.0 - 3.0im)
    @test !(v1 == v2)
    @test isequal(v1, v2)
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

    @testset "Simplices in a simplex eps apart are converged" begin
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

    @testset "Simplices in a chain eps apart are converged" begin
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

    @testset "Simplices vertices are in order" begin
      for _ ∈ 1:10
        dim = 2
        T = Float64
        U = ComplexF64
        v1 = Vertex(rand(dim), rand(U))
        v2 = Vertex(rand(dim), rand(U))
        v3 = Vertex(rand(dim), rand(U))
        s = Simplex([v1, v2, v3])
        p = s.perm
        @test abs(value(s[p[1]])) < abs(value(s[p[2]]))
        @test abs(value(s[p[2]])) < abs(value(s[p[3]]))
        @test issortedbyangle(s)
      end
    end

  end

  @testset "End-to-end tests roots" begin
    defaults = WindingNelderMead.convergenceconfig(2, Float64)
    @testset "Single root" begin
      totalits= 0
      for _ in 1:1000
        stopval=10.0^(-rand(3:14))
        root = rand(ComplexF64)
        objective(x) = (x[1] + im * x[2]) - root
        ics = rand(2)
        sizes = rand(2) / 10
        solution = WindingNelderMead.optimise(x -> x[1] + im * x[2] - root,
          ics, sizes, stopval=stopval, maxiters=10_000)
        (s, n, returncode, its) = solution
        totalits+= its
        if returncode == :STOPVAL_REACHED
          @test abs(value(bestvertex(s))) <= stopval #defaults[:stopval]
        elseif returncode == :XTOL_REACHED && n != 0
          @test isapprox(bestvertex(s).position[1], real(root),
                         rtol=defaults[:xtol_rel][1], atol=defualts[:xtol_abs][1])
          @test isapprox(bestvertex(s).position[2], imag(root),
                         rtol=defaults[:xtol_rel][2], atol=defaults[:xtol_abs][1])
        else
          @test false
        end
      end
    end
  end

  @testset "unittest 1" begin
    defaults = WindingNelderMead.convergenceconfig(2, Float64)
    (root, stopval, ics, sizes) = (0.9266219999309768 + 0.7854086758392804im, 1.0e-12, [0.1563639702460078, 0.8403436031429723], [0.0035266489474599094, 0.8706954808674581])
    history = []
    function objective(x::Vector)
      push!(history, x)
      return (x[1] + im * x[2]) - root
    end
    solution = WindingNelderMead.optimise(objective, ics, sizes, stopval=stopval)
    (s, n, returncode, its) = solution
    if returncode == :STOPVAL_REACHED
      @test abs(value(bestvertex(s))) <= stopval
    elseif returncode == :XTOL_REACHED && n != 0
      @test isapprox(bestvertex(s).position[1], real(root),
                     rtol=defaults[:xtol_rel][1], atol=defualts[:xtol_abs][1])
      @test isapprox(bestvertex(s).position[2], imag(root),
                     rtol=defaults[:xtol_rel][2], atol=defaults[:xtol_abs][2])
    else
      @test false
    end
  end

  @testset "check errors are caught" begin
    @test_throws ArgumentError WindingNelderMead.optimise(x->im, [0.0], [0.0])
    @test_throws ArgumentError WindingNelderMead.optimise(x->im, [0.0], [1.0,
                                                                         2.0])
  end

end

