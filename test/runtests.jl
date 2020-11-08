using Random, Test, WindingNelderMead, LinearAlgebra
using WindingNelderMead: Vertex, Simplex, windingnumber, windingangle
using WindingNelderMead: centre, assessconvergence, position, value, bestvertex

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

    #@testset "Simplices with identical vertices is converged" begin
    #  dim = 2
    #  T = Float64
    #  U = ComplexF64
    #  pos = T[1.0, 1.0]
    #  val = rand(U)
    #  v1 = Vertex(pos, val)
    #  v2 = Vertex(pos, val)
    #  v3 = Vertex(pos, val)
    #  s = Simplex([v1, v2, v3])

    #  defaults = WindingNelderMead.convergenceconfig(dim, T)
    #  isconverged, returncode = assessconvergence(s, defaults)
    #  @test isconverged
    #  @test returncode == :XTOL_REACHED
    #end

    #@testset "Simplices in a simplex eps apart are converged" begin
    #  dim = 2
    #  T = Float64
    #  U = ComplexF64
    #  v1 = Vertex([one(T), one(T)], rand(U))
    #  v2 = Vertex([one(T), one(T) + eps(T)], rand(U))
    #  v3 = Vertex([one(T) + eps(T), one(T)], rand(U))
    #  s = Simplex([v1, v2, v3])
    #  defaults = WindingNelderMead.convergenceconfig(dim, T)
    #  isconverged, returncode = assessconvergence(s, defaults)
    #  @test isconverged
    #  @test returncode == :XTOL_REACHED
    #end

    #@testset "Simplices in a chain eps apart are converged" begin
    #  dim = 2
    #  T = Float64
    #  U = ComplexF64
    #  v1 = Vertex([one(T), one(T)], rand(U))
    #  v2 = Vertex([one(T), one(T) + eps(T)], rand(U))
    #  v3 = Vertex([one(T), one(T) + eps(T) + eps(T)], rand(U))
    #  s = Simplex([v1, v2, v3])
    #  defaults = WindingNelderMead.convergenceconfig(dim, T)
    #  isconverged, returncode = assessconvergence(s, defaults)
    #  @assert all(isapprox(position(v1)[d], position(v2)[d],
    #                    atol=defaults[:xtol_abs][d],
    #                    rtol=defaults[:xtol_rel][d]) for d ∈ 1:dim)
    #  @assert all(isapprox(position(v2)[d], position(v3)[d],
    #                    atol=defaults[:xtol_abs][d],
    #                    rtol=defaults[:xtol_rel][d]) for d ∈ 1:dim)
    #  @assert !all(isapprox(position(v1)[d], position(v3)[d],
    #                    atol=defaults[:xtol_abs][d],
    #                    rtol=defaults[:xtol_rel][d]) for d ∈ 1:dim)
    #  @test isconverged
    #  @test returncode == :XTOL_REACHED
    #end

  end

  @testset "End-to-end tests roots" begin

    defaults = WindingNelderMead.convergenceconfig(2, Float64)
    #isconverged, returncode = assessconvergence(s, defaults)
    @testset "Single root" begin
      for i in 1:10
        xtol_abs=10.0^(-rand(3:15))
        function mock(x::Vector, root)
          return (x[1] + im * x[2]) - root
        end
        root = rand(ComplexF64)
        objective(x) = mock(x, root)
        solution = WindingNelderMead.optimise(objective, rand(2), rand(2),
          xtol_abs=xtol_abs)
        (s, n, reason, its) = solution
        if reason == :STOPVAL_REACHED
          @test norm(value(bestvertex(s))) <= defaults[:stopval]
        elseif reason == :XTOL_REACHED && n != 0
          @test isapprox(bestvertex(s).position[1], real(root),
                         rtol=defaults[:xtol_rel][1], atol=xtol_abs)
          @test isapprox(bestvertex(s).position[2], imag(root),
                         rtol=defaults[:xtol_rel][2], atol=xtol_abs)
        else
          continue
        end
      end
    end
  end

  #@testset "check errors are caught" begin
  #  @test_throws ArgumentError WindingNelderMead.optimise(x->im, [0.0], [1.0],
  #                                                              [1, 2])
  #  @test_throws ArgumentError WindingNelderMead.optimise(x->im, [0.0], [1.0],
  #                                                              [0])
  #  @test_throws ArgumentError WindingNelderMead.optimise(x->im, [0.0], [0.0],
  #                                                              [1])
  #  @test_throws ArgumentError WindingNelderMead.optimise(x->im, [0.0, 0.0],
  #    [1.0, 1.0], [2, 2], xtol_abs=0.0, xtol_rel=0.0)
  #end

end

