using Random, LinearAlgebra, Statistics, CSV, DataFrames, FreqTables, Distributions, JLD

# Question 1
function q1()
    Random.seed!(1234)

    # (a) Create matrices
    A = rand(Uniform(-5, 10), 10, 7)
    B = rand(Normal(-2, 15), 10, 7)
    C = [A[1:5, 1:5] B[1:5, 6:7]]
    D = copy(A)
    D[D .> 0] .= 0

    # (b) Number of elements in A
    println("Number of elements in A: ", length(A))

    # (c) Number of unique elements in D
    println("Number of unique elements in D: ", length(unique(D)))

    # (d) Create E (vec of B)
    E = vec(B)

    # (e) Create 3D array F
    F = cat(A, B, dims=3)

    # (f) Twist F
    F = permutedims(F, (3, 1, 2))

    # (g) Kronecker product
    G = kron(B, C)

    # (h) Save matrices to .jld file
    save("matrixpractice.jld", "A", A, "B", B, "C", C, "D", D, "E", E, "F", F, "G", G)

    # (i) Save subset of matrices to .jld file
    save("firstmatrix.jld", "A", A, "B", B, "C", C, "D", D)

    # (j) Export C as .csv
    CSV.write("Cmatrix.csv", DataFrame(C, :auto))

    # (k) Export D as tab-delimited .dat
    CSV.write("Dmatrix.dat", DataFrame(D, :auto), delim='\t')

    return A, B, C, D
end

# Question 2
function q2(A, B, C)
    # (a) Element-wise product of A and B
    AB = A .* B
    AB2 = A .* B  # Without loop

    # (b) Elements of C between -5 and 5
    Cprime = [c for c in C if -5 <= c <= 5]
    Cprime2 = filter(x -> -5 <= x <= 5, vec(C))

    # (c) Create 3D array X
    N, K, T = 15169, 6, 5
    X = Array{Float64}(undef, N, K, T)
    for t in 1:T
        X[:, 1, t] .= 1  # Intercept
        X[:, 2, t] = rand(Bernoulli(0.75 * (6 - t) / 5), N)
        X[:, 3, t] = rand(Normal(15 + t - 1, 5 * (t - 1)), N)
        X[:, 4, t] = rand(Normal(π * (6 - t) / 3, 1/ℯ), N)
        X[:, 5, t] = rand(Binomial(20, 0.6), N)
        X[:, 6, t] = rand(Binomial(20, 0.5), N)
    end

    # (d) Create β
    β = [
        [1 + 0.25*(t-1) for t in 1:T]';
        [log(t) for t in 1:T]';
        [-sqrt(t) for t in 1:T]';
        [exp(t) - exp(t+1) for t in 1:T]';
        [t for t in 1:T]';
        [t/3 for t in 1:T]'
    ]

    # (e) Create Y
    ε = rand(Normal(0, 0.36), N, T)
    Y = [sum(X[n, :, t] .* β[:, t]) + ε[n, t] for n in 1:N, t in 1:T]
end

# Question 3
function q3()
    # (a) Import and process data
    nlsw88 = CSV.read("nlsw88.csv", DataFrame)
    nlsw88[!, :wage] = coalesce.(nlsw88[!, :wage], missing)
    CSV.write("nlsw88_processed.csv", nlsw88)

    # (b) Percentage never married and college graduates
    pct_never_married = mean(nlsw88.never_married) * 100
    pct_college_grad = mean(nlsw88.collgrad == 4) * 100
    println("Percentage never married: ", pct_never_married)
    println("Percentage college graduates: ", pct_college_grad)

    # (c) Race distribution
    race_dist = freqtable(nlsw88.race)
    println("Race distribution: ", prop(race_dist))

    # (d) Summary statistics
    summarystats = describe(nlsw88)
    println("Number of missing grade observations: ", sum(ismissing.(nlsw88.grade)))

    # (e) Joint distribution of industry and occupation
    joint_dist = freqtable(nlsw88.industry, nlsw88.occupation)
    println("Joint distribution of industry and occupation: ", prop(joint_dist))

    # (f) Mean wage by industry and occupation
    wage_by_ind_occ = combine(groupby(nlsw88, [:industry, :occupation]), :wage => mean)
    println("Mean wage by industry and occupation: ", wage_by_ind_occ)
end

# Question 4
function matrixops(A, B)
    if size(A) != size(B)
        error("inputs must have the same size")
    end
    
    element_product = A .* B
    matrix_product = A' * B
    sum_elements = sum(A + B)
    
    return element_product, matrix_product, sum_elements
end

function q4()
    # Load matrices
    data = load("firstmatrix.jld")
    A, B = data["A"], data["B"]

    # (d) Evaluate matrixops with A and B
    result_AB = matrixops(A, B)
    println("Result of matrixops(A, B): ", result_AB)

    # (f) Evaluate matrixops with C and D
    C, D = data["C"], data["D"]
    try
        result_CD = matrixops(C, D)
        println("Result of matrixops(C, D): ", result_CD)
    catch e
        println("Error: ", e)
    end

    # (g) Evaluate matrixops with ttl_exp and wage
    nlsw88 = CSV.read("nlsw88_processed.csv", DataFrame)
    ttl_exp = convert(Array, coalesce.(nlsw88.ttl_exp, 0))
    wage = convert(Array, coalesce.(nlsw88.wage, 0))
    result_exp_wage = matrixops(ttl_exp, wage)
    println("Result of matrixops(ttl_exp, wage): ", result_exp_wage)
end

# Run all functions
A, B, C, D = q1()
q2(A, B, C)
q3()
q4()

# Question 5: Unit tests
using Test

@testset "Problem Set Tests" begin
    @test length(q1()) == 4  # q1 should return 4 matrices
    
    @test size(q2(rand(10,7), rand(10,7), rand(5,7))[1]) == (10, 7)  # AB should be 10x7
    
    @test isa(q3(), Nothing)  # q3 should return nothing
    
    @test length(matrixops(rand(3,3), rand(3,3))) == 3  # matrixops should return 3 items
end