#question 01
using DataFrames, CSV, HTTP, Optim, LinearAlgebra, Statistics, ForwardDiff, Distributions, GLM, Random, FreqTables, Test
# Loading the data
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS4-mixture/nlsw88t.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

# Preparing the data
X = Matrix{Float64}(df[:, [:age, :white, :collgrad]])
Z = Matrix{Float64}(df[:, [Symbol("elnwage$i") for i in 1:8]])
y = Vector{Int}(df.occ_code)

# Defining the number of choices and individuals
J = 8  
N = size(X, 1) 

# Function to compute choice probabilities
function choice_probs(θ::AbstractVector{T}, X::Matrix{Float64}, Z::Matrix{Float64}, J::Int) where T
    β = reshape(θ[1:end-1], size(X, 2), J-1)
    γ = θ[end]
    V = zeros(T, N, J)
    for j in 1:J
        if j < J
            V[:, j] = X * β[:, j] .+ γ * Z[:, j]
        else
            V[:, j] = γ * Z[:, j]  # Base alternative
        end
    end
    expV = exp.(V)
    return expV ./ sum(expV, dims=2)
end

# Log-likelihood function
function log_likelihood(θ::AbstractVector{T}, X::Matrix{Float64}, Z::Matrix{Float64}, y::Vector{Int}, J::Int, N::Int) where T
    probs = choice_probs(θ, X, Z, J)
    return -sum(log.(probs[CartesianIndex.(1:N, y)]))
end

#objective function
function create_objective(X::Matrix{Float64}, Z::Matrix{Float64}, y::Vector{Int}, J::Int, N::Int)
    return θ::AbstractVector -> log_likelihood(θ, X, Z, y, J, N)
end

# Initial values
β_init = zeros(size(X, 2), J-1)
γ_init = 0.0
θ_init = [vec(β_init); γ_init]

#objective function
obj_fun = create_objective(X, Z, y, J, N)

# Optimization
result = optimize(obj_fun, θ_init, BFGS(), Optim.Options(show_trace=true, iterations=1000))

#results
θ_hat = Optim.minimizer(result)
β_hat = reshape(θ_hat[1:end-1], size(X, 2), J-1)
γ_hat = θ_hat[end]

# Hessian and standard errors
H = ForwardDiff.hessian(obj_fun, θ_hat)
se = sqrt.(diag(inv(H)))

# results
println("Estimated β:")
display(β_hat)
println("\nEstimated γ: ", γ_hat)
println("\nStandard Errors:")
display(reshape(se[1:end-1], size(β_hat)))
println("SE for γ: ", se[end])

#question 02
# Extract the estimated γ
γ_hat = θ_hat[end]

# Print the estimated γ and its standard error
println("Estimated γ: ", γ_hat)
println("Standard Error of γ: ", se[end])

# Interpretation
println("\nInterpretation:")
println("γ represents the effect of log wage on the odds of choosing each occupation relative to the base category.")
println("A positive γ indicates that higher wages increase the odds of choosing each occupation.")

# Comparing with Problem Set 3 
γ_ps3 = -0.0942
println("\nComparison to Problem Set 3:")
println("γ from PS3: ", γ_ps3)
println("γ from current model: ", γ_hat)

if abs(γ_hat) > abs(γ_ps3)
    println("The current γ estimate has a larger magnitude, suggesting a stronger wage effect.")
else
    println("The current γ estimate has a smaller magnitude, suggesting a weaker wage effect.")
end

println("The current estimate may make more sense than the one from PS03")

#question 03-a
include("lgwt.jl")
#distribution
d = Normal(0,1) 

#quadrature nodes and weights for 7 grid points
nodes, weights = lgwt(7,-4,4)
# now compute the integral over the density and verify it's 1
sum(weights.*pdf.(d,nodes))
# now compute the expectation and verify it's 0
sum(weights.*nodes.*pdf.(d,nodes))

#distribution with mean =0 and standard deviation=2
d = Normal(0, 2) 
nodes, weights = lgwt(7, -5*2, 5*2)
integral_result = sum(weights .* (nodes.^2) .* pdf.(d, nodes))
println("Integral result: ", integral_result)
true_value = 4
println("True value: ", true_value)
println("Relative error: ", abs(integral_result - true_value) / true_value)


#question 03-b
#distribution with mean=0 and standard deviation=2
d = Normal(0,2)

#quadrature nodes and weights for 10 grid points
nodes, weights = lgwt(10,-5*2,5*2)

#computing the integral of x² times the pdf
integral_result = sum(weights .* (nodes.^2) .* pdf.(d,nodes))
println("Integral result: ", integral_result)

# comparing with true value (variance of N(0,2) is 4)
true_value = 4
println("True value: ", true_value)
println("Relative error: ", abs(integral_result - true_value) / true_value)

#question 03-c
# Set random seed for reproducibility
Random.seed!(123)

#the distribution
d = Normal(0, 2) 

#integration limits
a, b = -5 * 2, 5 * 2  # -5σ to 5σ

# Function for Monte Carlo integration
function monte_carlo_integral(f, a, b, D)
    X = rand(Uniform(a, b), D)
    return (b - a) * mean(f.(X))
end

#functions to integrate
f1(x) = x^2 * pdf(d, x)  # For ∫ x² f(x) dx
f2(x) = x * pdf(d, x)    # For ∫ x f(x) dx
f3(x) = pdf(d, x)        # For ∫ f(x) dx

#integrals with D = 1,000,000
D_large = 1_000_000
integral1 = monte_carlo_integral(f1, a, b, D_large)
integral2 = monte_carlo_integral(f2, a, b, D_large)
integral3 = monte_carlo_integral(f3, a, b, D_large)

println("Results with D = 1,000,000:")
println("∫ x² f(x) dx ≈ ", integral1, " (True value: 4)")
println("∫ x f(x) dx ≈ ", integral2, " (True value: 0)")
println("∫ f(x) dx ≈ ", integral3, " (True value: 1)")

#integrals with D = 1,000
D_small = 1_000
integral1_small = monte_carlo_integral(f1, a, b, D_small)
integral2_small = monte_carlo_integral(f2, a, b, D_small)
integral3_small = monte_carlo_integral(f3, a, b, D_small)

println("\nResults with D = 1,000:")
println("∫ x² f(x) dx ≈ ", integral1_small, " (True value: 4)")
println("∫ x f(x) dx ≈ ", integral2_small, " (True value: 0)")
println("∫ f(x) dx ≈ ", integral3_small, " (True value: 1)")

#Compute relative errors
error1_large = abs(integral1 - 4) / 4
error2_large = abs(integral2 - 0) / 1 
error3_large = abs(integral3 - 1) / 1

error1_small = abs(integral1_small - 4) / 4
error2_small = abs(integral2_small - 0) / 1
error3_small = abs(integral3_small - 1) / 1

println("\nRelative errors (D = 1,000,000):")
println("Error for ∫ x² f(x) dx: ", error1_large)
println("Error for ∫ x f(x) dx: ", error2_large)
println("Error for ∫ f(x) dx: ", error3_large)

println("\nRelative errors (D = 1,000):")
println("Error for ∫ x² f(x) dx: ", error1_small)
println("Error for ∫ x f(x) dx: ", error2_small)
println("Error for ∫ f(x) dx: ", error3_small)

println("\nComparison and Discussion:")
println("The Monte Carlo integration with 1,000,000 draws provides very accurate approximations for all three integrals")
println("compared to the results with 1,000 draws, which are less accurate but still reasonably close to the true values.")
println("The integral of the pdf (∫ f(x) dx) is particularly well-approximated, even with fewer draws.")
println("The accuracy of ∫ x f(x) dx ≈ 0 demonstrates that the method correctly captures the symmetry of the normal distribution.")
println("Increasing the number of draws from 1,000 to 1,000,000 significantly improves the accuracy of the approximations with Monte Carlo integration.")

#question 03-d
include("lgwt.jl")

#distribution
d = Normal(0, 2)  
a, b = -10, 10    

# Function to integrate
f(x) = x^2 * pdf(d, x)

# Quadrature integration
function quadrature_integral(n)
    nodes, weights = lgwt(n, a, b)
    sum(weights .* f.(nodes))
end

# Monte Carlo integration
function monte_carlo_integral(D)
    X = rand(Uniform(a, b), D)
    (b - a) * mean(f.(X))
end

# Compare methods
n_quad = 10  # number of quadrature points
D_mc = 1000  # number of Monte Carlo draws

quad_result = quadrature_integral(n_quad)
mc_result = monte_carlo_integral(D_mc)

println("Quadrature result (n=$n_quad): ", quad_result)
println("Monte Carlo result (D=$D_mc): ", mc_result)
println("True value: 4")

# Demonstrate similarity in formulas
println("\nSimilarity in formulas:")
println("Quadrature: ∫f(x)dx ≈ Σ wᵢ * f(xᵢ)")
println("Monte Carlo: ∫f(x)dx ≈ (b-a)/D * Σ f(Xᵢ)")

println("\nNote the similarity:")
println("Both methods approximate the integral as a weighted sum.")
println("In quadrature, weights (wᵢ) are predetermined by the method.")
println("In Monte Carlo, all weights are equal: (b-a)/D.")
println("Quadrature uses specific points (xᵢ), while Monte Carlo uses random points (Xᵢ).")
println("Increasing n in quadrature or D in Monte Carlo generally improves accuracy.")

#question 04
#Prepare the data
X = Matrix{Float64}(df[:, [:age, :white, :collgrad]])
Z = Matrix{Float64}(df[:, [Symbol("elnwage$i") for i in 1:8]])
y = Vector{Int}(df.occ_code)

#number of choices and individuals
J = 8  
N = size(X, 1) 

# Include the lgwt function for quadrature
include("lgwt.jl")

# Function to compute choice probabilities
function choice_probs(β::Matrix{Float64}, γ::Float64, X::Matrix{Float64}, Z::Matrix{Float64})
    N, K = size(X)
    V = X * β .+ γ .* (Z .- Z[:, end])
    expV = exp.(V)
    return expV ./ sum(expV, dims=2)
end

# Log-likelihood function with quadrature
function log_likelihood(θ::Vector{Float64}, X::Matrix{Float64}, Z::Matrix{Float64}, y::Vector{Int}, J::Int, N::Int, R::Int)
    K = size(X, 2)
    β = reshape(θ[1:K*(J-1)], K, J-1)
    μγ, σγ = θ[end-1:end]
    
    nodes, weights = lgwt(R, -4, 4)  # Adjust range as needed
    
    ll = 0.0
    for i in 1:N
        prob_i = 0.0
        for r in 1:R
            γ = μγ + σγ * nodes[r]
            probs = choice_probs(β, γ, X[i:i, :], Z[i:i, :])
            prob_i += weights[r] * probs[y[i]]
        end
        ll += log(prob_i)
    end
    
    return -ll  # Return negative log-likelihood for min
end

# Initial values
K = size(X, 2)
θ_init = [vec(zeros(K, J-1)); 0.0; 1.0]  # Initialize β to zeros, μγ to 0, σγ to 1

# Create the objective function
R = 7  #quadrature points
obj_fun = θ -> log_likelihood(θ, X, Z, y, J, N, R)

# Optimization
result = optimize(obj_fun, θ_init, BFGS(), Optim.Options(show_trace=true, iterations=1000))

#results
θ_hat = Optim.minimizer(result)
β_hat = reshape(θ_hat[1:K*(J-1)], K, J-1)
μγ_hat, σγ_hat = θ_hat[end-1:end]

#standard errors
H = ForwardDiff.hessian(obj_fun, θ_hat)
se = sqrt.(diag(inv(H)))

#results
println("Estimated β:")
display(β_hat)
println("\nEstimated μγ: ", μγ_hat)
println("Estimated σγ: ", σγ_hat)
println("\nStandard Errors:")
display(reshape(se[1:K*(J-1)], K, J-1))
println("SE for μγ: ", se[end-1])
println("SE for σγ: ", se[end])

#question 05
# Prepare the data
X = Matrix{Float64}(df[:, [:age, :white, :collgrad]])
Z = Matrix{Float64}(df[:, [Symbol("elnwage$i") for i in 1:8]])
y = Vector{Int}(df.occ_code)

# Define the number of choices and individuals
J = 8  
N = size(X, 1)  

# Function to compute choice probabilities
function choice_probs(β::Matrix{Float64}, γ::Float64, X::Matrix{Float64}, Z::Matrix{Float64})
    N, K = size(X)
    V = X * β .+ γ .* (Z .- Z[:, end])
    expV = exp.(V)
    return expV ./ sum(expV, dims=2)
end

# Log-likelihood function with Monte Carlo
function log_likelihood(θ::Vector{Float64}, X::Matrix{Float64}, Z::Matrix{Float64}, y::Vector{Int}, J::Int, N::Int, S::Int)
    K = size(X, 2)
    β = reshape(θ[1:K*(J-1)], K, J-1)
    μγ, σγ = θ[end-1:end]
    
    ll = 0.0
    for i in 1:N
        prob_i = 0.0
        for s in 1:S
            γ = μγ + σγ * randn()
            probs = choice_probs(β, γ, X[i:i, :], Z[i:i, :])
            prob_i += probs[y[i]]
        end
        ll += log(prob_i / S)
    end
    
    return -ll  # Return negative log-likelihood for min
end

# Initial values
K = size(X, 2)
θ_init = [vec(zeros(K, J-1)); 0.0; 1.0]  # Initialize β to zeros, μγ to 0, σγ to 1

# objective function
S = 100  # Number of Monte Carlo draws (you might want to increase this for final estimation)
obj_fun = θ -> log_likelihood(θ, X, Z, y, J, N, S)

# Optimization
result = optimize(obj_fun, θ_init, BFGS(), Optim.Options(show_trace=true, iterations=1000))

# results
θ_hat = Optim.minimizer(result)
β_hat = reshape(θ_hat[1:K*(J-1)], K, J-1)
μγ_hat, σγ_hat = θ_hat[end-1:end]

#standard errors
H = ForwardDiff.hessian(obj_fun, θ_hat)
se = sqrt.(diag(inv(H)))

# results
println("Estimated β:")
display(β_hat)
println("\nEstimated μγ: ", μγ_hat)
println("Estimated σγ: ", σγ_hat)
println("\nStandard Errors:")
display(reshape(se[1:K*(J-1)], K, J-1))
println("SE for μγ: ", se[end-1])
println("SE for σγ: ", se[end])

#question 06
include("lgwt.jl")
function allwrap(url::String, S::Int, R::Int, max_iterations::Int)
    # Set random seed for reproducibility
    Random.seed!(1234)

    # Load the data
    df = CSV.read(HTTP.get(url).body, DataFrame)

    # Prepare the data
    X = Matrix{Float64}(df[:, [:age, :white, :collgrad]])
    Z = Matrix{Float64}(df[:, [Symbol("elnwage$i") for i in 1:8]])
    y = Vector{Int}(df.occ_code)

    # Define the number of choices and individuals
    J = 8  # number of choices
    N = size(X, 1)  # number of individuals
    K = size(X, 2)  # number of covariates

    # Function to compute choice probabilities
    function choice_probs(β::AbstractMatrix{T}, γ::T, X::Matrix{Float64}, Z::Matrix{Float64}) where T
        N, K = size(X)
        V = zeros(T, N, J)
        for j in 1:J
            V[:, j] = X * β[:, j] .+ γ * (Z[:, j] .- Z[:, J])
        end
        expV = exp.(V)
        return expV ./ sum(expV, dims=2)
    end

    # Multinomial Logit Log-likelihood
    function mnl_loglikelihood(θ::AbstractVector{T}) where T
        β = reshape(θ[1:K*(J-1)], K, J-1)
        γ = θ[end]
        
        ll = zero(T)
        for i in 1:N
            probs = choice_probs(hcat(β, zeros(T, K, 1)), γ, X[i:i, :], Z[i:i, :])
            ll += log(probs[1, y[i]])
        end
        
        return -ll  # Return negative log-likelihood for minimization
    end

    # Mixed Logit Log-likelihood with Quadrature
    function mxl_quad_loglikelihood(θ::AbstractVector{T}) where T
        β = reshape(θ[1:K*(J-1)], K, J-1)
        μγ, σγ = θ[end-1:end]
        
        nodes, weights = lgwt(R, -4, 4)  # Assume lgwt is defined elsewhere
        
        ll = zero(T)
        for i in 1:N
            prob_i = zero(T)
            for r in 1:R
                γ = μγ + σγ * nodes[r]
                probs = choice_probs(hcat(β, zeros(T, K, 1)), γ, X[i:i, :], Z[i:i, :])
                prob_i += weights[r] * probs[1, y[i]]
            end
            ll += log(prob_i)
        end
        
        return -ll  # Return negative log-likelihood for minimization
    end

    # Mixed Logit Log-likelihood with Monte Carlo
    function mxl_mc_loglikelihood(θ::AbstractVector{T}) where T
        β = reshape(θ[1:K*(J-1)], K, J-1)
        μγ, σγ = θ[end-1:end]
        
        ll = zero(T)
        for i in 1:N
            prob_i = zero(T)
            for s in 1:S
                γ = μγ + σγ * randn()
                probs = choice_probs(hcat(β, zeros(T, K, 1)), γ, X[i:i, :], Z[i:i, :])
                prob_i += probs[1, y[i]]
            end
            ll += log(prob_i / S)
        end
        
        return -ll  # Return negative log-likelihood for minimization
    end

    # Function to estimate and print results
    function estimate_and_print(obj_fun, θ_init, model_name)
        result = optimize(obj_fun, θ_init, BFGS(), Optim.Options(show_trace=true, iterations=max_iterations))
        θ_hat = Optim.minimizer(result)
        β_hat = reshape(θ_hat[1:K*(J-1)], K, J-1)
        
        println("\n--- $model_name Results ---")
        println("Estimated β:")
        display(β_hat)
        
        if length(θ_hat) > K*(J-1) + 1
            μγ_hat, σγ_hat = θ_hat[end-1:end]
            println("\nEstimated μγ: ", μγ_hat)
            println("Estimated σγ: ", σγ_hat)
        else
            γ_hat = θ_hat[end]
            println("\nEstimated γ: ", γ_hat)
        end
        
        # Attempt to compute standard errors
        try
            H = ForwardDiff.hessian(obj_fun, θ_hat)
            se = sqrt.(diag(inv(H)))
            
            println("\nStandard Errors:")
            display(reshape(se[1:K*(J-1)], K, J-1))
            
            if length(θ_hat) > K*(J-1) + 1
                println("SE for μγ: ", se[end-1])
                println("SE for σγ: ", se[end])
            else
                println("SE for γ: ", se[end])
            end
        catch e
            if isa(e, SingularException)
                println("\nWarning: Unable to compute standard errors due to singular Hessian.")
                println("This may indicate issues with model convergence or multicollinearity.")
            else
                println("\nAn error occurred while computing standard errors: ", e)
            end
        end
        
        println("\nOptimization convergence: ", Optim.converged(result))
        println("Final log-likelihood: ", -Optim.minimum(result))
    end

    # Multinomial Logit Estimation
    θ_init_mnl = vcat(vec(zeros(K, J-1)), 0.0)
    estimate_and_print(mnl_loglikelihood, θ_init_mnl, "Multinomial Logit")

    # Mixed Logit Estimation with Quadrature
    θ_init_mxl = vcat(vec(zeros(K, J-1)), 0.0, 1.0)
    estimate_and_print(mxl_quad_loglikelihood, θ_init_mxl, "Mixed Logit with Quadrature")

    # Mixed Logit Estimation with Monte Carlo
    estimate_and_print(mxl_mc_loglikelihood, θ_init_mxl, "Mixed Logit with Monte Carlo")
end

# Call the function
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS4-mixture/nlsw88t.csv"
S = 100  # Number of Monte Carlo draws
R = 7    # Number of quadrature points
max_iterations = 1000  # Maximum number of iterations for optimization

allwrap(url, S, R, max_iterations)

println("\nAll estimations completed")

#question 07
include("hw04.jl")  

@testset "Mixed Logit Estimation Extended Tests" begin
    # Setup mock data
    Random.seed!(1234)
    N, K, J = 100, 3, 4
    X = randn(N, K)
    Z = randn(N, J)
    y = rand(1:J, N)
    R = 5  # Number of quadrature points
    S = 50  # Number of Monte Carlo draws

    @testset "Choice Probabilities" begin
        β = randn(K, J-1)
        γ = 0.5
        probs = choice_probs(β, γ, X, Z)
        
        @test size(probs) == (N, J)
        @test all(0 .<= probs .<= 1)
        @test all(isapprox.(sum(probs, dims=2), 1, atol=1e-6))
        
        # Test with extreme values
        β_extreme = fill(1000.0, K, J-1)
        probs_extreme = choice_probs(β_extreme, γ, X, Z)
        @test !any(isnan.(probs_extreme))
        @test all(isapprox.(sum(probs_extreme, dims=2), 1, atol=1e-6))
    end

    @testset "Multinomial Logit Log-Likelihood" begin
        θ = vcat(vec(randn(K, J-1)), 0.5)
        ll = mnl_loglikelihood(θ, X, Z, y, J, N)
        
        @test typeof(ll) == Float64
        @test ll < 0  # Log-likelihood should be negative
        
        # Test gradient
        g = ForwardDiff.gradient(θ -> mnl_loglikelihood(θ, X, Z, y, J, N), θ)
        @test length(g) == length(θ)
        @test all(isfinite.(g))
    end

    @testset "Mixed Logit Quadrature Log-Likelihood" begin
        θ = vcat(vec(randn(K, J-1)), 0.0, 1.0)
        ll = mxl_quad_loglikelihood(θ, X, Z, y, J, N, R)
        
        @test typeof(ll) == Float64
        @test ll < 0  # Log-likelihood should be negative
        
        # Test gradient
        g = ForwardDiff.gradient(θ -> mxl_quad_loglikelihood(θ, X, Z, y, J, N, R), θ)
        @test length(g) == length(θ)
        @test all(isfinite.(g))
    end

    @testset "Mixed Logit Monte Carlo Log-Likelihood" begin
        θ = vcat(vec(randn(K, J-1)), 0.0, 1.0)
        ll = mxl_mc_loglikelihood(θ, X, Z, y, J, N, S)
        
        @test typeof(ll) == Float64
        @test ll < 0  # Log-likelihood should be negative
        
        # Test gradient
        g = ForwardDiff.gradient(θ -> mxl_mc_loglikelihood(θ, X, Z, y, J, N, S), θ)
        @test length(g) == length(θ)
        @test all(isfinite.(g))
    end

    @testset "Gauss-Legendre Quadrature" begin
        nodes, weights = lgwt(5, -1, 1)
        @test length(nodes) == 5
        @test length(weights) == 5
        @test sum(weights) ≈ 2  # Integral of 1 over [-1, 1]
        @test sum(nodes .* weights) ≈ 0  # Integral of x over [-1, 1]
    end

    @testset "Simplified Estimation Process" begin
        # Multinomial Logit
        θ_init_mnl = vcat(vec(zeros(K, J-1)), 0.0)
        result_mnl = optimize(θ -> mnl_loglikelihood(θ, X, Z, y, J, N), θ_init_mnl, BFGS(), Optim.Options(iterations=100))
        
        @test Optim.converged(result_mnl)
        @test length(Optim.minimizer(result_mnl)) == length(θ_init_mnl)
        
        # Mixed Logit with Quadrature
        θ_init_mxl = vcat(vec(zeros(K, J-1)), 0.0, 1.0)
        result_quad = optimize(θ -> mxl_quad_loglikelihood(θ, X, Z, y, J, N, R), θ_init_mxl, BFGS(), Optim.Options(iterations=100))
        
        @test Optim.converged(result_quad)
        @test length(Optim.minimizer(result_quad)) == length(θ_init_mxl)
        
        # Mixed Logit with Monte Carlo
        result_mc = optimize(θ -> mxl_mc_loglikelihood(θ, X, Z, y, J, N, S), θ_init_mxl, BFGS(), Optim.Options(iterations=100))
        
        @test Optim.converged(result_mc)
        @test length(Optim.minimizer(result_mc)) == length(θ_init_mxl)
    end

    @testset "Consistency Checks" begin
        θ = vcat(vec(randn(K, J-1)), 0.0, 1.0)
        ll_quad = mxl_quad_loglikelihood(θ, X, Z, y, J, N, R)
        ll_mc = mxl_mc_loglikelihood(θ, X, Z, y, J, N, S)
        
        # Check if quadrature and Monte Carlo give similar results
        @test isapprox(ll_quad, ll_mc, rtol=0.1)
        
        # Check if mixed logit reduces to multinomial logit when σγ = 0
        θ_mnl = vcat(θ[1:end-2], 0.0, 0.0)
        ll_mnl = mnl_loglikelihood(θ_mnl[1:end-1], X, Z, y, J, N)
        ll_mxl_quad = mxl_quad_loglikelihood(θ_mnl, X, Z, y, J, N, R)
        @test isapprox(ll_mnl, ll_mxl_quad, rtol=1e-5)
    end
end

