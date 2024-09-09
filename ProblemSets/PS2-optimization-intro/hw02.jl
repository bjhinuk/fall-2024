using Optim, HTTP, GLM, DataFrames, CSV, LinearAlgebra, FreqTables, Random

#question 01
f(x) = -x[1]^4-10x[1]^3-2x[1]^2-3x[1]-2
negf(x) = x[1]^4+10x[1]^3+2x[1]^2+3x[1]+2
startval = rand(1) # random number as starting value
result = optimize(negf, startval, LBFGS())
println("argmin (minimizer) is ",Optim.minimizer(result)[1])
println("min is ",Optim.minimum(result))
#question 0url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)
X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.married.==1

function ols(beta, X, y)
    ssr = (y.-X*beta)'*(y.-X*beta)
    return ssr
end

beta_hat_ols = optimize(b -> ols(b, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
println(beta_hat_ols.minimizer)

bols = inv(X'*X)*X'*y
df.white = df.race.==1
bols_lm = lm(@formula(married ~ age + white + collgrad), df)
#question 03
# logistic function
logistic(x) = 1 / (1 + exp(-x))

# negative log-likelihood function for logistic regression
function neg_log_likelihood(β, X, y)
    η = X * β
    p = logistic.(η)
    -sum(y .* log.(p) + (1 .- y) .* log.(1 .- p))
end

# for example
n = 1000  # no of observations
p = 3     # no of predictors (including intercept)

X = hcat(ones(n), randn(n, p-1))  # with intercept
β_true = [0.5, -1.0, 2.0]         # True coeff
y = rand(n) .< logistic.(X * β_true)  

# guessing β
β_init = zeros(p)

# objective function for Optim
obj(β) = neg_log_likelihood(β, X, y)

# Using Optim to find MLE
result = optimize(obj, β_init, LBFGS())

# results
β_hat = Optim.minimizer(result)
min_nll = Optim.minimum(result)

println("Estimated coefficients: ", β_hat)
println("Minimum negative log-likelihood: ", min_nll)
println("Convergence: ", Optim.converged(result))
#question 04
# same example as before
n = 1000  # no of observations
p = 3     # no of predictors (including intercept)

X = hcat(ones(n), randn(n, p-1))  # matrix with intercept
β_true = [0.5, -1.0, 2.0]         # True coeff
y = rand(n) .< 1 ./ (1 .+ exp.(-X * β_true))  

# Creating a DataFrame
dfxy = DataFrame(
    y = y,
    x1 = X[:, 2],
    x2 = X[:, 3]
)

# logistic regression model using GLM
glm_model = glm(@formula(y ~ x1 + x2), dfxy, Binomial(), LogitLink())

# model summary
println(glm_model)

# coefficients
β_hat_glm = coef(glm_model)
println("\nEstimated coefficients (GLM):")
println("Intercept: ", β_hat_glm[1])
println("x1: ", β_hat_glm[2])
println("x2: ", β_hat_glm[3])

# log-likelihood
ll = loglikelihood(glm_model)
println("\nLog-likelihood: ", ll)
println("Negative log-likelihood: ", -ll)

# Comparing with true coefficients
println("\nTrue coefficients:")
println("Intercept: ", β_true[1])
println("x1: ", β_true[2])
println("x2: ", β_true[3])
#comparing
println("Coefficient comparison:")
println("Parameter   | Optim        | GLM")
println("-----------|--------------|------------")
println("Intercept  | $(round(β_hat[1], digits=6)) | $(round(β_hat_glm[1], digits=6))")
println("x1         | $(round(β_hat[2], digits=6)) | $(round(β_hat_glm[2], digits=6))")
println("x2         | $(round(β_hat[3], digits=6)) | $(round(β_hat_glm[3], digits=6))")

# Compare negative log-likelihood
nll_optim = min_nll
nll_glm = -loglikelihood(glm_model)

println("\nNegative Log-Likelihood comparison:")
println("Optim: $nll_optim")
println("GLM:   $nll_glm")
#question 05
freqtable(df, :occupation) # note small number of obs in some occupations
df = dropmissing(df, :occupation)
df[df.occupation.==10,:occupation] .= 9
df[df.occupation.==11,:occupation] .= 9
df[df.occupation.==12,:occupation] .= 9
df[df.occupation.==13,:occupation] .= 9
freqtable(df, :occupation) # problem solved

X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.occupation

function mlogit(alpha, X, d)

    # your turn

    return loglike
end
#question 06
function logistic_regression_comparison()
    # logistic function
    logistic(x) = 1 / (1 + exp(-x))

    # negative log-likelihood function for logistic regression
    function neg_log_likelihood(β, X, y)
        η = X * β
        p = logistic.(η)
        -sum(y .* log.(p) + (1 .- y) .* log.(1 .- p))
    end

    # example 
    Random.seed!(123) 
    n = 1000  # number of observations
    p = 3     # number of predictors (including intercept)

    X = hcat(ones(n), randn(n, p-1))  # matrix with intercept
    β_true = [0.5, -1.0, 2.0]         # True coeffs
    y = rand(n) .< logistic.(X * β_true)  

    # Optim approach
    # guess for β
    β_init = zeros(p)

    # objective function for Optim
    obj(β) = neg_log_likelihood(β, X, y)

    # Using Optim to find the MLE
    result = optimize(obj, β_init, LBFGS())

    # results
    β_hat_optim = Optim.minimizer(result)
    min_nll_optim = Optim.minimum(result)

    println("Optim Results:")
    println("Estimated coefficients: ", β_hat_optim)
    println("Minimum negative log-likelihood: ", min_nll_optim)
    println("Convergence: ", Optim.converged(result))
    println()

    # GLM approach
    # Creating a DataFrame
    dfxy = DataFrame(
        y = y,
        x1 = X[:, 2],
        x2 = X[:, 3]
    )

    # logistic regression model using GLM
    glm_model = glm(@formula(y ~ x1 + x2), dfxy, Binomial(), LogitLink())

    #coefficients and log-likelihood
    β_hat_glm = coef(glm_model)
    ll_glm = loglikelihood(glm_model)

    println("GLM Results:")
    println("Estimated coefficients:")
    println("  Intercept: ", β_hat_glm[1])
    println("  x1: ", β_hat_glm[2])
    println("  x2: ", β_hat_glm[3])
    println("Log-likelihood: ", ll_glm)
    println("Negative log-likelihood: ", -ll_glm)
    println()

    # Compare results
    println("Comparison:")
    println("Parameter   | Optim        | GLM")
    println("-----------|--------------|------------")
    println("Intercept  | $(round(β_hat_optim[1], digits=6)) | $(round(β_hat_glm[1], digits=6))")
    println("x1         | $(round(β_hat_optim[2], digits=6)) | $(round(β_hat_glm[2], digits=6))")
    println("x2         | $(round(β_hat_optim[3], digits=6)) | $(round(β_hat_glm[3], digits=6))")
    println()
    println("Negative Log-Likelihood:")
    println("Optim: $min_nll_optim")
    println("GLM:   $(-ll_glm)")
    
    rel_diff = abs(min_nll_optim - (-ll_glm)) / abs(-ll_glm)
    println("Relative difference: $(round(rel_diff * 100, digits=6))%")
    println()

    println("True coefficients:")
    println("Intercept: ", β_true[1])
    println("x1: ", β_true[2])
    println("x2: ", β_true[3])
end

# Call the function
logistic_regression_comparison()
#question 07
using Test
@testset "Logistic Regression Tests" begin
    @testset "Logistic Function" begin
        @test logistic(0) ≈ 0.5
        @test logistic(100) ≈ 1.0 atol=1e-6
        @test logistic(-100) ≈ 0.0 atol=1e-6
    end

    @testset "Negative Log-Likelihood" begin
        X = [1.0 0.5; 1.0 -0.5]
        y = [1.0, 0.0]
        β = [0.0, 0.0]
        @test neg_log_likelihood(β, X, y) ≈ 1.3862943611198906
    end

    @testset "Data Generation" begin
        Random.seed!(123)
        n, p = 1000, 3
        β_true = [0.5, -1.0, 2.0]
        X, y = generate_data(n, p, β_true)
        @test size(X) == (n, p)
        @test length(y) == n
        @test all(y .∈ Ref([0, 1]))
    end

    @testset "Optim Logistic Regression" begin
        Random.seed!(123)
        n, p = 1000, 3
        β_true = [0.5, -1.0, 2.0]
        X, y = generate_data(n, p, β_true)
        β_init = zeros(p)
        β_hat, min_nll = optim_logistic_regression(X, y, β_init)
        @test length(β_hat) == p
        @test all(isfinite.(β_hat))
        @test isfinite(min_nll)
    end

    @testset "GLM Logistic Regression" begin
        Random.seed!(123)
        n, p = 1000, 3
        β_true = [0.5, -1.0, 2.0]
        X, y = generate_data(n, p, β_true)
        β_hat, nll = glm_logistic_regression(X, y)
        @test length(β_hat) == p
        @test all(isfinite.(β_hat))
        @test isfinite(nll)
    end

    @testset "Comparison of Optim and GLM" begin
        Random.seed!(123)
        n, p = 1000, 3
        β_true = [0.5, -1.0, 2.0]
        X, y = generate_data(n, p, β_true)
        β_init = zeros(p)
        
        β_hat_optim, min_nll_optim = optim_logistic_regression(X, y, β_init)
        β_hat_glm, nll_glm = glm_logistic_regression(X, y)
        
        @test β_hat_optim ≈ β_hat_glm atol=1e-4
        @test min_nll_optim ≈ nll_glm atol=1e-4
    end
end
