using Pkg

Pkg.add("Optim")
Pkg.add("HTTP")
Pkg.add("GLM")
#question 01
using Optim
f(x) = -x[1]^4-10x[1]^3-2x[1]^2-3x[1]-2
negf(x) = x[1]^4+10x[1]^3+2x[1]^2+3x[1]+2
startval = rand(1) # random number as starting value
result = optimize(negf, startval, LBFGS())
println("argmin (minimizer) is ",Optim.minimizer(result)[1])
println("min is ",Optim.minimum(result))
#question 02
using DataFrames
using CSV
using HTTP
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)
X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.married.==1

function ols(beta, X, y)
    ssr = (y.-X*beta)'*(y.-X*beta)
    return ssr
end

beta_hat_ols = optimize(b -> ols(b, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
println(beta_hat_ols.minimizer)

using GLM
bols = inv(X'*X)*X'*y
df.white = df.race.==1
bols_lm = lm(@formula(married ~ age + white + collgrad), df)
#question 03
using Optim
using LinearAlgebra

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
using GLM
using DataFrames
using LinearAlgebra

# same example as before
n = 1000  # no of observations
p = 3     # no of predictors (including intercept)

X = hcat(ones(n), randn(n, p-1))  # matrix with intercept
β_true = [0.5, -1.0, 2.0]         # True coeff
y = rand(n) .< 1 ./ (1 .+ exp.(-X * β_true))  

# Creating a DataFrame
df = DataFrame(
    y = y,
    x1 = X[:, 2],
    x2 = X[:, 3]
)

# logistic regression model using GLM
glm_model = glm(@formula(y ~ x1 + x2), df, Binomial(), LogitLink())

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
using Pkg
Pkg.add("FreqTables")
using FreqTables
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