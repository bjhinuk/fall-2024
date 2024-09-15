#question 01
using Optim, LinearAlgebra, Statistics, DataFrames, CSV, HTTP, ForwardDiff

# Load the data
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS3-gev/nlsw88w.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

# Prepare the data
X = [df.age df.white df.collgrad]
Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4,
         df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
y = df.occupation

# Function to compute choice probabilities
function choice_probs(β, X, Z)
    J = size(Z, 2)
    N = size(X, 1)
    
    # Separate β into coefficients for X and Z
    β_X = β[1:end-1]
    γ = β[end]
    
    # Compute utilities
    V = X * reshape(β_X, :, J-1) .+ γ * (Z[:, 1:J-1] .- Z[:, J])
    
    # Compute probabilities
    P = exp.(V) ./ (1 .+ sum(exp.(V), dims=2))
    P = hcat(P, 1 .- sum(P, dims=2))
    
    return P
end

# Log-likelihood function
function log_likelihood(β, X, Z, y)
    P = choice_probs(β, X, Z)
    ll = 0.0
    for i in 1:size(X, 1)
        # Add a small constant to prevent log of zero
        ll += log(max(P[i, y[i]], 1e-10))
    end
    return -ll  # Negative log-likelihood for minimization
end

# Set up the optimization
J = size(Z, 2)
K = size(X, 2)
β_init = vcat(zeros(K * (J-1)), 0.1)  # Initial guess for β

# Optimize
result = optimize(β -> log_likelihood(β, X, Z, y),
                  β_init,
                  BFGS(),
                  Optim.Options(show_trace = true))

# Extract and print results
β_hat = Optim.minimizer(result)
println("Estimated coefficients:")
for j in 1:J-1
    println("β$j: ", β_hat[(j-1)*K+1:j*K])
end
println("γ: ", β_hat[end])

# Compute standard errors
H = ForwardDiff.hessian(β -> log_likelihood(β, X, Z, y), β_hat)
se = sqrt.(diag(inv(H)))
println("\nStandard errors:")
for j in 1:J-1
    println("SE(β$j): ", se[(j-1)*K+1:j*K])
end
println("SE(γ): ", se[end])

#question 02
function interpret_gamma(γ, Z, β_hat, X)
    # Calculate the average wage across all occupations
    avg_wage = mean(exp.(Z))
    
    # Calculate the percentage change in odds for a 1% increase in wage
    pct_change = (exp(0.01 * γ) - 1) * 100
    
    # Calculate the elasticity
    avg_prob = mean(1 .- choice_probs(β_hat, X, Z)[:, end])
    elasticity = γ * avg_prob
    
    println("Interpretation of γ coefficient:")
    println("γ = ", round(γ, digits=4))
    println("Average wage across all occupations: \$$(round(avg_wage, digits=2))")
    println("A 1% increase in the wage of an occupation relative to the base category")
    println("is associated with a $(round(pct_change, digits=2))% increase in the odds")
    println("of choosing that occupation over the base category.")
    println("The elasticity of the choice probability with respect to wage is $(round(elasticity, digits=4))")
end

# Assuming β_hat is the estimated coefficient vector from Question 1
# and γ is the last element of β_hat
γ = β_hat[end]

# Call the interpretation function
interpret_gamma(γ, Z, β_hat, X)

#question 03
# Prepare the data
X = [df.age df.white df.collgrad]
Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4,
         df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
y = df.occupation

# Define nests
WC = [1, 2, 3]  # White collar
BC = [4, 5, 6, 7]  # Blue collar
Other = [8]  # Other

# Function to compute nested logit probabilities
function nested_logit_probs(β, X, Z)
    β_WC, β_BC, λ_WC, λ_BC, γ = β[1:3], β[4:6], β[7], β[8], β[9]
    J = size(Z, 2)
    N = size(X, 1)
    
    P = zeros(N, J)
    
    for i in 1:N
        # Compute nest-specific terms
        V_WC = exp.(X[i,:]' * β_WC ./ λ_WC) .* exp.(γ * (Z[i, WC] .- Z[i, Other[1]]) ./ λ_WC)
        V_BC = exp.(X[i,:]' * β_BC ./ λ_BC) .* exp.(γ * (Z[i, BC] .- Z[i, Other[1]]) ./ λ_BC)
        
        # Compute inclusive values
        IV_WC = sum(V_WC)^λ_WC
        IV_BC = sum(V_BC)^λ_BC
        
        # Compute probabilities
        denom = 1 + IV_WC + IV_BC
        
        P[i, WC] = (IV_WC / denom) .* (V_WC ./ sum(V_WC))
        P[i, BC] = (IV_BC / denom) .* (V_BC ./ sum(V_BC))
        P[i, Other] .= 1 / denom
    end
    
    return P
end

# Log-likelihood function
function log_likelihood(β, X, Z, y)
    P = nested_logit_probs(β, X, Z)
    ll = sum(log.(P[i, y[i]] + 1e-10) for i in 1:size(X, 1))
    return -ll  # Negative log-likelihood for minimization
end

# Function to compute gradient numerically
function numerical_gradient(f, x, ε=1e-5)
    n = length(x)
    grad = zeros(n)
    for i in 1:n
        x_plus = copy(x)
        x_minus = copy(x)
        x_plus[i] += ε
        x_minus[i] -= ε
        grad[i] = (f(x_plus) - f(x_minus)) / (2ε)
    end
    return grad
end

# Set up the optimization
K = size(X, 2)
β_init = [zeros(2*K); ones(2); 0.1]  # Initial guess for β

# Optimize
result = optimize(β -> log_likelihood(β, X, Z, y),
                  β_init,
                  BFGS(),
                  Optim.Options(show_trace = true))

# Extract and print results
β_hat = Optim.minimizer(result)
println("Estimated coefficients:")
println("β_WC: ", β_hat[1:3])
println("β_BC: ", β_hat[4:6])
println("λ_WC: ", β_hat[7])
println("λ_BC: ", β_hat[8])
println("γ: ", β_hat[9])

# Compute standard errors using numerical Hessian
function numerical_hessian(f, x, ε=1e-5)
    n = length(x)
    H = zeros(n, n)
    for i in 1:n
        for j in 1:n
            x_pp = copy(x)
            x_pm = copy(x)
            x_mp = copy(x)
            x_mm = copy(x)
            x_pp[i] += ε; x_pp[j] += ε
            x_pm[i] += ε; x_pm[j] -= ε
            x_mp[i] -= ε; x_mp[j] += ε
            x_mm[i] -= ε; x_mm[j] -= ε
            H[i,j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4ε^2)
        end
    end
    return H
end

H = numerical_hessian(β -> log_likelihood(β, X, Z, y), β_hat)
se = sqrt.(diag(inv(H)))
println("\nStandard errors:")
println("SE(β_WC): ", se[1:3])
println("SE(β_BC): ", se[4:6])
println("SE(λ_WC): ", se[7])
println("SE(λ_BC): ", se[8])
println("SE(γ): ", se[9])

# Function to interpret results
function interpret_results(β_hat, se)
    println("\nInterpretation of results:")
    println("1. Nest coefficients (β_WC and β_BC):")
    println("   These represent the effect of individual characteristics on the likelihood")
    println("   of choosing occupations in the White Collar and Blue Collar nests, respectively.")
    println("   Positive coefficients indicate increased likelihood, negative decreased likelihood.")
    
    println("\n2. Inclusive Value Parameters (λ_WC and λ_BC):")
    println("   λ_WC = ", round(β_hat[7], digits=4), " (SE: ", round(se[7], digits=4), ")")
    println("   λ_BC = ", round(β_hat[8], digits=4), " (SE: ", round(se[8], digits=4), ")")
    println("   These measure the degree of independence within each nest.")
    println("   Values closer to 1 indicate more independence, closer to 0 more correlation.")
    
    println("\n3. Wage coefficient (γ):")
    println("   γ = ", round(β_hat[9], digits=4), " (SE: ", round(se[9], digits=4), ")")
    println("   This represents the effect of wage differences on occupational choice.")
    println("   A positive coefficient indicates that higher wages increase the likelihood")
    println("   of choosing an occupation, all else being equal.")
end

# Call the interpretation function
interpret_results(β_hat, se)

#question 04
function allwrap()
    # Load Data
    function load_data()
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS3-gev/nlsw88w.csv"
        df = CSV.read(HTTP.get(url).body, DataFrame)
        X = [df.age df.white df.collgrad]
        Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4,
                 df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
        y = df.occupation
        return X, Z, y
    end

    # Question 1: Multinomial Logit
    function multinomial_logit_probs(β, X, Z)
        J = size(Z, 2)
        N = size(X, 1)
        β_X = β[1:end-1]
        γ = β[end]
        V = X * reshape(β_X, :, J-1) .+ γ * (Z[:, 1:J-1] .- Z[:, J])
        P = exp.(V) ./ (1 .+ sum(exp.(V), dims=2))
        P = hcat(P, 1 .- sum(P, dims=2))
        return P
    end

    function multinomial_logit_likelihood(β, X, Z, y)
        P = multinomial_logit_probs(β, X, Z)
        ll = sum(log.(P[i, y[i]] + 1e-10) for i in 1:size(X, 1))
        return -ll
    end

    function estimate_multinomial_logit(X, Z, y)
        println("Estimating Multinomial Logit Model...")
        J = size(Z, 2)
        K = size(X, 2)
        β_init = vcat(zeros(K * (J-1)), 0.1)
        result = optimize(β -> multinomial_logit_likelihood(β, X, Z, y), β_init, BFGS())
        β_hat = Optim.minimizer(result)
        return β_hat
    end

    # Question 2: Interpret Gamma Coefficient
    function interpret_gamma(γ, Z)
        avg_wage = mean(exp.(Z))
        pct_change = (exp(0.01 * γ) - 1) * 100
        println("\nInterpretation of γ coefficient:")
        println("γ = ", round(γ, digits=4))
        println("Average wage across all occupations: \$$(round(avg_wage, digits=2))")
        println("A 1% increase in the wage of an occupation relative to the base category")
        println("is associated with a $(round(pct_change, digits=2))% increase in the odds")
        println("of choosing that occupation over the base category.")
    end

    # Question 3: Nested Logit
    function nested_logit_probs(β, X, Z)
        β_WC, β_BC, λ_WC, λ_BC, γ = β[1:3], β[4:6], β[7], β[8], β[9]
        WC, BC, Other = [1, 2, 3], [4, 5, 6, 7], [8]
        J, N = size(Z, 2), size(X, 1)
        P = zeros(N, J)
        for i in 1:N
            V_WC = exp.(X[i,:]' * β_WC ./ λ_WC) .* exp.(γ * (Z[i, WC] .- Z[i, Other[1]]) ./ λ_WC)
            V_BC = exp.(X[i,:]' * β_BC ./ λ_BC) .* exp.(γ * (Z[i, BC] .- Z[i, Other[1]]) ./ λ_BC)
            IV_WC, IV_BC = sum(V_WC)^λ_WC, sum(V_BC)^λ_BC
            denom = 1 + IV_WC + IV_BC
            P[i, WC] = (IV_WC / denom) .* (V_WC ./ sum(V_WC))
            P[i, BC] = (IV_BC / denom) .* (V_BC ./ sum(V_BC))
            P[i, Other] .= 1 / denom
        end
        return P
    end

    function nested_logit_likelihood(β, X, Z, y)
        P = nested_logit_probs(β, X, Z)
        ll = sum(log.(P[i, y[i]] + 1e-10) for i in 1:size(X, 1))
        return -ll
    end

    function estimate_nested_logit(X, Z, y)
        println("\nEstimating Nested Logit Model...")
        K = size(X, 2)
        β_init = [zeros(2*K); ones(2); 0.1]
        result = optimize(β -> nested_logit_likelihood(β, X, Z, y), β_init, BFGS())
        β_hat = Optim.minimizer(result)
        return β_hat
    end

    function numerical_hessian(f, x, ε=1e-5)
        n = length(x)
        H = zeros(n, n)
        for i in 1:n
            for j in 1:n
                x_pp, x_pm, x_mp, x_mm = copy(x), copy(x), copy(x), copy(x)
                x_pp[i] += ε; x_pp[j] += ε
                x_pm[i] += ε; x_pm[j] -= ε
                x_mp[i] -= ε; x_mp[j] += ε
                x_mm[i] -= ε; x_mm[j] -= ε
                H[i,j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4ε^2)
            end
        end
        return H
    end

    function compute_standard_errors(f, β_hat)
        H = numerical_hessian(f, β_hat)
        return sqrt.(diag(inv(H)))
    end

    function interpret_nested_logit_results(β_hat, se)
        println("\nNested Logit Estimation Results:")
        println("================================")
        println("Estimated coefficients:")
        println("β_WC: ", β_hat[1:3])
        println("β_BC: ", β_hat[4:6])
        println("λ_WC: ", β_hat[7])
        println("λ_BC: ", β_hat[8])
        println("γ: ", β_hat[9])
        println("\nStandard errors:")
        println("SE(β_WC): ", se[1:3])
        println("SE(β_BC): ", se[4:6])
        println("SE(λ_WC): ", se[7])
        println("SE(λ_BC): ", se[8])
        println("SE(γ): ", se[9])
        
        println("\nInterpretation of results:")
        println("1. Nest coefficients (β_WC and β_BC):")
        println("   These represent the effect of individual characteristics on the likelihood")
        println("   of choosing occupations in the White Collar and Blue Collar nests, respectively.")
        println("   Positive coefficients indicate increased likelihood, negative decreased likelihood.")
        
        println("\n2. Inclusive Value Parameters (λ_WC and λ_BC):")
        println("   λ_WC = ", round(β_hat[7], digits=4), " (SE: ", round(se[7], digits=4), ")")
        println("   λ_BC = ", round(β_hat[8], digits=4), " (SE: ", round(se[8], digits=4), ")")
        println("   These measure the degree of independence within each nest.")
        println("   Values closer to 1 indicate more independence, closer to 0 more correlation.")
        
        println("\n3. Wage coefficient (γ):")
        println("   γ = ", round(β_hat[9], digits=4), " (SE: ", round(se[9], digits=4), ")")
        println("   This represents the effect of wage differences on occupational choice.")
        println("   A positive coefficient indicates that higher wages increase the likelihood")
        println("   of choosing an occupation, all else being equal.")
    end

    # Main execution
    X, Z, y = load_data()
    
    # Question 1: Multinomial Logit
    β_multinomial = estimate_multinomial_logit(X, Z, y)
    println("Multinomial Logit Coefficients:")
    println(β_multinomial)
    
    # Question 2: Interpret Gamma
    γ_multinomial = β_multinomial[end]
    interpret_gamma(γ_multinomial, Z)
    
    # Question 3: Nested Logit
    β_nested = estimate_nested_logit(X, Z, y)
    se_nested = compute_standard_errors(β -> nested_logit_likelihood(β, X, Z, y), β_nested)
    interpret_nested_logit_results(β_nested, se_nested)
end

# Call the allwrap function
allwrap()

#question 05
using Test               
function run_tests()
    test_sets = [
        ("Data Properties", test_data_properties),
        ("Multinomial Logit Properties", test_multinomial_logit_properties),
        ("Nested Logit Properties", test_nested_logit_properties),
        ("Likelihood Properties", test_likelihood_properties),
        ("Coefficient Interpretation", test_coefficient_interpretation),
        ("Numerical Properties", test_numerical_properties),
        ("Gradient Calculation", test_gradient_calculation)
    ]

    for (name, test_function) in test_sets
        try
            @testset "$name" begin
                test_function()
            end
        catch e
            println("Error in test set '$name': ", e)
            println("Stacktrace:")
            for (exception, backtrace) in Base.catch_stack()
                showerror(stdout, exception, backtrace)
                println()
            end
        end
    end
end

function test_data_properties()
    X, Z, y = load_data()
    @test size(X, 2) == 3
    @test size(Z, 2) == 8
    @test length(unique(y)) == 8
    @test all(0 .<= X[:, 2] .<= 1)
    @test all(0 .<= X[:, 3] .<= 1)
    @test all(Z .> 0)
    @test eltype(X) <: Float64
    @test eltype(Z) <: Float64
    @test eltype(y) <: Int
end

function test_multinomial_logit_properties()
    X, Z, y = load_data()
    β_test = rand(size(X, 2) * (size(Z, 2) - 1) + 1)
    probs = multinomial_logit_probs(β_test, X, Z)
    
    ratio_1_2 = probs[:, 1] ./ probs[:, 2]
    ratio_1_2_reduced = probs[:, 1] ./ (probs[:, 1] .+ probs[:, 2])
    @test all(isapprox.(ratio_1_2, ratio_1_2_reduced, atol=1e-6))
    
    β_test_changed = copy(β_test)
    β_test_changed[1] += 0.5
    probs_changed = multinomial_logit_probs(β_test_changed, X, Z)
    @test mean(probs_changed[:, 1]) > mean(probs[:, 1])
    
    @test all(isapprox.(sum(probs, dims=2), 1, atol=1e-6))
    @test all(0 .<= probs .<= 1)
    
    β_extreme = fill(1e6, length(β_test))
    probs_extreme = multinomial_logit_probs(β_extreme, X, Z)
    @test !any(isnan.(probs_extreme))
    @test all(isapprox.(sum(probs_extreme, dims=2), 1, atol=1e-6))
    
    β_scaled = 2 * β_test
    probs_scaled = multinomial_logit_probs(β_scaled, X, Z)
    @test all(isapprox.(probs, probs_scaled, atol=1e-6))
end

function test_nested_logit_properties()
    X, Z, y = load_data()
    β_test = rand(2 * size(X, 2) + 3)
    probs = nested_logit_probs(β_test, X, Z)
    
    ratio_1_2 = probs[:, 1] ./ probs[:, 2]
    ratio_1_2_reduced = probs[:, 1] ./ (probs[:, 1] .+ probs[:, 2])
    @test all(isapprox.(ratio_1_2, ratio_1_2_reduced, atol=1e-6))
    
    ratio_1_4 = probs[:, 1] ./ probs[:, 4]
    ratio_1_4_reduced = probs[:, 1] ./ (probs[:, 1] .+ probs[:, 4])
    @test !all(isapprox.(ratio_1_4, ratio_1_4_reduced, atol=1e-6))
    
    @test all(isapprox.(sum(probs, dims=2), 1, atol=1e-6))
    @test all(0 .<= probs .<= 1)
    
    λ_WC, λ_BC = β_test[end-2:end-1]
    @test 0 < λ_WC <= 1
    @test 0 < λ_BC <= 1
end

function test_likelihood_properties()
    X, Z, y = load_data()
    β_multinomial = rand(size(X, 2) * (size(Z, 2) - 1) + 1)
    β_nested = rand(2 * size(X, 2) + 3)
    
    ll_multinomial = multinomial_logit_likelihood(β_multinomial, X, Z, y)
    ll_nested = nested_logit_likelihood(β_nested, X, Z, y)
    
    @test ll_multinomial < 0
    @test ll_nested < 0
    
    probs_multinomial = multinomial_logit_probs(β_multinomial, X, Z)
    y_perfect = [findmax(probs_multinomial[i, :])[2] for i in 1:size(X, 1)]
    ll_perfect = multinomial_logit_likelihood(β_multinomial, X, Z, y_perfect)
    @test ll_perfect > ll_multinomial
end

function test_coefficient_interpretation()
    X, Z, y = load_data()
    β_test = rand(2 * size(X, 2) + 3)
    probs_base = nested_logit_probs(β_test, X, Z)
    
    β_edu = copy(β_test)
    β_edu[3] += 0.1
    probs_edu = nested_logit_probs(β_edu, X, Z)
    @test mean(probs_edu[:, 1:3]) > mean(probs_base[:, 1:3])
    
    β_wage = copy(β_test)
    β_wage[end] += 0.1
    probs_wage = nested_logit_probs(β_wage, X, Z)
    wage_effect = [cor(Z[:, j], probs_wage[:, j] - probs_base[:, j]) for j in 1:8]
    @test all(wage_effect .> 0)
    
    @test β_test[end] > 0
end

function test_numerical_properties()
    f(x) = sum(x.^2)
    x_test = [1.0, 2.0]
    
    H = numerical_hessian(f, x_test)
    @test issymmetric(H)
    @test isposdef(H)
    
    se = compute_standard_errors(f, x_test)
    @test all(se .> 0)
    @test se[2] < se[1]
end

function test_gradient_calculation()
    if @isdefined(gradient_multinomial_logit)
        X, Z, y = load_data()
        β_test = rand(size(X, 2) * (size(Z, 2) - 1) + 1)
        grad = gradient_multinomial_logit(β_test, X, Z, y)
        @test length(grad) == length(β_test)
        @test all(isfinite.(grad))
    end
end

# Run all tests
run_tests()

println("All comprehensive tests completed. Check the output for any error messages.")

