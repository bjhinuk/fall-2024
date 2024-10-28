#question 01
using Pkg
Pkg.add("MultivariateStats")
using DataFrames, CSV, GLM, Statistics, LinearAlgebra, MultivariateStats, Optim, Distributions, ForwardDiff, Test
function analyze_wage_data()
    try
        data = CSV.read("E:/OKLAHOMA/Fall 24/metrics III/fall-2024/ProblemSets/PS8-factor/nlsy.csv", DataFrame)
        
        # Print column names to verify
        println("\nAvailable columns in dataset:")
        println(names(data))
        
        # linear regression model
        println("\nEstimating regression model")
        model = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr), data)
        
        # model statistics
        r_squared = r²(model)
        adj_r_squared = adjr²(model)
        num_obs = nobs(model)
        
        # results
        println("\nRegression Results for Log Wage Model:")
        println(model)
        
        println("\nModel Statistics:")
        println("Number of observations: ", num_obs)
        println("R²: ", round(r_squared, digits=4))
        println("Adjusted R²: ", round(adj_r_squared, digits=4))
        
        return model, data
        
    catch e
        println("Error in analysis: ", e)
        # Print the full error stack trace for debugging
        println("Stack trace:")
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt)
            println()
        end
    end
end

# analysis
println("Starting analysis")
result = analyze_wage_data()

#question 02
# computing ASVAB correlations
function compute_asvab_correlations()
    try
        println("Reading data...")
        data = CSV.read("E:/OKLAHOMA/Fall 24/metrics III/fall-2024/ProblemSets/PS8-factor/nlsy.csv", DataFrame)
        
        # Extracting ASVAB variables into a matrix
        asvabMat = Matrix(select(data, r"asvab"))  # Select all columns with "asvab" in name
        
        # correlation matrix
        correlation_matrix = cor(asvabMat)
        
        # results
        println("\nASVAB Variables Correlation Matrix:")
        
        # ASVAB variable names
        asvab_vars = names(select(data, r"asvab"))
        
        # Print header row
        print("\t")
        for var in asvab_vars
            print(var, "\t")
        end
        println()
        
        # correlation matrix with row labels
        for i in 1:length(asvab_vars)
            print(asvab_vars[i], "\t")
            for j in 1:length(asvab_vars)
                print(round(correlation_matrix[i,j], digits=3), "\t")
            end
            println()
        end
        
        return correlation_matrix, asvab_vars
    catch e
        println("Error in analysis: ", e)
        println("Stack trace:")
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt)
            println()
        end
    end
end

# analysis
println("Computing ASVAB correlations...")
corr_matrix, var_names = compute_asvab_correlations()

#question 03
function analyze_wage_with_asvab()
    data = CSV.read("E:/OKLAHOMA/Fall 24/metrics III/fall-2024/ProblemSets/PS8-factor/nlsy.csv", DataFrame)
    
    # Original model (without ASVAB variables)
    println("\nOriginal Model (without ASVAB):")
    println("==============================")
    model_original = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr), data)
    println(model_original)
    println("\nOriginal Model R²: ", round(r²(model_original), digits=4))
    println("Original Model Adj. R²: ", round(adjr²(model_original), digits=4))
    
    # Extended model (with ASVAB variables)
    println("\nExtended Model (with ASVAB):")
    println("===========================")
    model_extended = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + 
                               grad4yr + asvabAR + asvabCS + asvabMK + asvabNO + asvabPC + asvabWK), data)
    println(model_extended)
    println("\nExtended Model R²: ", round(r²(model_extended), digits=4))
    println("Extended Model Adj. R²: ", round(adjr²(model_extended), digits=4))
    
    # note about multicollinearity
    println("\nNote about potential multicollinearity:")
    println("Including all ASVAB variables simultaneously might lead to")
    println("multicollinearity issues due to their potential high correlations.")
    
    return model_original, model_extended, data
end

# analysis
println("Starting wage analysis...")
model_orig, model_ext, data = analyze_wage_with_asvab()

#question 04
function wage_regression_with_pca()
    data = CSV.read("E:/OKLAHOMA/Fall 24/metrics III/fall-2024/ProblemSets/PS8-factor/nlsy.csv", DataFrame)
    
    # Extract ASVAB variables into a matrix
    asvabMat = Matrix(select(data, [:asvabAR, :asvabCS, :asvabMK, :asvabNO, :asvabPC, :asvabWK]))
    
    # Transpose matrix for PCA (need N x J matrix)
    asvabMat = asvabMat'
    
    println("\nComputing Principal Component")
    # Fit PCA model with one component
    M = fit(PCA, asvabMat; maxoutdim=1)
    
    # Transform data to get first principal component
    asvabPCA = MultivariateStats.transform(M, asvabMat)
    
    # Convert PCA result to vector and add to dataframe
    data.asvab_pc1 = vec(asvabPCA')
    
    # Regression with first principal component
    println("\nEstimating regression with first principal component...")
    model_pca = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + 
                           grad4yr + asvab_pc1), data)
    
    # results
    println("\nRegression Results (with first principal component):")
    println(model_pca)
    
    # model statistics
    r2 = r²(model_pca)
    adj_r2 = adjr²(model_pca)
    n_obs = nobs(model_pca)
    
    println("\nModel Statistics:")
    println("Number of observations: ", n_obs)
    println("R²: ", round(r2, digits=4))
    println("Adjusted R²: ", round(adj_r2, digits=4))
    
    # proportion of variance explained by first PC
    println("\nPCA Statistics:")
    println("Proportion of variance explained by first PC: ", 
            round(principalratio(M), digits=4))
    
    return model_pca, M, data
end

# analysis
println("Starting PCA regression analysis")
model_pca, pca_fit, data = wage_regression_with_pca()

#question 05
function wage_regression_with_fa()
    data = CSV.read("E:/OKLAHOMA/Fall 24/metrics III/fall-2024/ProblemSets/PS8-factor/nlsy.csv", DataFrame)
    
    # Extract ASVAB variables into a matrix
    asvabMat = Matrix(select(data, [:asvabAR, :asvabCS, :asvabMK, :asvabNO, :asvabPC, :asvabWK]))
    
    # Transpose matrix for Factor Analysis (need N x J matrix)
    asvabMat = asvabMat'
    
    println("\nComputing Factor Analysis")
    # Fit Factor Analysis model with one factor
    M = fit(FactorAnalysis, asvabMat; maxoutdim=1)
    
    # Transform data to get factor scores
    asvabFA = MultivariateStats.transform(M, asvabMat)
    
    # Convert FA result to vector and add to dataframe
    data.asvab_factor1 = vec(asvabFA')
    
    # Regression with factor score
    println("\nEstimating regression with factor scores")
    model_fa = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + 
                          grad4yr + asvab_factor1), data)
    
    # results
    println("\nRegression Results (with factor score):")
    println(model_fa)
    
    # model statistics
    r2 = r²(model_fa)
    adj_r2 = adjr²(model_fa)
    n_obs = nobs(model_fa)
    
    println("\nModel Statistics:")
    println("Number of observations: ", n_obs)
    println("R²: ", round(r2, digits=4))
    println("Adjusted R²: ", round(adj_r2, digits=4))
    
    # factor analysis information
    println("\nFactor Analysis Statistics:")
    println("Projection Matrix:")
    display(projection(M))
    
    return model_fa, M, data
end

# analysis
println("Factor Analysis regression")
model_fa, fa_fit, data = wage_regression_with_fa()

#question 06
function estimate_measurement_system()
    data = CSV.read("E:/OKLAHOMA/Fall 24/metrics III/fall-2024/ProblemSets/PS8-factor/nlsy.csv", DataFrame)
    
    # Include lgwt function for Gauss-Legendre quadrature
    include("lgwt.jl")
    
    # Standardize ASVAB scores
    asvab_cols = [:asvabAR, :asvabCS, :asvabMK, :asvabNO, :asvabPC, :asvabWK]
    for col in asvab_cols
        data[!, col] = (data[!, col] .- mean(data[!, col])) ./ std(data[!, col])
    end
    
    # Measurement equation covariates
    X = Matrix(select(data, [:black, :hispanic, :female]))
    X = hcat(ones(nrow(data)), X)
    
    # Wage equation covariates
    X_wage = Matrix(select(data, [:black, :hispanic, :female, :schoolt, :gradHS, :grad4yr]))
    X_wage = hcat(ones(nrow(data)), X_wage)
    
    # Standardized ASVAB scores
    asvab_vars = Matrix(select(data, asvab_cols))
    
    # Standardize log wage
    y = (data.logwage .- mean(data.logwage)) ./ std(data.logwage)
    
    function compute_likelihood(theta, x_m, x, M, y_i, nodes, weights)
        try
            # Parameter dimensions
            n_alpha = 4    
            n_asvab = 6    
            n_beta = 7     
            
            # Extract and transform parameters for numerical stability
            alpha = reshape(theta[1:(n_alpha*n_asvab)], n_alpha, n_asvab)
            gamma = theta[(n_alpha*n_asvab+1):(n_alpha*n_asvab+n_asvab)]
            beta = theta[(n_alpha*n_asvab+n_asvab+1):(n_alpha*n_asvab+n_asvab+n_beta)]
            delta = theta[end-1]
            sigma = exp(theta[end])
            
            like = 0.0
            
            for (q, w) in zip(nodes, weights)
                asvab_like = 1.0
                
                # ASVAB equations with numerical stability checks
                for j in 1:n_asvab
                    mu_j = clamp(dot(x_m, view(alpha, :, j)) + gamma[j] * q, -100, 100)
                    diff = (M[j] - mu_j) / sigma
                    if !isfinite(diff)
                        return 1e-100
                    end
                    asvab_like *= max(pdf(Normal(), diff), 1e-100)
                end
                
                # Wage equation with numerical stability checks
                wage_mu = clamp(dot(x, beta) + delta * q, -100, 100)
                wage_diff = (y_i - wage_mu) / sigma
                if !isfinite(wage_diff)
                    return 1e-100
                end
                wage_like = max(pdf(Normal(), wage_diff), 1e-100)
                
                # Combine likelihoods with numerical stability
                term = w * asvab_like * wage_like * max(pdf(Normal(), q), 1e-100)
                like += isfinite(term) ? term : 0.0
            end
            
            return max(like, 1e-100)
        catch e
            println("Error in likelihood computation: ", e)
            return 1e-100
        end
    end
    
    function neg_log_likelihood(theta)
        try
            nodes, weights = lgwt(7, -4, 4)
            
            loglike = 0.0
            N = size(X, 1)
            
            for i in 1:N
                x_m_i = view(X, i, :)
                x_i = view(X_wage, i, :)
                M_i = view(asvab_vars, i, :)
                y_i = y[i]
                
                like_i = compute_likelihood(theta, x_m_i, x_i, M_i, y_i, nodes, weights)
                loglike += log(max(like_i, 1e-100))
            end
            
            return isfinite(-loglike) ? -loglike : 1e100
        catch e
            println("Error in negative log likelihood computation: ", e)
            return 1e100
        end
    end
    
    # Initialize parameters with careful starting values
    n_params = 4*6 + 6 + 7 + 1 + 1
    initial_theta = zeros(n_params)
    
    # Set conservative starting values
    initial_theta[1:24] .= 0.01  # alpha parameters
    initial_theta[25:30] .= 0.01  # gamma parameters
    initial_theta[31:37] .= 0.01  # beta parameters
    initial_theta[38] = 0.01      # delta parameter
    initial_theta[39] = 0.0       # log sigma (exp(0) = 1)
    
    # Optimize with more conservative settings
    println("\nStarting optimization...")
    result = optimize(neg_log_likelihood, 
                     initial_theta,
                     LBFGS(),
                     Optim.Options(show_trace = true,
                                 iterations = 1000,
                                 g_tol = 1e-4,
                                 x_tol = 1e-4,
                                 f_tol = 1e-4))
    
    # Process results
    if Optim.converged(result)
        theta_hat = Optim.minimizer(result)
        println("\nConverged to solution.")
        
        # Print key results
        println("\nFinal negative log-likelihood: ", Optim.minimum(result))
        
        # Print parameter estimates in groups
        println("\nParameter Estimates:")
        println("===================")
        
        alpha_mat = reshape(theta_hat[1:24], 4, 6)
        println("\nAlpha parameters:")
        display(alpha_mat)
        
        println("\nGamma parameters:")
        println(theta_hat[25:30])
        
        println("\nBeta parameters:")
        println(theta_hat[31:37])
        
        println("\nDelta parameter:")
        println(theta_hat[38])
        
        println("\nSigma parameter:")
        println(exp(theta_hat[39]))
        
        return theta_hat, result
    else
        println("Optimization did not converge")
        return nothing, result
    end
end

# Run estimation with error handling
println("Starting estimation...")
try
    theta_hat, optimization_result = estimate_measurement_system()
catch e
    println("Error in estimation: ", e)
end

#question 07
using Test
using DataFrames
using CSV
using Statistics
using LinearAlgebra
using Distributions
using MultivariateStats
using GLM

@testset "PS8 Tests" begin
    # Test data loading and preparation
    @testset "Data Loading" begin
        # Test data file exists and can be read
        @test isfile("E:/OKLAHOMA/Fall 24/metrics III/fall-2024/ProblemSets/PS8-factor/nlsy.csv")
        data = CSV.read("E:/OKLAHOMA/Fall 24/metrics III/fall-2024/ProblemSets/PS8-factor/nlsy.csv", DataFrame)
        @test data isa DataFrame
        @test nrow(data) > 0
        @test :logwage in names(data)
        @test :asvab1 in names(data)
    end

    # Test Question 1: Basic Regression
    @testset "Basic Wage Regression" begin
        data = CSV.read("E:/OKLAHOMA/Fall 24/metrics III/fall-2024/ProblemSets/PS8-factor/nlsy.csv", DataFrame)
        
        # Test model fitting
        model = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr), data)
        @test model isa StatsModels.TableRegressionModel
        
        # Test R² is between 0 and 1
        @test 0 ≤ r²(model) ≤ 1
        
        # Test number of coefficients
        @test length(coef(model)) == 7  # intercept + 6 variables
    end

    # Test Question 2: ASVAB Correlations
    @testset "ASVAB Correlations" begin
        data = CSV.read("E:/OKLAHOMA/Fall 24/metrics III/fall-2024/ProblemSets/PS8-factor/nlsy.csv", DataFrame)
        asvab_mat = Matrix(select(data, [:asvabAR, :asvabCS, :asvabMK, :asvabNO, :asvabPC, :asvabWK]))
        corr_matrix = cor(asvab_mat)
        
        # Test correlation matrix properties
        @test size(corr_matrix) == (6, 6)
        @test isapprox(diag(corr_matrix), ones(6))
        @test issymmetric(corr_matrix)
        @test all(-1 .≤ corr_matrix .≤ 1)
    end

    # Test Question 3: Extended Regression
    @testset "Extended Regression with ASVAB" begin
        data = CSV.read("E:/OKLAHOMA/Fall 24/metrics III/fall-2024/ProblemSets/PS8-factor/nlsy.csv", DataFrame)
        
        model_ext = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + 
                               grad4yr + asvabAR + asvabCS + asvabMK + asvabNO + asvabPC + asvabWK), data)
        
        @test model_ext isa StatsModels.TableRegressionModel
        @test length(coef(model_ext)) == 13  # original 7 + 6 ASVAB variables
        @test 0 ≤ r²(model_ext) ≤ 1
    end

    # Test Question 4: PCA
    @testset "PCA Analysis" begin
        data = CSV.read("E:/OKLAHOMA/Fall 24/metrics III/fall-2024/ProblemSets/PS8-factor/nlsy.csv", DataFrame)
        
        # Prepare ASVAB matrix
        asvab_mat = Matrix(select(data, [:asvabAR, :asvabCS, :asvabMK, :asvabNO, :asvabPC, :asvabWK]))'
        
        # Test PCA
        M = fit(PCA, asvab_mat; maxoutdim=1)
        @test M isa PCA
        
        # Test transformed data
        transformed = MultivariateStats.transform(M, asvab_mat)
        @test size(transformed, 1) == 1  # one component
        @test size(transformed, 2) == size(asvab_mat, 2)  # same number of observations
    end

    # Test Question 5: Factor Analysis
    @testset "Factor Analysis" begin
        data = CSV.read("E:/OKLAHOMA/Fall 24/metrics III/fall-2024/ProblemSets/PS8-factor/nlsy.csv", DataFrame)
        
        # Prepare ASVAB matrix
        asvab_mat = Matrix(select(data, [:asvabAR, :asvabCS, :asvabMK, :asvabNO, :asvabPC, :asvabWK]))'
        
        # Test Factor Analysis
        M = fit(FactorAnalysis, asvab_mat; maxoutdim=1)
        @test M isa FactorAnalysis
        
        # Test transformed data
        transformed = MultivariateStats.transform(M, asvab_mat)
        @test size(transformed, 1) == 1
        @test size(transformed, 2) == size(asvab_mat, 2)
    end

    # Test Question 6: Maximum Likelihood Estimation
    @testset "MLE Implementation" begin
        data = CSV.read("E:/OKLAHOMA/Fall 24/metrics III/fall-2024/ProblemSets/PS8-factor/nlsy.csv", DataFrame)
        
        # Test Gauss-Legendre quadrature
        include("lgwt.jl")
        nodes, weights = lgwt(7, -4, 4)
        @test length(nodes) == 7
        @test length(weights) == 7
        @test sum(weights) ≈ 8.0  # total interval length
        
        # Test data preparation
        X = Matrix(select(data, [:black, :hispanic, :female]))
        X = hcat(ones(nrow(data)), X)
        @test size(X, 2) == 4  # intercept + 3 demographics
        
        # Test parameter dimensions
        n_params = 4*6 + 6 + 7 + 1 + 1  # alpha + gamma + beta + delta + sigma
        @test n_params == 39  # verify total number of parameters
    end
end