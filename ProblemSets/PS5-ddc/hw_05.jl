#question 01
using DataFrames, CSV, HTTP, Optim, LinearAlgebra, Statistics, ForwardDiff, Distributions, GLM, Random, FreqTables, Test, DataFramesMeta

url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdataBeta0.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

# Creating bus id variable
df = @transform(df, :bus_id = 1:nrow(df))

# Reshaping
dfy = select(df, :bus_id, r"^Y", :RouteUsage, :Branded)
dfy_long = stack(dfy, r"^Y", variable_name=:time, value_name=:Y)
dfy_long.time = parse.(Int, replace.(string.(dfy_long.time), "Y"=>""))

# Reshaping the odometer variable
dfx = select(df, :bus_id, r"^Odo")
dfx_long = stack(dfx, r"^Odo", variable_name=:time, value_name=:Odometer)
dfx_long.time = parse.(Int, replace.(string.(dfx_long.time), "Odo"=>""))

# reshaped dataframes
df_long = leftjoin(dfy_long, dfx_long, on = [:bus_id, :time])
sort!(df_long, [:bus_id, :time])

println(first(df_long, 5))

#question 02
# data for logistic regression
df_long = @transform(df_long, :mileage = :Odometer / 10000)  

# Estimating the static logit model
logit_model = glm(@formula(Y ~ mileage + Branded), df_long, Binomial(), LogitLink())

# Printing the summary of the model
println("Summary of the static logit model:")
println(logit_model)

# the estimated coefficients
θ_static = coef(logit_model)
println("\nEstimated coefficients (θ parameters):")
println("θ₀ (Intercept): ", θ_static[1])
println("θ₁ (Mileage): ", θ_static[2])
println("θ₂ (Branded): ", θ_static[3])

# printing the log-likelihood
println("\nLog-likelihood of the model:")
println(loglikelihood(logit_model))

# interpretation
println("\nInterpretation:")
println("- θ₀ represents the base utility of running a bus (not replacing the engine).")
println("- θ₁ represents the change in utility for each 10,000 miles on the odometer.")
println("- θ₂ represents the change in utility if the bus is from a high-end manufacturer (branded)")

# julia has been evaluating the third equation for ages now, so I am submitting the code for it. I can show you in class tomorrow
# I am sorry I could get till 147 iterations till now, monday night (I have been trying since last thursday really)
# question 03
include("create_grids.jl")
# 3(a): Read in the data
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdata.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

Y = Matrix(df[:, [Symbol("Y$t") for t in 1:20]])
Odo = Matrix(df[:, [Symbol("Odo$t") for t in 1:20]])
Xst = Matrix(df[:, [Symbol("Xst$t") for t in 1:20]])

# 3(b): state transition matrices
function create_grids()
    zval = range(0.25, 1.25, step=0.01)
    zbin = length(zval)
    xval = range(0, 45, step=0.125)
    xbin = length(xval)
    
    xtran = zeros(zbin * xbin, xbin)
    for z in 1:zbin, x in 1:xbin
        for x_next in x:xbin
            if x_next == x
                xtran[(z-1)*xbin + x, x_next] = 1 - exp(-zval[z] * 0.125)
            else
                xtran[(z-1)*xbin + x, x_next] = exp(-zval[z] * (xval[x_next] - xval[x])) - 
                                                exp(-zval[z] * (xval[x_next] + 0.125 - xval[x]))
            end
        end
        xtran[(z-1)*xbin + x, :] ./= sum(xtran[(z-1)*xbin + x, :])
    end
    
    return zval, zbin, xval, xbin, xtran
end

zval, zbin, xval, xbin, xtran = create_grids()

# indices for faster access
row_indices = [1 + (df.Zst[i] - 1) * xbin for i in 1:1000]
branded_indices = [df.Branded[i] + 1 for i in 1:1000]

# 3(c-e): constructing log likelihood
@views @inbounds function ddc_objective(θ)
    β = 0.9
    T, N = 20, 1000
    
    # future value array
    FV = zeros(zbin * xbin, 2, T + 1)
    
    # Backward recursion
    for t in T:-1:1, b in 0:1, z in 1:zbin
        for x in 1:xbin
            row = x + (z-1)*xbin
            v1 = θ[1] + θ[2]*xval[x] + θ[3]*b + 
                 β * dot(xtran[row,:], FV[(z-1)*xbin+1:z*xbin, b+1, t+1])
            v0 = β * dot(xtran[1+(z-1)*xbin,:], FV[(z-1)*xbin+1:z*xbin, b+1, t+1])
            FV[row, b+1, t] = β * log(exp(v0) + exp(v1))
        end
    end
    
    # Computing log-likelihood
    loglik = 0.0
    for i in 1:N
        row0 = row_indices[i]
        b_index = branded_indices[i]
        for t in 1:T
            row1 = Xst[i,t] + (df.Zst[i] - 1) * xbin
            v_diff = θ[1] + θ[2] * Odo[i,t] + θ[3] * (b_index-1) +
                     β * dot(xtran[row1,:] .- xtran[row0,:], 
                             FV[row0:row0+xbin-1, b_index, t+1])
            p1 = 1 / (1 + exp(-v_diff))
            loglik += Y[i,t] * log(p1) + (1 - Y[i,t]) * log(1 - p1)
        end
    end
    
    return -loglik  # Return negative log-likelihood for min
end

# 3(f-h): Estimating the model
function estimate_ddc_model()
    initial_θ = [0.0, 0.0, 0.0]  
    result = optimize(ddc_objective, initial_θ, BFGS(), Optim.Options(show_trace = true))
    return Optim.minimizer(result)
end

# estimation
estimated_θ = estimate_ddc_model()
println("Estimated parameters: ", estimated_θ)

# question 04
@testset "Dynamic Discrete Choice Model Tests" begin
    @testset "Data Loading and Preprocessing" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdata.csv"
        df = CSV.read(HTTP.get(url).body, DataFrame)
        
        @test size(df, 1) == 1000
        @test size(df, 2) > 60  
        @test :Y1 in names(df)
        @test :Odo1 in names(df)
        @test :Xst1 in names(df)
        @test :Zst in names(df)
        
        # converting to arrays
        Y = Matrix(df[:, r"^Y"])
        Odo = Matrix(df[:, r"^Odo"])
        Xst = Matrix(df[:, r"^Xst"])
        
        @test size(Y) == (1000, 20)
        @test size(Odo) == (1000, 20)
        @test size(Xst) == (1000, 20)
    end

    @testset "Grid Creation" begin
        zval, zbin, xval, xbin, xtran = create_grids()
        
        @test length(zval) > 0
        @test zbin > 0
        @test length(xval) > 0
        @test xbin > 0
        @test size(xtran, 1) == zbin * xbin
        @test size(xtran, 2) == xbin
        @test all(sum(xtran, dims=2) .≈ 1)  
    end

    @testset "Future Value Computation" begin
    
        T = 20
        zbin, xbin = 5, 10
        β = 0.9
        θ = [1.0, -0.1, 0.5]  
        
        FV = zeros(zbin * xbin, 2, T + 1)
        
        # Test function to compute future values
        function compute_future_values!(FV, xtran, zval, xval, θ, β)
            for t in T:-1:1, b in 0:1, z in 1:zbin, x in 1:xbin
                row = x + (z-1)*xbin
                v1 = θ[1] + θ[2]*xval[x] + θ[3]*b + 
                     β * xtran[row,:]' * FV[(z-1)*xbin+1:z*xbin, b+1, t+1]
                v0 = β * xtran[1+(z-1)*xbin,:]' * FV[(z-1)*xbin+1:z*xbin, b+1, t+1]
                FV[row, b+1, t] = β * log(exp(v0) + exp(v1))
            end
        end
        
        compute_future_values!(FV, xtran, zval, xval, θ, β)
        
        @test !any(isnan, FV)
        @test all(FV .>= 0)
        @test size(FV) == (zbin * xbin, 2, T + 1)
    end

    @testset "Log Likelihood Computation" begin
        
        N, T = 100, 20
        Y = rand(0:1, N, T)
        Odo = cumsum(rand(1:5, N, T), dims=2)
        Xst = ceil.(Int, Odo ./ 1.25)
        Zst = rand(1:5, N)
        B = rand(0:1, N)
        
        
        θ = [1.0, -0.1, 0.5]
        β = 0.9
        FV = rand(50, 2, T + 1)  
        xtran = rand(50, 10)
        xtran ./= sum(xtran, dims=2) 
        
        function compute_log_likelihood(Y, Odo, Xst, Zst, B, θ, β, FV, xtran)
            loglik = 0.0
            for i in 1:N, t in 1:T
                row0 = 1 + (Zst[i] - 1) * 10
                row1 = Xst[i,t] + (Zst[i] - 1) * 10
                v_diff = θ[1] + θ[2] * Odo[i,t] + θ[3] * B[i] +
                         β * (xtran[row1,:] .- xtran[row0,:])' * FV[row0:row0+9, B[i]+1, t+1]
                p1 = 1 / (1 + exp(-v_diff))
                loglik += Y[i,t] * log(p1) + (1 - Y[i,t]) * log(1 - p1)
            end
            return loglik
        end
        
        loglik = compute_log_likelihood(Y, Odo, Xst, Zst, B, θ, β, FV, xtran)
        
        @test !isnan(loglik)
        @test loglik < 0  # Log-likelihood should be negative
    end
end

