"""
    forwardsub(L,b)

Solve the lower-triangular linear system with matrix `L` and
right-hand side vector `b`.
"""
function forwardsub(L,b)
    n = size(L,1)
    x = zeros(n)
    x[1] = b[1]/L[1,1]
    for i in 2:n
        s = sum( L[i,j]*x[j] for j in 1:i-1 )
        x[i] = ( b[i] - s ) / L[i,i]
    end
    return x
end

"""
    backsub(U,b)

Solve the upper-triangular linear system with matrix `U` and
right-hand side vector `b`.
"""
function backsub(U,b)
    n = size(U,1)
    x = zeros(n)
    x[n] = b[n]/U[n,n]
    for i in n-1:-1:1
        s = sum( U[i,j]*x[j] for j in i+1:n )
        x[i] = ( b[i] - s ) / U[i,i]
    end
    return x
end

"""
    lufact(A)

Compute the LU factorization of square matrix `A`, returning the
factors.
"""
function lufact(A)
    n = size(A,1)        # detect the dimensions from the input
    L = diagm(ones(n))   # ones on main diagonal, zeros elsewhere
    U = zeros(n,n)
    Aₖ = float(copy(A))  # make a working copy 

    # Reduction by outer products
    for k in 1:n-1
        U[k,:] = Aₖ[k,:]
        L[:,k] = Aₖ[:,k]/U[k,k]
        Aₖ -= L[:,k]*U[k,:]'
    end
    U[n,n] = Aₖ[n,n]
    return LowerTriangular(L),UpperTriangular(U)
end

"""
    plufact(A)

Compute the PLU factorization of square matrix `A`, returning the
triangular factors and a row permutation vector.
"""
function plufact(A)
    n = size(A,1)
    L = zeros(n,n)
    U = zeros(n,n)
    p = fill(0,n)
    Aₖ = float(copy(A))

    # Reduction by outer products
    for k in 1:n-1
        p[k] = argmax(abs.(Aₖ[:,k]))
        U[k,:] = Aₖ[p[k],:]
        L[:,k] = Aₖ[:,k]/U[k,k]
        Aₖ -= L[:,k]*U[k,:]'
    end
    p[n] = argmax(abs.(Aₖ[:,n]))
    U[n,n] = Aₖ[p[n],n]
    L[:,n] = Aₖ[:,n]/U[n,n]
    return LowerTriangular(L[p,:]),U,p
end

#######################################  Start of Widad's Practice  #######################################

# 2.1 Polynomial Interpolation (p.29)

# Demo 2.1.3 on page 30 ( we have 4 data point and we seek and approximation with a polynomial of degree 3)

year = [1982, 2000, 2010, 2015]
pop = [1008.18, 1262.64, 1337.82, 1374.62]

t = year .- 1980
y = pop

# construct the Vandermonde matrix
V = [t[i]^j for i=1:4, j=0:3]

#solve the linear system ( using backslash operation)
c = V\y

# compute the residual as a check
r = y - V*c # a relative difference comparable to the ϵmach = 2^-52 ( machine epsilon / machine precision) is expected 

# By our definition the c are coefficients of the polynomial p(t) = c0 + c1*t + c2*t^2 + c3*t^3 (ascending degree order)
# for the interpolation polynomial. We can use the polynomial to estimate the population in 2005:

p = Polynomial(c) # Construct a polynomial from its coefficients

# evaluate the polynomial at 2005 ( (❗) by our definition t = year - 1980 : 2015 - 1980 = 35)

p(2005-1980) # 1328.5# in millions of people 

# visualize the data and the polynomial
using Plots
scatter(t,y,label="actual", legend=:topleft, xlabel = " years since 1980", ylabel = "population in millions", title = "population of China")

# choose 500 times in the interval [0,35] (❗) by our definition t = year - 1980 
tt = range(0,35,length=500)
yy = p.(tt) # evaluate the polynomial at each time
plot!(tt,yy,label="interpolant") # add the interpolant to the plot

#2.4 LU Factorization (p. 50)
#Demo 2.4.3 on page 51

L = tril(rand(1:9,3,3))
U = triu(rand(1:9,3,3))

multip1 = L[:,1]*U[1,:]' + L[:,2]*U[2,:]' + L[:,3]*U[3,:]' # matrix multiplication
@show L 
@show L[:,1]
@show L[:,2]
@show L[:,3]
@show U
@show U[1,:]
@show U[2,:]
@show U[3,:]

multip2 = L*U # matrix multiplication
@assert multip1 == multip2
#Demo 2.4.4 on page 52
A₁ = [
    2 0 4 3
    -4 5 -7 -10
    1 15 2 -4.5 
    -2 0 2 -13
]
L = diagm(ones(4)) # ones on main diagonal, zeros elsewhere
U = zeros(4,4)
U[1,:] = A₁[1,:]
@show U
# fist column of L ( (❗) definition considered : A = LU)
L[:,1] = A₁[:,1]/U[1,1]
@show L
# we have obtained the first term in the sum for LU and we subtract it from A₁ to get A₂
A₂ = A₁ - L[:,1]*U[1,:]'
U[2,:] = A₂[2,:]
L[:,2] = A₂[:,2]/U[2,2]
@show L
# we have obtained the second term in the sum for LU and we subtract it from A₂ to get A₃
A₃ = A₂ - L[:,2]*U[2,:]'
U[3,:] = A₃[3,:]
L[:,3] = A₃[:,3]/U[3,3]
A₄ = A₃ - L[:,3]*U[3,:]'
# finally we pick up the last unknown in the factors 
U[4,4] = A₄[4,4]
# now we have all L and U factors
@show L
@show U
# check the result
@assert A₁ == L*U

# Flop (floating-point operation) counting
# Demo 2.5.6 page 61

randn(5,5) *randn(5) #throwaway to force compilation

n = 400:200:6000
 t =[]
 for n in n 
     A = randn(n,n)
     x = randn(n)
     time = @elapsed for j in 1:50 ; A*x ; end
     push!(t,time)
 end

 # Plotting the time as a function of n on a log-log scale
@show n
@show t
# Normal scale plot
scatter(n,t,label = "data", legend = false,
        xaxis=("n"), yaxis=("time (s)"),
        title = "Timing of matrix-vector multiplications")
# Log-log scale plot
scatter(n,t,label = "data", legend = false,
        xaxis=(:log10,L"n"), yaxis=(:log10,"time (s)"),
        title = "Timing of matrix-vector multiplications")
# plot of a line that represents O(n^2) growth (❗ all such lines have the slope of 2)
plot!(n,t[end]*(n/n[end]).^2,label = L"O(n^2)", legend = :topleft, l=:dash)

# Row pivoting 
# A = LU factorization is not always stable for every nonsingular matrix A
# Sometimes the factorization odes not even exist
# Demo 2.6.1 page 66

A = [2 0 4 3; -4 5 -7 -10; 1 15 2 -4.5; -2 0 2 -13]
L,U = FNC.lufact(A)
@show L
@show U

A[[2,4],:]  = A[[4,2],:] 
@show A
L, U = FNC.lufact(A)
@show L
@show U

# Note (❗) : The presence if NaN in the result indicates that some impossible operation 
# was required. Let's find the source of the problem 

U[1,:] = A[1,:]
L[:,1] = A[:,1]/U[1,1]
A-= L[:,1]*U[1,:]'
U[2,:] = A[2,:]
@show U[2,2]
@show A[:,2]
L[:,2] = A[:,2]/U[2,2] # division by zero (❗)

# Note LU factorization is equivalent to Gaussian elimination with no row swaps 
# Row swaps are necessary to avoid division by zero 
