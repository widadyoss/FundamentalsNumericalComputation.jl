using FundamentalsNumericalComputation
"""
    horner(c,x)

Evaluate a polynomial whose coefficients are given in ascending
order in `c`, at the point `x`, using Horner's rule.
"""
function horner(c,x)
    n = length(c)
    y = c[n]
    for k in n-1:-1:1
        y = x*y + c[k]
    end
    return y
end

# testing lambda functions syntax on page 20 
mycfun(x) = exp(c*sin(x))

c=1; mycfun(3)
ans1 = exp(1*sin(3))
c=2; mycfun(3)
ans2 = exp(2*sin(3))

# poor conditioned quadratic polynomial on page 21 
# p(x) = ax2`+ bx + c considering a = 1, b = ((10^6) + (10^-6)), c = 1
a = 1; b = -((10^6) + (10^-6)); c = 1
@show x1 = (-b + sqrt(b^2 - 4*a*c))/(2*a)
@show x2 = (-b - sqrt(b^2 - 4*a*c))/(2*a)
error = abs(1e-6-x2)/1e-6
@show accurate_digits = -log10(error)

# Backward error analysis on page 24
r = [-2.0,-1,1,1,3,6]
p = FNC.fromroots(r) # construct polynomial from known roots
r̃ = sort(roots(p))
println("Root errors")
@. abs(r-r̃)/r # @. is a macro that broadcasts the operation to all elements of the vector
abs(r[1]-r̃[1]) / r[1] # relative error of first root( just to get a feel for the @. operation)
# the backward error:
p̃ = fromroots(r̃)
#get the coefficiencts errors
c,c̃ = coeffs(p),coeffs(p̃)   
println("Coefficient errors:")
@. abs(c-c̃)/c