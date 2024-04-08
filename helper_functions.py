import numpy as np
import pandas as pd
import math
import gurobipy as gp
from gurobipy import GRB
import itertools
import random
import scipy.integrate
import scipy.stats
from sklearn import linear_model

#Compute expectation and variance of Z random variable parameterized by m and v

def z_expectation_variance(m,v):
    #m is question mean
    #v is question variance
    integ_bound = 30.0
    
    #Set up functions for calculating expectation and variance of Z
    fun1 = lambda z: ((1 + math.e**(-m - math.sqrt(v)*z))**(-1))*(math.e**((-z**2)/2))/math.sqrt(2*math.pi)
    fun2 = lambda z: z*((1 + math.e**(-m - math.sqrt(v)*z))**(-1))*(math.e**((-z**2)/2))/math.sqrt(2*math.pi)
    fun3 = lambda z: z*z*((1 + math.e**(-m - math.sqrt(v)*z))**(-1))*(math.e**((-z**2)/2))/math.sqrt(2*math.pi)
    
    #Calculate expectation and variance of Z. C is the normalization constant that ensures the pdf of Z
    #integrates to be 1. 
    C = scipy.integrate.quad(fun1, -integ_bound, integ_bound)[0]
    mu_z = scipy.integrate.quad(fun2, -integ_bound, integ_bound)[0]/C
    var_z = (scipy.integrate.quad(fun3, -integ_bound, integ_bound)[0] / C) - mu_z**2
    
    return [mu_z, var_z]

#This function is used to generate an nxn matrix M(r) such that M_ij = r^|i-j| for 0<r<1. This matrix is called a 
#Kac-Murdock-Szego matrix.
def KMS_Matrix(n,r):
    #n: this is the number of rows and columns of the matrix
    #r: this is the coefficient given above that determines the value of the matrice's entries 
    
    
    M = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            M[i,j] = r**abs(i-j)
        
    return M

#Define a set that has all the differences between binary products

def product_diff_list(n):
    #n: the number of attributes of the products
    
    #Example of itertools.product:
    #itertools.product(range(2), repeat = 3) --> 000 001 010 011 100 101 110 111
    p_d_l = list(itertools.product([-1.0,0.0,1.0],repeat = n))
    
    #Return the index of the tuple with all 0s.
    zero_index = p_d_l.index(tuple([0]*n))

    #Note that at this point, product_diff_list contains some redundant information. Due to
    #the symmetry of the one-step and two-step acquisition function in terms of question mean, question pairs such as 
    #(-1,-1,-1,...,-1) and (1,1,1,...,1) (i.e. negative multiples) will evaluate as the same under the one-step and two-step
    #acquisition functions. Due to the structure of product_diff_list, we can remove every question pair before and including
    #the question pair with all zero entries in order to remove this redundant information.
    for i in range(0,zero_index + 1):
        p_d_l.pop(0)
    
    p_d_l = [np.array(a) for a in p_d_l]
    return p_d_l

def moment_matching_update(x,y,mu_prior,Sig_prior):
    #x and y are a question pair, x is preferred over y.
    #mu_prior and Sig_prior are expectation and covariance matrix
    #Make sure x, y, mu_prior, and Sig_prior are numpy arrays
    x_vec = np.array(x)
    y_vec = np.array(y)
    mu_prior_vec = np.array(mu_prior)
    Sig_prior_vec = np.array(Sig_prior)
    
    #Define question expectation and question variance
    v = x_vec - y_vec
    mu_v = np.dot(mu_prior_vec,v)
    Sig_dot_v = np.dot(Sig_prior_vec,v)
    Sig_v = np.dot(v,Sig_dot_v)
    
    #Save np.dot(Sig_prior_vec,v) as a variable (DONE)
    
    #Calculate expectation and variance for Z random variable
    
    [mu_z, var_z] = z_expectation_variance(mu_v,Sig_v)
    
    
    #Calculate the update expectation and covariance matrix for 
    #posterior
    mu_posterior = mu_prior_vec + (mu_z/math.sqrt(Sig_v))*Sig_dot_v
    Sig_posterior = ((var_z-1)/Sig_v)*np.outer(Sig_dot_v,Sig_dot_v) + Sig_prior_vec
    
    return mu_posterior, Sig_posterior

def g_fun(m,v,n=2):
    #m and v are question mean and variance arguments
    
    #A bound for integration so to prevent numerical issues.
    integ_bound = 30.0
    
    #fun1 represents the pdf of the Z(m,v) random variable before normalizing
    fun1 = lambda z: ((1 + math.e**(-m - math.sqrt(v)*z))**(-1))*(math.e**((-z**2)/2))/math.sqrt(2*math.pi)
    
    #fun2 will be used for calculating the expectation of Z(m,v)
    fun2 = lambda z: z*((1 + math.e**(-m - math.sqrt(v)*z))**(-1))*(math.e**((-z**2)/2))/math.sqrt(2*math.pi)
    
    #fun3 will be used in calculating the variance of Z(m,v)
    fun3 = lambda z: z*z*((1 + math.e**(-m - math.sqrt(v)*z))**(-1))*(math.e**((-z**2)/2))/math.sqrt(2*math.pi)
    
    #fun4 will be used in calculating the expectation of Z(-m,v)
    fun4 = lambda z: z*(1-(1 + math.e**(-m - math.sqrt(v)*z))**(-1))*(math.e**((-z**2)/2))/math.sqrt(2*math.pi)
    
    #fun5 will be used in calculating the variance of Z(-m,v)
    fun5 = lambda z: z*z*(1-(1 + math.e**(-m - math.sqrt(v)*z))**(-1))*(math.e**((-z**2)/2))/math.sqrt(2*math.pi)
    
    #C is the normalization constant of the pdf for Z. This is the p(m,v) term.
    C = scipy.integrate.quad(fun1, -integ_bound, integ_bound)[0]
    
    #Calculate variance for Z(m,v) and Z(-m,v)
    if C < 10**(-6):
        C = 0
        Sig_z_1 = 1
    else:
        mu_z_1 = scipy.integrate.quad(fun2, -integ_bound, integ_bound)[0]/C
        Sig_z_1 = (scipy.integrate.quad(fun3, -integ_bound, integ_bound)[0]/C) - mu_z_1**2
    
    if (1-C) < 10**(-6):
        C = 1
        Sig_z_2 = 1
    else:
        mu_z_2 = scipy.integrate.quad(fun4, -integ_bound, integ_bound)[0]/(1-C)
        Sig_z_2 = (scipy.integrate.quad(fun5, -integ_bound, integ_bound)[0]/(1-C)) - mu_z_2**2
    
    #g_fun can take on arbitrary n
    return [C*Sig_z_1**(1/float(n)) + (1-C)*Sig_z_2**(1/float(n)), C*Sig_z_1**(1/float(n)), (1-C)*Sig_z_2**(1/float(n))]

#A function for doing linear regression on the log(g_fun).
def g_fun_linear_regression(m_lower,m_upper,v_lower,v_upper,m_axis_num,v_axis_num):
    #m_lower is the lower bound for the grid in variable m
    #m_upper is the upper bound for the grid in variable m
    #v_lower is the lower bound for the grid in variable v
    #v_upper is the upper bound for the grid in variable v
    #m_axis_num and v_axis_num are the number of points used to make the m-axis and v-axis.
    #The total number of grid points is m_axis_num*v_axis_num.
    
    #construct grid data for m and v variables
    m_grid = np.linspace(m_lower, m_upper, num = m_axis_num)
    v_grid = np.linspace(v_lower, v_upper, num = v_axis_num)
    
    #initiate an array to collect data on log(g)
    gfun_array = np.zeros((m_axis_num*v_axis_num,3))
    
    #Format/collect data
    for i in range(0,m_axis_num):
        for j in range(0,v_axis_num):
            gfun_array[i*v_axis_num + j] = [m_grid[i],v_grid[j],math.log(g_fun(m_grid[i],v_grid[j])[0])]
        
    df_gfunction = pd.DataFrame(gfun_array, columns = ['m','v','g'])

    Y = df_gfunction['g'] # dependent variable
    X = df_gfunction[['m', 'v']] # independent variable
    lm = linear_model.LinearRegression()
    lm.fit(X, Y) # fitting the model

    m_coeff = lm.coef_[0]
    v_coeff = lm.coef_[1]

    return[m_coeff,v_coeff]

#This function is used for calculating the probability of a user picking x over y.
def question_selection_prob(mu,Sig,x,y):
    #mu is mean parameter of the prior
    #Sig is the covariance matrix of the prior
    #x is the "preferred" product ( i.e, we will calculate P(x>y) )
    #y is the secondary product
    integ_bound = 30.0
    
    mu,Sig = np.array(mu),np.array(Sig)
    x,y = np.array(x),np.array(y)
    
    #Set up function to calculate the probability of choosing x over y
    m = np.dot(mu,x-y)
    v = np.dot(x-y,np.dot(Sig,x-y))
    fun1 = lambda z: ((1 + math.e**(-m - math.sqrt(v)*z))**(-1))*(math.e**((-z**2)/2))/math.sqrt(2*math.pi)
    
    probability = scipy.integrate.quad(fun1, -integ_bound, integ_bound)[0]
    
    return probability

#Formulate an optimization problem to get the optimal solution to the one-step lookahead problem

def g_opt(mu, Sig, mu_log_coeff, Sig_log_coeff):
    #mu: expectation of prior on beta
    #Sig: covariance matrix of prior on beta
    #mu_log_coeff: the estimated coefficient c_1 that goes with m in the linear model log(g) = c_1*m + c_2*v
    #Sig_log_coeff: the estimated coefficient c_2 that goes with v in the linear model log(g) = c_1m + c_2*v
    
    #n is number of attributes
    n = len(Sig[0])
    
    #Log coefficients:
    mu_s = mu_log_coeff*mu 
    Sig_s = Sig_log_coeff*Sig 
    
    # Create a new model
    m = gp.Model("mip1")
    m.setParam('OutputFlag', 0)
    #m.setParam('MIPGap', 0)
    #m.params.NonConvex = 2
    #m.params.DualReductions = 0
    
    #Set up x and y binary vectors, other variables
    x = m.addMVar(shape = n, vtype = GRB.BINARY, name = "x")
    y = m.addMVar(shape = n, vtype = GRB.BINARY, name = "y")
    
    #Objective function, constant values obtained from R lm function, regression on log(g)
    #for 0<=mu<=3 and 2.5<=sig<=10 
    m.setObjective(mu_s@x - mu_s@y + x@Sig_s@x - y@Sig_s@x - x@Sig_s@y + y@Sig_s@y,
                   GRB.MINIMIZE)

    #Set up constraint so that x and y are different
    m.addConstr(x@x - x@y - y@x + y@y >= 1)
    
    #We want mu(x-y) >= 0 due to the symmetry of the g function
    m.addConstr(mu@x - mu@y >= 0)
    
    m.optimize()
    
    #Return solution x and y
    Vars = m.getVars()
    x_sol = []
    y_sol = []
    for u in range(0,n):
        x_sol.append(Vars[u].x)
        
    for w in range(n,2*n):
        y_sol.append(Vars[w].x)
    
    
    return [m.objVal,x_sol,y_sol]

#A function to evaluate the two_step value of a question given prior parameters mu_0 and Sig_0
#x_0 and y_0 are a question pair.
#This is the exact two-step g value.
def two_step_g_acq(mu_0,Sig_0,mu_log_coeff,Sig_log_coeff,x_0,y_0):
    #mu_0 and Sig_0: These are the parameters (expectation and covariance) of the prior distribution
    #mu_log_coeff and Sig_log_coeff: These are parameters used in the linear model approximation log(g) = c_1*m + c_2*v
    #x_0 and y_0: These are a question pair that we are interested in evaluating
    
    #Ensure that the given arguments are numpy arrays for processing below.
    x_0_vec = np.array(x_0)
    y_0_vec = np.array(y_0)
    mu_0_vec = np.array(mu_0)
    Sig_0_vec = np.array(Sig_0)
    
    #Define first stage variables
    m_0 = np.dot(mu_0_vec,x_0_vec - y_0_vec)
    v_0 = np.dot(x_0_vec-y_0_vec,np.dot(Sig_0_vec,x_0_vec-y_0_vec))
    
    #Gather the posterior information given the two scenarios where the individual picks x over y or they pick
    #y over x
    [mu_10,Sig_10] = moment_matching_update(x_0_vec,y_0_vec,mu_0_vec,Sig_0_vec)
    [mu_11,Sig_11] = moment_matching_update(y_0_vec,x_0_vec,mu_0_vec,Sig_0_vec)
    
    #Solve g_opt(mu_10,Sig_10)[0] and g_opt(mu_11,Sig_11)[0] in order to get optimal questions for each scenario, where
    #each scenario is the individual picking x or y.
    
    [x_10,y_10] = g_opt(mu_10,Sig_10,mu_log_coeff,Sig_log_coeff)[1:]
    [x_11,y_11] = g_opt(mu_11,Sig_11,mu_log_coeff,Sig_log_coeff)[1:]
    
    #Define second stage variables.
    m_10 = np.dot(np.array(mu_10),np.array(x_10) - np.array(y_10))
    v_10 = np.dot(np.array(x_10) - np.array(y_10),np.dot(np.array(Sig_10),np.array(x_10)-np.array(y_10)))
    m_11 = np.dot(np.array(mu_11),np.array(x_11) - np.array(y_11))
    v_11 = np.dot(np.array(x_11) - np.array(y_11),np.dot(np.array(Sig_11),np.array(x_11)-np.array(y_11)))
    
    #Calculate the two-step value. fst_stg_g_sum_term are the two summation terms of g(m_0,v_0)
    fst_stg_g_sum_term = g_fun(m_0,v_0)[1:]
    two_step_g = g_fun(m_10,v_10)[0]*fst_stg_g_sum_term[0] + g_fun(m_11,v_11)[0]*fst_stg_g_sum_term[1]
    
    return [two_step_g,x_10,y_10,x_11,y_11]

#Given a trinary vector of 0, 1, and -1, find two binary products whose difference is the trinary vector.
def question_extractor(prod):
    #prod: This is a trinary vector of 0, 1, and -1 that represents the difference between two products, which
    #are represented by binary vectors.
    
    x = [0]*len(prod)
    y = [0]*len(prod)
    for i in range(0,len(prod)):
        if prod[i] == 1.0:
            x[i] = 1.0
            y[i] = 0.0
        if prod[i] == 0.0:
            x[i] = 0.0
            y[i] = 0.0
        if prod[i] == -1.0:
            x[i] = 0.0
            y[i] = 1.0
    return x,y

#This code is to perform full enumerative rollout 

def rollout(mu,Sig,x,y,rollout_length,mu_log_coeff,Sig_log_coeff):
    #mu: This is the initial mean parameter that we start with
    #Sig: This is the initial covariance matrix we start with
    #x,y: These are the two products (question) that we start the rollout process with
    #rollout_length: This is how far ahead we wish to "rollout" the initial question (x,y) under the one
    #step lookahead base policy
    #mu_log_coeff: the estimated coefficient c_1 that goes with m in the linear model log(g) = c_1*m + c_2*v
    #Sig_log_coeff: the estimated coefficient c_2 that goes with v in the linear model log(g) = c_1m + c_2*v
    #trimming_parameter: This is the probability of success parameter we use in the bernoulli random variable that is used
    #to determine whether we perform trimming at the kth level in the trajectory tree.
    
    #Calculate the probability of user picking x over y and y over x
    prob_x = question_selection_prob(mu,Sig,x,y)
    prob_y = 1.0 - prob_x
    
    #Update mu and Sig depending on if the user picks x or y.
    mu_x,Sig_x = moment_matching_update(x,y,mu,Sig)
    mu_y,Sig_y = moment_matching_update(y,x,mu,Sig)
    
    #Save updated parameters and probabilities
    N_x = [mu_x,Sig_x,prob_x]
    N_y = [mu_y,Sig_y,prob_y]
    
    N = [N_x,N_y]
    
    for i in range(rollout_length):
        #Instantiate a list that will hold nodes at the (i+1)th level after N = [N_x,N_y]
        Node_list = []
        
        for node in N:
            
            #Extract mean, covariance matrix, and (accumulated) probability.
            mu_n = node[0]
            Sig_n = node[1]
            prob_n = node[2]
            
            #Based off of the mean and covariance matrix, find the optimal one-step query, (x_n,y_n)
            x_n,y_n = g_opt(mu_n,Sig_n,mu_log_coeff,Sig_log_coeff)[1:] 
            
            #Calculate the probability of the user choosing xn or yn
            prob_xn = question_selection_prob(mu_n,Sig_n,x_n,y_n)
            prob_yn = 1.0 - prob_xn
            
            
            #Calculate the accumulated probability up to this node
            accumulate_prob_xn = prob_xn * prob_n
            accumulate_prob_yn = prob_yn * prob_n

            #Perform a moment matching update to get new parameters mu and Sig for both cases when the user
            #prefers x_n and y_n.
            mu_nx,Sig_nx = moment_matching_update(x_n,y_n,mu_n,Sig_n)
            mu_ny,Sig_ny = moment_matching_update(y_n,x_n,mu_n,Sig_n)

            #Store the parameters and accumulated probability in nodes.
            N_xn = [mu_nx,Sig_nx,accumulate_prob_xn]
            N_yn = [mu_ny,Sig_ny,accumulate_prob_yn]

            Node_list.append(N_xn)
            Node_list.append(N_yn)

                
        N = Node_list
    
    weighted_det_sum = sum(np.sqrt(np.linalg.det(node[1]))*node[2] for node in N)
    
    return weighted_det_sum

#This code is used to perform rollout using monte-carlo method

def monte_carlo_rollout(mu,Sig,x,y,rollout_length,mu_log_coeff,Sig_log_coeff,sample_budget):
    #mu: This is the initial mean parameter that we start with
    #Sig: This is the initial covariance matrix we start with
    #x,y: These are the two products (question) that we start the rollout process with
    #rollout_length: This is how far ahead we wish to "rollout" the initial question (x,y) under the one
    #step lookahead base policy
    #mu_log_coeff: The estimated coefficient c_1 that goes with m in the linear model log(g) = c_1*m + c_2*v
    #Sig_log_coeff: The estimated coefficient c_2 that goes with v in the linear model log(g) = c_1m + c_2*v
    #sample_budget: This is the number of trajectories that we want to use.
    
    #Calculate the probability of user picking x over y and y over x
    Sig_list = []
    
    prob_x = question_selection_prob(mu,Sig,x,y)
    prob_y = 1.0 - prob_x
    
    #Look at which product is preferred
    if prob_x >= prob_y:
        prefer_prob = prob_x 
        prefer_product = x
        not_prefer_product = y
    else:
        prefer_prob = prob_y 
        prefer_product = y
        not_prefer_product = x
    
    #Sample 'sample_budget' number of trajectories.
    for i in range(sample_budget):
        
        #Create a bernoulli random variable with probability parameter equal to the probability of the product having
        #the higher probability of selection between x and y.
        path_selector = scipy.stats.bernoulli.rvs(prefer_prob)
        
        #Perform moment matching according to the value of the bernoulli random variable, path_selector. If path_selector
        #is 1.0, we will go down the branch with higher probability. If path_selector is 0.0, we will go down the branch 
        #with lower probability.
        if path_selector == 1.0:
            mu_n,Sig_n = moment_matching_update(prefer_product,not_prefer_product,mu,Sig)
        else:
            mu_n,Sig_n = moment_matching_update(not_prefer_product,prefer_product,mu,Sig)
        
        #We start the rollout process on initial question (x,y)
        for j in range(rollout_length):
            
            #Solve the one-step lookahead problem to get the next question pair
            x_n,y_n = g_opt(mu_n,Sig_n,mu_log_coeff,Sig_log_coeff)[1:] 
            
            #Calculate the probability of x_n and y_n given mu_n and Sig_n
            prob_xn = question_selection_prob(mu_n,Sig_n,x_n,y_n)
            prob_yn = 1.0 - prob_xn
            
            #Check which of x_n and y_n has a higher probability. 
            if prob_xn >= prob_yn:
                prefer_prob_n = prob_xn 
                prefer_product_n = x_n
                not_prefer_product_n = y_n
            else:
                prefer_prob_n = prob_yn 
                prefer_product_n = y_n
                not_prefer_product_n = x_n
            
            #Create a bernoulli random variable with parameter prefer_prob_n.
            path_selector_n = scipy.stats.bernoulli.rvs(prefer_prob_n)
            
            #Perform moment matching according to the value of the bernoulli random variable, path_selector_n. 
            #If path_selector_n
            #is 1.0, we will go down the branch with higher probability. If path_selector is 0.0, we will go down the branch 
            #with lower probability.
            if path_selector_n == 1.0:
                mu_n,Sig_n = moment_matching_update(prefer_product_n,not_prefer_product_n,mu_n,Sig_n)
            else:
                mu_n,Sig_n = moment_matching_update(not_prefer_product_n,prefer_product_n,mu_n,Sig_n)
                
        
        #After we finish one trajectory, we append the resulting covariance matrix to a list.
        Sig_list.append(Sig_n)
        
    
    #Calculate an estimate for the determinant. We use the sample average of the determinants of the covariances coming from
    #different trajectories.
    determinant_estimate = (1/sample_budget)*sum(np.sqrt(np.linalg.det(S)) for S in Sig_list)
    
    return determinant_estimate

#This function constructs a batch design of size k <= (number of attributes) where we enforce mutual 
#Sigma-orthogonality between the k questions. The orthogonality condition makes it so that the D-error minimization
#can be written as a product of g (one-step lookahead) functions. For this function, delta is considered as a continuous
#variable and is penalized in the objective function.

def batch_design_delta_penalty(mu,Sig,batch_size,mu_log_coeff,Sig_log_coeff,M=100,t_lim=100):
    #mu: expectation of prior on beta
    #Sig: Covariance matrix of prior on beta
    #batch_size: the number of questions we want to return in our batch design. This should be less or equal to the number
    #of attributes
    #mu_log_coeff: the estimated coefficient c_1 that goes with m in the linear model log(g) = c_1*m + c_2*v
    #Sig_log_coeff: the estimated coefficient c_2 that goes with v in the linear model log(g) = c_1m + c_2*v
    #M: this is a parameter which will be used as a constant to penalize the orthogonality constraint term delta.
    #t_lim: a time limit on the running time of the optimization procedure. Not sure if t=100 is sufficient at the moment.
    
    #This is the number of attributes for the products
    n = len(Sig[0])
    
    #These are terms corresponding to the linear and quadratic terms in the objective function.
    mu_s = mu_log_coeff*mu
    Sig_s = Sig_log_coeff*Sig
    
    m = gp.Model("mip1")
    m.setParam('Timelimit',t_lim)
    m.setParam('OutputFlag', 0)
    #m.params.NonConvex = 2
    
    #Set up the x_i and y_i, i = 1,...,batchsize
    X = m.addMVar((batch_size,n),vtype = GRB.BINARY)
    Y = m.addMVar((batch_size,n),vtype = GRB.BINARY)
    delta = m.addVar(lb=0.0, vtype = GRB.CONTINUOUS)
    
    #Set up the objective function, which is the sum of (batch_size) linearized g functions.
    m.setObjective(sum([mu_s@X[i] - mu_s@Y[i] + X[i]@Sig_s@X[i] -X[i]@(2.0*Sig_s)@Y[i] + 
                   Y[i]@Sig_s@Y[i]  for i in range(batch_size)]) + M*delta,GRB.MINIMIZE)
    
    #Set up the constraints that force the products in question i to be different, as well as forcing the symmetry
    #exploitation condition.
    for i in range(batch_size):
        m.addConstr(X[i]@X[i] - X[i]@Y[i] - Y[i]@X[i] + Y[i]@Y[i] >= 1)
        m.addConstr(mu@X[i] - mu@Y[i] >= 0)
    
    #Set up the Sigma-orthogonality constraint for all questions i and j, i not equal to j.
    for i in range(batch_size):
        for j in range(i+1,batch_size):
            m.addConstr(X[i]@Sig@X[j] - X[i]@Sig@Y[j] - Y[i]@Sig@X[j] + Y[i]@Sig@Y[j] - delta <= 0)
            m.addConstr(X[i]@Sig@X[j] - X[i]@Sig@Y[j] - Y[i]@Sig@X[j] + Y[i]@Sig@Y[j] + delta >= 0)
    
    m.optimize()
    
    #This will be the list of products
    Q = [ [] for i in range(batch_size)]
    
    for i in range(batch_size):
        Q[i].append(X[i].X)
        Q[i].append(Y[i].X)
        
    return[Q,delta.X]

def coordinate_exchange_acq(mu,Sig,mu_log_coeff,Sig_log_coeff,batch_size,rollout_length,MC_budget,rel_gap_threshold,
                           include_batch = False, include_one_step = True):
    #mu: This is the initial mean parameter that we start with
    #Sig: This is the initial covariance matrix we start with
    #mu_log_coeff: The estimated coefficient c_1 that goes with m in the linear model log(g) = c_1*m + c_2*v
    #Sig_log_coeff: The estimated coefficient c_2 that goes with v in the linear model log(g) = c_1m + c_2*v
    #batch_size: the number of questions we want to return in our batch design. This should be less or equal to the number
    #of attributes
    #rollout_length: This is how far ahead we wish to "rollout" the initial question (x,y) under the one
    #step lookahead base policy
    #MC_budget: This is the budget we allow for monte carlo method of rollout. At this point in time, we will use MC if
    #rollout_length is greater than 8.
    #rel_gap_threshold: This is used to see if the perturbed question outperforms the current question
    #include_batch: This determines whether we also iterate over the batch questions.
    #include_one_step: This determines whether we want to include the one-step optimal question within our batch. This can
    #help ensure that rollout performs at least as well as one-step look ahead.
    
    #question_component tells us how many components are in (x,y)
    attr = len(mu)
    question_component = 2*attr
    
    #Used to store the initial set of questions coming from batch and one-step method
    init_set_of_questions = []
    
    #Used to store the set of questions after performing coordinate exchange on the initial batch
    final_set_of_questions = []
    
    #This will store the rollout values of the final set of questions
    final_det_values = []
    
    #Determine if the initial set of questions includes the batch design
    if include_batch:
        init_set_of_questions = batch_design_delta_penalty(mu,Sig,batch_size,mu_log_coeff,Sig_log_coeff)[0]
    
    #Determine if the initial set of questions includes the one-step optimal solution
    if include_one_step:
        [one_step_x,one_step_y] = g_opt(mu,Sig,mu_log_coeff,Sig_log_coeff)[1:]
        init_set_of_questions.append([one_step_x,one_step_y])
    
    #Need to use monte_carlo if rollout_length is greater than 8
    if rollout_length>=8:
        #iterate over all the questions
        for question in init_set_of_questions:
            current_question = [q[:] for q in question]#question[:]
            x = current_question[0]
            y = current_question[1]
            #perform monte carlo rollout on the current question. Store its value
            current_roll_value = monte_carlo_rollout(mu,Sig,x,y,rollout_length,mu_log_coeff,Sig_log_coeff,MC_budget)
            #This counter is used to determine when the coordinate exchange process stops for this question.
            counter = 0
            #Begin the coordinate exchange process for the current question
            while counter < question_component:
                #Initiate a variable called perturb question. This question will start as the same as the current question,
                #but later on we will change a single attribute in one of the products and see if that improves the rollout
                #value.
                perturb_question = [cq[:] for cq in current_question]
                #If the counter is less than the number of product attributes, we will change one of the entries in 'x'
                if counter < attr:
                    #Changing an entry in 'x'
                    perturb_question[0][counter] = abs(1.0-current_question[0][counter])
                    x_perturb = perturb_question[0]
                    y_perturb = perturb_question[1]
                    #Make sure the perturbed 'x' and 'y' are not equal. If so, check its rollout value.
                    if np.dot(np.array(x_perturb)-np.array(y_perturb),np.array(x_perturb)-np.array(y_perturb))>0:
                        perturb_roll_value = monte_carlo_rollout(mu,Sig,x_perturb,y_perturb,rollout_length,mu_log_coeff,
                                                                 Sig_log_coeff,MC_budget)
                #If counter is greater than number of product attributes, we will change one of the entries in 'y'.
                else:
                    perturb_question[1][counter-attr] = abs(1.0 - current_question[1][counter-attr])
                    x_perturb = perturb_question[0]
                    y_perturb = perturb_question[1]
                    if np.dot(np.array(x_perturb)-np.array(y_perturb),np.array(x_perturb)-np.array(y_perturb))>0:
                        perturb_roll_value = monte_carlo_rollout(mu,Sig,x_perturb,y_perturb,rollout_length,mu_log_coeff,
                                                                 Sig_log_coeff,MC_budget)
                
                #If the perturbed question's rollout value outperforms the current question's rollout value by some
                #relative threshold, then we will replace the current question with the perturbed question
                if (current_roll_value - perturb_roll_value)/current_roll_value >= rel_gap_threshold:
                    current_question = [q[:] for q in perturb_question]#perturb_question
                    current_roll_value = perturb_roll_value
                    counter = 0
                else:
                    counter = counter + 1
            
            #After the coordinate exchange process, we place the resulting question and its rollout value in a list.
            final_set_of_questions.append(current_question)
            final_det_values.append(current_roll_value)
    
    #Use regular rollout if rollout length is less than 8. Same coordinate exchange process as in the 
    #monte carlo rollout method    
    else:
        for question in init_set_of_questions:
            current_question = [q[:] for q in question]
            x = current_question[0]
            y = current_question[1]
            current_roll_value = rollout(mu,Sig,x,y,rollout_length,mu_log_coeff,Sig_log_coeff)
            counter = 0
            while counter < question_component:
                perturb_question = [cq[:] for cq in current_question]
                if counter < attr:
                    perturb_question[0][counter] = abs(1.0-current_question[0][counter])
                    x_perturb = perturb_question[0]
                    y_perturb = perturb_question[1]
                    if np.dot(np.array(x_perturb)-np.array(y_perturb),np.array(x_perturb)-np.array(y_perturb))>0:
                        perturb_roll_value = rollout(mu,Sig,x_perturb,y_perturb,rollout_length,mu_log_coeff,
                                                                 Sig_log_coeff)
                else:
                    perturb_question[1][counter-attr] = abs(1.0 - current_question[1][counter-attr])
                    x_perturb = perturb_question[0]
                    y_perturb = perturb_question[1]
                    if np.dot(np.array(x_perturb)-np.array(y_perturb),np.array(x_perturb)-np.array(y_perturb))>0:
                        perturb_roll_value = rollout(mu,Sig,x_perturb,y_perturb,rollout_length,mu_log_coeff,
                                                                 Sig_log_coeff)
                    
                if (current_roll_value - perturb_roll_value)/current_roll_value >= rel_gap_threshold:
                    current_question = [q[:] for q in perturb_question]
                    current_roll_value = perturb_roll_value
                    counter = 0
                else:
                    counter = counter + 1
                
            final_set_of_questions.append(current_question)
            final_det_values.append(current_roll_value)
    
    #Find the minimimum rollout value among all the questions that have went through coordinate exchange. Pick the one with
    #the smallest rollout value.
    min_index = np.argmin(np.array(final_det_values))
    rollout_coordinate_opt_question = final_set_of_questions[min_index]
    
    return rollout_coordinate_opt_question

#A function which returns a list of all the enumerated two-step acquisition values of a given product set. We use the exact
#two-step g function. We return the prod_diff_set in case there was sampling done and we have interest in the sampled
#question pairs. We also return a list of all the first stage and two second stage questions.
def enum_two_step(mu_vec, Sig_mat, mu_log_coeff, Sig_log_coeff, prod_diff_set, prod_samp_num = 0):
    #mu_vec: expectation of prior on beta
    #Sig_mat: covariance matrix of prior on beta
    #mu_log_coeff and Sig_log_coeff: coefficients that are used in the optimization prblem.
    #prod_diff_set: set of question pairs we are enumerating over. This should be created by using product_diff_list.
    #prod_samp_num: Should be a positive integer. Used in obtaining a random sample of product pairs if needed.
    
    #Sample a number of product pairs from the product pairs list, if needed in the case where
    #there are a large number of attributes. Will possibly need a seed for random.sample()
    if prod_samp_num>0:
        prod_diff_set = random.sample(prod_diff_set,prod_samp_num)
    
    #define a list to store enumerated two-step values, along with a list to save the first stage and two second stage
    #questions.
    prod_diff_len = len(prod_diff_set)
    two_step_g_val = [0]*prod_diff_len
    first_stage_second_stage_questions = [0]*prod_diff_len
    
    #calculate two-step values for all question pairs
    for i in range(0, prod_diff_len):
        x_0,y_0 = question_extractor(prod_diff_set[i])
        two_step = two_step_g_acq(mu_vec,Sig_mat,mu_log_coeff,Sig_log_coeff,x_0,y_0)
        two_step_g_val[i] = two_step[0]
        first_stage_second_stage_questions[i] = [x_0,y_0,two_step[1],two_step[2],two_step[3],two_step[4]]
        
    return two_step_g_val,prod_diff_set,first_stage_second_stage_questions

#A function which returns the best performing solution and their two-step acquisition values, along with the corresponding
#first stage and two second stage questions.
#We use the exact two-step g function. Used for experimenting and gaining insight into two-step acquisition.

def enum_two_step_opt(two_step_values,first_second_stage_question):
    #two_step_values: A list of two_step values. This should come from the function enum_two_step
    
    
    #Find min index of the two-step enumeration. Use these values to return the optimal two-step value,
    #along with their corresponding questions.
    two_step_array = np.array(two_step_values)
    min_index = np.argmin(two_step_array)
    max_index = np.argmax(two_step_array)
    opt_x0 = first_second_stage_question[min_index][0]
    opt_y0 = first_second_stage_question[min_index][1]
    opt_x10 = first_second_stage_question[min_index][2]
    opt_y10 = first_second_stage_question[min_index][3]
    opt_x11 = first_second_stage_question[min_index][4]
    opt_y11 = first_second_stage_question[min_index][5]
    opt_val = two_step_values[min_index]

    return [opt_val,opt_x0,opt_y0,opt_x10,opt_y10,opt_x11,opt_y11]

#This function is used for constructing a batch design and performing rollout on this batch design, returning the question
#amongst the batch that results in the lowest average determinant value

def rollout_with_batch_design_acquisition(mu,Sig,mu_log_coeff,Sig_log_coeff,batch_size,rollout_length,MC_budget,include_one_step = False,
                                          penalty_term = 100):
    #mu: This is the initial mean parameter that we start with
    #Sig: This is the initial covariance matrix we start with
    #mu_log_coeff: The estimated coefficient c_1 that goes with m in the linear model log(g) = c_1*m + c_2*v
    #Sig_log_coeff: The estimated coefficient c_2 that goes with v in the linear model log(g) = c_1m + c_2*v
    #batch_size: the number of questions we want to return in our batch design. This should be less or equal to the number
    #of attributes
    #rollout_length: This is how far ahead we wish to "rollout" the initial question (x,y) under the one
    #step lookahead base policy
    #MC_budget: This is the budget we allow for monte carlo method of rollout. At this point in time, we will use MC if
    #rollout_length is greater than 8.
    #include_one_step: This determines whether we want to include the one-step optimal question within our batch. This can
    #help ensure that rollout performs at least as well as one-step look ahead. Default value is False.
    #penalty_term: This is used to set the penalty level for orthogonality in the orthogonal batch design optimization problem. A higher penalty
    #term will lead to a more Sigma_orthogonal design, while a lower penalty term will lead to less Sigma_orthogonality in the design
    
    #Construct the batch based off of mu,Sig, and batch_size
    batch = batch_design_delta_penalty(mu,Sig,batch_size,mu_log_coeff,Sig_log_coeff,M = penalty_term)[0]
    
    #If desired, include the one-step look ahead optimal question within this batch to help ensure performance
    #is at least as good as one-step look ahead
    if include_one_step:
        [one_step_x,one_step_y] = g_opt(mu,Sig,mu_log_coeff,Sig_log_coeff)[1:]
        batch.append([one_step_x,one_step_y])
    #For each question in the batch, perform rollout of length rollout_length and save the average of the determinant
    #values for each question. If rollout_length is greater than or equal to 8, use MC method instead, 
    #as it seems enumeration becomes slow after this point.
    cov_avg_det_values = []
    if rollout_length >= 8:
        for question in batch:
            x = question[0]
            y = question[1]
            avg_det_value = monte_carlo_rollout(mu,Sig,x,y,rollout_length,mu_log_coeff,Sig_log_coeff,MC_budget)
            cov_avg_det_values.append(avg_det_value)
    else:
        for question in batch:
            x = question[0]
            y = question[1]
            avg_det_value = rollout(mu,Sig,x,y,rollout_length,mu_log_coeff,Sig_log_coeff)
            cov_avg_det_values.append(avg_det_value)
    
    
    #Pick the question with the lowest average determinant value. Call it opt_question
    min_index = np.argmin(np.array(cov_avg_det_values))
    opt_question = batch[min_index]
    
    return opt_question

