# Approximate_Dynamic_Programming_Methods_in_Bayesian_Preference_Elicitation
 
A collection of code for the experiment ran in "Approximate Dynamic Programming Methods in Bayesian Preference Elicitation".

**Description of Files.** <br />
------

<ins>helper_functions.py</ins>
This is a Python file which contains functions that are used in the questionnaire simulation experiment of the paper "Approximate Dynamic Programming Methods in Bayesian Preference Elicitation". It contains the following functions:
* z_expectation_variance: The function is used to calculate the expectation and variance of the random variable $Z(m,\sigma)$.
* KMS_Matrix: This function is used to generate an nxn matrix $M(r)$ such that $M_ij = r^|i-j|$ for $0<r<1$. This matrix is called a Kac-Murdock-Szego matrix.
* product_diff_list: This function constructs a set that has all the differences between binary products.
* moment_matching_update: This function is used to perform the moment matching approximation discussed in Section 2 of the paper.
* g_fun: This function is used to evaluate the factor by which a single query reduces D-error. This function is discussed in the section "One-Step Look-Ahead". 
* g_fun_linear_regression: This function is used for doing regression on the log transformation of the function $g(m,\sigma)$. The parameters found using this function are then used to define the linear approximation of the one-step lookahead objective function.
* question_selection_prob: This function is used for calculating the probability of a DM picking x over y, given a prior distribution on their partworth.
* g_opt: This function is used to formulate an optimization problem to get a solution to the one-step lookahead problem.
* two_step_g_acq: This function evaluates the two-step lookahead value of a question given prior parameters mu_0 and Sig_0.
* question_extractor: This function, given a trinary vector of 0, 1, and -1, can find two binary products whose difference is the trinary vector.
* rollout: This function is used to evaluate the rollout value of a query pair (x,y).
* monte_carlo_rollout: This function is used to approximate the rollout value of a query pair (x,y) by performing Monte-Carlo sampling of the paths from the rollout tree to reduce computation time.
* batch_design_delta_penalty: This function constructs a batch design of size k <= (number of attributes) where we enforce mutual Sigma-orthogonality between the k questions. The orthogonality condition makes it so that the D-error minimizationcan be written as a product of g (one-step lookahead) functions. For this function, delta is considered as a continuous variable and is penalized in the objective function to control orthogonality.
* coordinate_exchange_acq: This function is used to perform rollout via coordinate exchange.
* enum_two_step: This function returns a list of all the enumerated two-step acquisition values of a given product set. We use the exact two-step g function. We return the prod_diff_set in case there was sampling done and we have interest in the sampled question pairs. We also return a list of all the first stage and two second stage questions.
* enum_two_step_opt: Given the output (two_step_values and first_second_stage_question) from enum_two_step, this function returns the optimal two-step value along with the first stage question and the two second stage questions which give the optimal value.
* rollout_with_batch_design_acquisition: This function is used for constructing a batch design and performing rollout on this batch design, returning the question amongst the batch that results in the lowest average determinant value.


