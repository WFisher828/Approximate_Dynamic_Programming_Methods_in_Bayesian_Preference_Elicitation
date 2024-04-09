# Approximate_Dynamic_Programming_Methods_in_Bayesian_Preference_Elicitation
 
A collection of code for the experiment ran in "Approximate Dynamic Programming Methods in Bayesian Preference Elicitation".

**Description of Files.** <br />
------

<ins>helper_functions.py</ins>
This is a Python file which contains functions that are used in the questionnaire simulation experiment of the paper "Approximate Dynamic Programming Methods in Bayesian Preference Elicitation". It contains the following functions:
* z_expectation_variance: The function is used to calculate the expectation and variance of the random variable $Z(m,\sigma)$.
* KMS_Matrix: This function is used to generate an nxn matrix $M(r)$ such that $M_{ij} = r^{|i-j|}$ for $0 < r < 1$. This matrix is called a Kac-Murdock-Szego matrix.
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


<ins>Sequential_Experiment_ADP.ipynb</ins>
This is a Jupyter Notebook file which contains code to perform the numerical experiment presented in "Approximate Dynamic Programming Methods in Bayesian Preference Elicitation". To run this file, we must have access to the file helper_functions.

<ins>Plotting.ipynb</ins>
This is a Jupyter Notebook file which contains code to plot images once an experiment is ran using Sequential_Experiment_ADP.ipynb. This file will plot 6 graphs, with four related to D-error reduction and two related to reduction in normalized MSE. The four related to D-error are split upon two factors, those being the "min" vs. "max" cases described in the paper and the "full questionnaire" vs. "last four queries". The two figures related to normalized MSE are split upon the "min" and "max" cases. 

<ins>parameter_file.txt</ins>
This is a txt file which contains all of the possible experiment settings, each row corresponds to a particular combination of settings. Column 1 corresponds to number of attributes. Column 2 corresponds to prior expectation. Column 3 corresponds to prior covariance. Column 4 corresponds to signal-to-noise ratio. Column 5 corresponds to look-ahead horizon. Lastly, column 6 corresponds to orthogonality penalty parameter. See "Running the Experiment" below for levels of each factor (also described in the paper). One may make a file similar to this if they wish to write a batch script to run a subset of multiple experiments in parallel if they have access to a computing cluster.

<ins>Supplementary_Datasets</ins>
This folder contains datasets from experiments similar to the one case presented in the Simulation Study section of our paper. These experiments were ran on Clemson University's Palmetto supercomputing cluster. Each file in this folder corresponds to a combination of experiment settings discussed in the Experiment Setup section of our paper. These files contain the following columns:
* Method: 0 - One-step look-ahead; 1 - Rollout over batch design; 2 - Two-step look-ahead; 3 - Rollout via coordinate exchange
* Min/Max: 0 - Min mse case, 1 - Max mse case.
* Rep: Within each experiment we replicated the questionnaire 50 times for both the min case and the max case. (Range from 1 to 50)
* Question: Within a replication of a questionnaire, we had 16 questions. (Range from 1 to 16)
* Norm_MSE: The normalized MSE value (see Experiment Setup section) at question k under a particular method, Min/Max case, and replication.
* Det: Determinant of the posterior covariance matrix at question k under a particular method, Min/Max case, and replication.
* Hitrate: We did not discuss this metric in the paper due to it giving similar information as Norm_MSE. This metric is reported after each replication of a questionnaire. To calculate the hitrate, we use the final posterior expectation as the estimates of the utility model and check whether the estimated utility model is able to select the same alternative as the true utility model for every question pair. The hitrate is then the proportion of question pairs where the estimated utility model and the true utility model (both without Gumbel error) select the same alternative.
* Prob. Sel: Probability of correct selection. We did not discuss this metric in the paper. This metric is reported once for each method. After a questionnaire replication is complete, we see whether the estimated utility model has the same maximizer as the true utility model. The probability of correct selection is then the proportion of replications where the true utility model and estimated utility model have the same maximizer.
* Mu_i : This is the i-th component of the posterior expectation at question k.
* True_i: This is the i-th component of the true partworth vector (will differ depending on the Min/Max case).
To plot the results of a specific experiment file, one will need to download the experiment file and then use the file Plotting.ipynb (these two files should be in the same folder). Plotting.ipynb directly calls the experiment file by the user specifying the experiment conditions in cell 2 of Plotting.ipynb. 

**Running the Experiment**: <br />
------

<ins>Required Software</ins>: In order to run the experiment one will need to have installed [Python](https://www.python.org/), [Jupyter Notebook](https://jupyter.org/), and [Gurobi](https://www.gurobi.com/) on their machine. 

Next, the python file "helper_functions.py", which contains functions used in the experiment, must be downloaded as the Jupyter Notebook file makes calls to these functions.

Now, we will give directions for running the experiment.

This experiment has a factorial structure, having six experimental settings. See the paper for details of the settings. These settings are:
* number of attributes (6 or 12)
* prior expectation (homogeneous (all ones) or heterogeneous )
* prior covariance (homogeneous diagonal (identity matrix), heterogeneous diagonal, or non-diagonal (the Kac-Murdock-Szego matrix)
* signal-to-noise ratio (low = 0.25, normal = 1.0, high = 4.0)
* Look-ahead Horizon (1/3 of the number of attributes, 2/3 the number of attributes, equal to the number of attributes)
* Orthogonality penalization paramter (small (M = 0.01), large (M= 10.0))

Cell 3 of the Jupyter Notebook "Sequential_Experiment_ADP.ipynb" defines a function which allows the user to pick a combination of settings they wish to investigate. For example, if one wanted to investigate the case number of attributes = 6, homogeneous prior expectation, homogeneous diagonal prior covariance, normal signal to noise ratio, look-ahead horizon equal to 2/3 the number of attributes, and orthogonality penalization parameter equal to 10, the would run experiment_settings(1,1,1,2,2,2). See the options available in Cell 3 for other specifications. 

In Cell 4 of the Jupyter Notebook "Sequential_Experiment_ADP.ipynb", one will see the following 

```python
#EXPERIMENT SETTING ARGUMENTS
attr_exp = int(sys.argv[1])
bp_expect_exp = int(sys.argv[2])
bp_cov_exp = int(sys.argv[3])
signal_noise_exp = int(sys.argv[4])
look_ahead_exp = int(sys.argv[5])
ortho_pen_exp = int(sys.argv[6])
```

Now, to run a particular experiment setting in Jupyter Notebook one will comment out each of the arguments int(sys.argv[-]) and select their preferred settings by choosing a number, using the encoding of options of different settings given in Cell 3.

Alternatively, If one has access to a computing cluster, they may download the notebook as a python file and write a Bash script to run multiple experiment settings at a time using a batch job array. A text file will need
to be defined with each row corresponding to a experiment setting and each column corresponding to number of attributes, prior expectation, prior covariance, signal-to-noise ratio, look-ahead horizon, and orthogonality penalization parameter (these are what the int(sys.argv[-]) are for).

After an experiment is run, a csv file will be returned (see Supplementary_Datasets for examples).

To plot the experiment results, one will use Plotting.ipynb. Make sure the experiment file you are interested in is in the same location as Plotting.ipynb. In Plotting.ipynb, one will specify the corresponding experiment setting in Cell 2, and then run the remaining cells to produce Figures. 



