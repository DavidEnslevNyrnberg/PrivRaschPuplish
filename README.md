# PrivRaschPuplish
Journal (PAP18) paper - http://kdd.di.unito.it/pap2018/papers/PAP_2018_paper_1.pdf<br />
This is the puplic code for a epsilon-differential private Rasch model

### File setup:
MATLAB/main.m <br />
> setup and plots for each experiment<br />

MATLAB/Experiment#.m<br />
> Workflow for specific setups

MATLAB/raschModel.m<br />
> input -> (rasch_beta, rasch_delta) ; output -> [N,I] probability matrix

MATLAB/rasch_neglog_likelihood.m<br />
> input -> (weights, data, lambda) ; output -> [likelihood, weight_gradient]

MATLAB/rasch_neglog_likelihood_private.m<br />
> input -> (weights, data, lambda, error_vector) ; output -> [likelihood, weight_gradient]

MATLAB/rasch_negloglik_grad_priv_suff.m<br />
> input -> (weights, privateized Xd*, privatized Xn*, lambda) ; output -> [negative_log_likelihood, gradient]<br />
__\* Xd or Xm; the sufficient statistics parameters__

MATLAB/single_beta.m<br />
> input(single beta value, global delta, lambda)  output -> [likelihood, beta_gradient]

MATLAB/suff_stat_parameter_est.m<br />
> input(data, lambda, epsilon, N, I, options) , output -> [privatized beta, privatized delta]

MATLAB/ItalianNetherlandData/<br />
> Public example dataset

### Authors:
MSc Student - Teresa Anna Steiner (s170063@student.dtu.dk)<br /> 
MSc Student - David Enslev Nyrnberg (s123997@student.dtu.dk)<br /> 
Professor - Lars Kai Hansen (lkai@dtu.dk)<br /> 

### Attribution:
When citing this workflow. Please link the paper above and this .git repository
