

***************************************************************************
* authors: Barbara Pascal, Samuel Vaiter, Nelly Pustelnik, Patrice Abry   *
* institution: laboratoire de Physique de l'ENS de Lyon                   *
* date: May 2020                                                          *
* License CeCILL-B                                                        *
***************************************************************************
*********************************************************
* RECOMMENDATIONS:                                   	*
* This toolbox is designed to work with Matlab 2018.b   *
*********************************************************

------------------------------------------------------------------------------------------------------------------------
DESCRIPTION:
Penalized Least Squares are widely used in signal and image processing. Yet, it suffers from a major limitation since it requires fine-tuning of the regularization parameters. Under assumptions on the noise probability distribution, Stein-based approaches provide unbiased estimator of the quadratic risk. The Generalized Stein Unbiased Risk Estimator is revisited to handle correlated Gaussian noise without requiring to invert the covariance matrix. Then, in order to avoid expansive grid search, it is necessary to design algorithmic scheme minimizing the quadratic risk with respect to regularization parameters. This work extends the Stein's Unbiased GrAdient estimator of the Risk of Deledalle et al. to the case of correlated Gaussian noise, deriving a general automatic tuning of regularization parameters. First, the theoretical asymptotic unbiasedness of the gradient estimator is demonstrated in the case of general correlated Gaussian noise. Then, the proposed parameter selection strategy is particularized to fractal texture segmentation, where problem 
formulation naturally entails inter-scale and spatially correlated noise. Numerical assessment is provided, as well as discussion of the practical issues. 


------------------------------------------------------------------------------------------------------------------------
SPECIFICATIONS for using gsugar toolbox:

The main functions are "bfgs_*meth*_gsugar.m" (image) and "bfgs_*meth*_gsugar_1D.m" (signal), with
  meth = 'rof' for T-ROF on linear regression estimate of local regularity
         'joint' for joint fractal texture segmentation
         'coupled' for coupled fractal texture segmentation
         'tv' for usual Total Variation denoising,
performing the segmentation with automatic and data-driven regularization parameter selection.

Demo files are provided in:
- "demo_gsugar.m": fractal texture segmentation
- "demo_gsugar_1D.m": fractal process segmentation
- "demo_gsugar_tv.m": Total Variation denoising for image segmentation
- "demo_gsugar_tv.m": Total Variation denoising for signal segmentation

This toolbox makes use of "GRANSO-master" (required) to perform BFGS quasi-Newton minimization of SURE. 
The toolbox can be dowloaded from http://www.timmitchell.com/software/GRANSO/.
Further, multiscale analysis is performed thanks to the toolbox "toolbox_pwMultiFractal", developped by H. Wendt
(see https://www.irit.fr/~Herwig.Wendt/software.html).
For convenience purposes, copies of the latter toolboxes are provided in the subfolders.

------------------------------------------------------------------------------------------------------------------------
RELATED PUBLICATION:

# B. Pascal, S. Vaiter, N. Pustelnik, P. Abry: Automated data-driven 
selection of the hyperparameters for Total-Variation based texture segmentation, 
(2020) arxiv:2004.09434

------------------------------------------------------------------------------------------------------------------------