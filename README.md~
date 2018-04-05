# StructuredSpinners
## Some code corresponding to AISTATS 2017 paper "Structured adaptive and random spinners for fast machine learning computations"

This file gives the recipe for reproducing the figures of the paper# 341:
"Structured adaptive and random spinners for fast machine learning computations".
Please have in mind that this code could be optimized and that results could be
different because of randomness.

I] setup

	1) Distribution name and version:
	Distributor ID:	Ubuntu
	Description:	Ubuntu 14.04.5 LTS
	Release:	14.04

	Kernel version:
	Linux – Kernel name
	3.13.0-101-generic – Kernel version number
	x86_64 – Machine hardware name (64 bit)

	2) Languages:
	Python is used with the following setup: "Python 2.7.12 :: Anaconda custom (64-bit)".
	R is used for plotting the curves: R version 3.0.2 (2013-09-25)

3) Dependencies

	For PYTHON:

	FFHT
	We use a highly optimized implementation of the fast Hadamard transform: FFHT from the FALCONN project.
	Link of FALCONN project: https://falconn-lib.org/
	In particular, you can download the corresponding library for FFHT at the following link:
	https://github.com/FALCONN-LIB/FFHT
	Please download the whole file and to install the Python package, run python setup.py install. 
	(For your information, the script example.py shows how to use FFHT from Python.)


II] How to reproduce figure 2: Locality-Sensitive Hashing (LSH) ?

	~ more than 30min to compute

	The corresponding files for this experiment are:
	StructuredMatrices.py
	collision_probability.py
	collision256-64.csv
	plot_collision_probability.R
	collision256-64_Kronecker.pdf
	collision256-64_Kronecker_zoom.pdf
	collision256-64_Kronecker_ref.pdf
	collision256-64_Kronecker_zoom_ref.pdf

	Use the file collision_probability.py.
	This depends on the file StructuredMatrices.py where specific matrix-vector products for structured matrices
	are implemented and on the following Python libraries: math, cmath, numpy, time, ffht and scipy.

	Note that the implementation of Kronecker matrices is not optimized like other structured matrices
	(it does not rely on C code), so the computation time is relatively long is comparison with others.

	To obtain the figures, first run collision_probability.py by writing in the command line:
	python collision_probability.py

	Be careful, by default, the number of points per each interval is 200 (to obtain curves faster) but in the paper 
	we did the experiment for 20000 points (which is very very long). Comment row 471 of the file
	if you want to do the experiment with 20000 points.
	Please not that we can obtain slightly different results (but comparable) because of the randomness.

	It produces the following file: collision256-64.csv
	Then add manually to this file for the first row the following header:
	distance,G,GCIRCULANTEK2K1,GTOEPLITZD2HD1,GSKEWD2HD1,HGAUSSHD2HD1,HD3HD2HD1

	Then, run the file plot_collision_probability.R (with RStudio, row by row) after having changed the path (row 2). It will produce:
	collision256-64_Kronecker.pdf
	and:
	collision256-64_Kronecker_zoom.pdf
	which correspond to the curves of the paper.

	collision256-64_Kronecker_ref.pdf and collision256-64_Kronecker_zoom_ref.pdf are the reference curves
	put in the paper.


III] How to reproduce figure 3: Accuracy of random feature map kernel approximation for the G50C dataset?

	The corresponding files for this experiment are:
	G50C.mat (dataset)
	StructuredMatrices.py
	gaussiankernel_G50C.csv
	angularkernel_G50C.csv
	plot_kernels.R
	gaussiankernel_G50C.pdf
	angularkernel_G50C.pdf

a) Gaussian kernel: left 

	~ 11 min to compute.

	Write kernel = kernel_list[0] and s = non_linearity_list[0] rows 883 and 885 for the gaussian kernel
	in file kernel_approximation_G50C.py.
	Run the file kernel_approximation_G50C.py by writing in the command line:
	python kernel_approximation_G50C.py

	It will produce the file gaussiankernel_G50C.csv. 

	Add manually the following header to file gaussiankernel_G50C.csv:
	n,G,CircK2K1,GTOEPLITZD2HD1,GSKEWD2HD1,HGAUSSHD2HD1,HD3HD2HD1

	Then run rows from 43 to 72 from plot_kernels.R to produce gaussiankernel_G50C.pdf
	Don't forget to change the path row 1.

b) Angular kernel: right

	~ 28 min to compute

	Write kernel = kernel_list[1] and s = non_linearity_list[1] rows 883 and 885 for the angular kernel
	in file kernel_approximation_G50C.py.
	Run the file kernel_approximation_G50C.py by writing in the command line:
	python kernel_approximation_G50C.py

	It will produce the file angularkernel_G50C.csv. 

	Add manually the following header to file angularkernel_G50C.csv:
	n,G,CircK2K1,GTOEPLITZD2HD1,GSKEWD2HD1,HGAUSSHD2HD1,HD3HD2HD1

	Then run rows from 7 to 36 plot_kernels.R to produce gaussiankernel_G50C.pdf

/!\ Please remark that the scale in plot_kernels.R depends significally on the obtained errors.
So you will probably need to change for instance row 13 and row 26 (do the same accordingly for
each kernel and dataset) to fit correctly:
row 13: ylim=c(0.01,0.14)
row 26: abline(h=seq(0.01,0.14,by=0.01),col='darkgray',lty=2,lwd=1)


IV] How to reproduce table 1: Speedups for Gaussian kernel approximation via structured spinners?

	In this experiment, we use synthetic datasets with the following dimensions : 
	2^9, 2^10, 2^11, 2^12, 2^13, 2^14 and 2^15 and we apply the kernel with a square matrix
	with the same dimension in order to exhibit more easily the obtained speedups.

	The corresponding files for this experiment are:
	StructuredMatrices.py
	kernel_approximation_speedup.py
	gausiankernel_speedups.csv

	Run kernel_approximation_speedup.py by writing in the command line:
	OMP_NUM_THREADS=1 python kernel_approximation_speedup.py.

	It will produce the file gausiankernel_speedups.csv where you should add the following header:
	m,GTOEPLITZD2HD1,GSKEWD2HD1,HGAUSSHD2HD1,HD3HD2HD1
	
	It will also print the ratio/speedup of :
	r2 HD3HD2HD1
	r3 GSKEWD2HD1
	r4 HGAUSSHD2HD1
	r5 GTOEPLITZD2HD1

V] How to reproduce table 2: Running time for the MLP, unstructured matrices vs structured spinners?

	~ 1 min to compute

	The corresponding files for this experiment are:
	StructuredMatrices.py
	neural_network_speedup.py

	This depends on the file StructuredMatrices.py where specific matrix-vector products for structured matrices
	are implemented and on the following Python libraries: math, cmath, numpy, time, and ffht.

	Run neural_network_speedup.py by writing in the command line:
	OMP_NUM_THREADS=1 python neural_network_speedup.py

	It will print list of times in second for dimensions from 2^4 to 2^12.


VI] How to reproduce figure 4: Test error for MPL (top) and convolutional network (bottom)?
TODO

VII] How to reproduce figure 5: Accuracy of random feature map kernel approximation for the USPST dataset?

	The corresponding files for this experiment are:
	USPST.mat (dataset)
	StructuredMatrices.py
	gaussiankernel_USPST.csv
	angularkernel_USPST.csv
	plot_kernels.R
	gaussiankernel_USPST.pdf
	angularkernel_USPST.pdf

	Same protocol than for III] but use the file kernel_approximation_USPST.py.
	(~ 3h15 gaussian kernel)
	(~ 8h angular kernel)

At the end, run rows from 82 to 112 plot_kernels.R to produce gaussiankernel_USPST.pdf
or rows from 120 to 150 to produce angularkernel_USPST.pdf

VIII] How to reproduce figure 6: Numerical illustration of the convergence (top) and computational
complexity (bottom) of the Newton sketch algorithm with various structured spinners?
TODO

-NEWTON SKETCHES
TODO

-NEURAL NETWORK
TODO



