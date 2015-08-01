import numpy as np 
import scipy as sp
import math as m 	
import pybrain as pb 			# machine learning for prediction models	
import csv, os				# parsing the input files and iterating over a set of them
import matplotlib as mpl
import matplotlib.pyplot as plt 		# simple plots of the results
import time
import cPickle				# saving long lists into files
import random as rnd

from scipy.optimize import fsolve	# non-linear equation solver 
from scipy import constants

from datetime import datetime
from matplotlib import dates
from matplotlib.cm import autumn

from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.tools.neuralnets import NNregression
from pybrain.supervised.trainers.backprop import BackpropTrainer
from pybrain.structure import SigmoidLayer, LinearLayer, BiasUnit
from pybrain.structure import FullConnection
from pybrain.structure import FeedForwardNetwork
from pybrain.tools.shortcuts import buildNetwork
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.customxml.networkreader import NetworkReader

from math import exp

#########################################################################
############# PARAMETER DEFINITIONS AND MODEL DESCRIPTION  #############
#########################################################################

######## physical constants ######## 

q = sp.constants.e 		# elementary charge 
k = sp.constants.k 		# Boltzmann constant

######## Standard Reference Conditions ######## 

T_ref 	= 25.0 + 273 	# ambient temperature in Kelvin, equal to 25 degrees Celsius
G_ref 	= 1000.0	# surrounding irradiance in W / m2
NOCT 	= 45.0		# Nominal Operating Cell Temperature in Celsius 

Eg_ref	= 1.121	# band gap energy in eV for the Standard Reference Temperature 

epsilon = 1e-7		# magic constant that makes everything work

######## Model parameters ########

### Manufacturer, Model and Version ###
# Kyocera KC85TS #

### Dimensions ###

N_s_cells	= 60.0	# number of cells in series in one module, forming a string
N_p_cells	= 1.0	# number of strings (cells) in parallel in one module
N_s_modules 	= 1.0	# number of modules in series in an array
N_p_modules	= 1.0	# number of modules in parallel in an array

N_p = N_p_cells * N_p_modules 	# total number of parallel cells in the array
N_s = N_s_cells * N_s_modules	# total number of cells in series in the array

N_s_p 	= N_s / N_p 	# ratio of the cells in series to the cells in parallel in the whole array

### Manufacturer specifications under Standard Reference Conditions ###

# FOR ONE CELL
# P_max_ref	= 3.589  	# Watts of maximum power
# V_mpp_ref	= 0.485	# Volts of maximum power voltage
# I_mpp_ref	= 7.4		# Ampers of maximum power current
# V_oc_ref	= 0.6083	# Volts of open circuit voltage
# I_sc_ref 	= 8.1		# Ampers of short circuit current

# t_coeff_voc 	= 0.00105	# V / deg. C - Temperature coefficient of V_oc
# t_coeff_isc 	= -0.0036	# A / deg. C - Temperature coefficient of I_sh

P_max_ref	= 215.0  	# Watts of maximum power
V_mpp_ref	= 29.1		# Volts of maximum power voltage
I_mpp_ref	= 7.4		# Ampers of maximum power current
V_oc_ref	= 36.5		# Volts of open circuit voltage
I_sc_ref 	= 8.1		# Ampers of short circuit current

t_coeff_voc 	= 0.00105	# V / deg. C - Temperature coefficient of V_oc
t_coeff_isc 	= -0.0036	# A / deg. C - Temperature coefficient of I_sh

### Manufacturer specifications under 800 W / m2 and NOCT ###

# P_max	= 87 		# Watts of maximum power
# V_mpp	= 17.4		# Volts of maximum power voltage
# I_mpp	= 5.02		# Ampers of maximum power current
# V_oc		= 21.7		# Volts of open circuit voltage
# I_sc 		= 5.34		# Ampers of short circuit current


########## FEEL FREE TO COMMENT ALL THIS SECTION IF DATA WAS ALREDAY COMPUTED ############

##############################################################################
############## MODELLING EQUATIONS FOR REFERENCE PARAMETERS ##############
##############################################################################

# summarizing the five equations with five unknown
def modelling_equations_ref(initialConditions):			

	I_irr_ref, I_0_ref, n_ref, R_p_ref, R_s_ref = initialConditions

	# Equation #1 sets the current passing through the PV array to 0 and thus we solve it for the open circuit conditions when V_a = V_oc_ref
	eq1 = N_p * I_irr_ref - N_p * I_0_ref * (m.exp((q * V_oc_ref) / (N_s * n_ref * k * T_ref)) - 1.0) - V_oc_ref / (N_s_p * R_p_ref) 

	# Equation #2 sets the voltage to 0 and therefore I_a = I_sc_ref - the short circuit conditions
	# SUSPICIOUS N_p * nkT ?? why not N_s * nkT; Why in paper: last summand nsp/nsp ??
	eq2 = -I_sc_ref + N_p * I_irr_ref - N_p * I_0_ref * (m.exp((q * I_sc_ref * R_s_ref) / (N_p * n_ref * k * T_ref)) - 1.0) - (I_sc_ref * R_s_ref) / (R_p_ref)

	# Equation #3 sets the current of the PV array to the maximum power point at Standard Reference Conditions, V_a should be than equal to V_mpp_ref
	eq3 = -I_mpp_ref + N_p * I_irr_ref - N_p * I_0_ref * (m.exp((q * (V_mpp_ref + I_mpp_ref * N_s_p * R_s_ref)) / (N_s * n_ref * k * T_ref)) - 1.0) - \
		(V_mpp_ref + I_mpp_ref * N_s_p * R_s_ref) / (N_s_p * R_p_ref)

	# Equation #4 solves for the maximum power point where the derivative of P = V_a * I_a at Standard Reference Conditions equals 0
	eq4 = -(I_mpp_ref / V_mpp_ref) + (((q * N_p * I_0_ref) / (N_s * n_ref * k * T_ref)) * \
		m.exp((q * (V_mpp_ref + I_mpp_ref * N_s_p * R_s_ref)) / (N_s * n_ref * k * T_ref)) + 1.0/(N_s_p * R_p_ref)) / \
		(1.0 + (((q * I_0_ref * R_s_ref) / (n_ref * k * T_ref)) * m.exp((q * (V_mpp_ref + I_mpp_ref * N_s_p * R_s_ref)) / (N_s * n_ref * k * T_ref))) + \
		(R_s_ref / R_p_ref))

	# Equation #5 uses the temperature coefficient ot the V_oc and solves for the equation (T - T)*t_coeff_voc = V_oc - V_oc_ref
	eq5 = 	N_p * I_irr_ref - N_p * I_0_ref *(m.exp((q * V_oc_ref) / (N_s * n_ref * k * T_ref)) - 1.0) - V_oc_ref / (N_s_p * R_p_ref)

	return (eq1, eq2, eq3, eq4, eq5)	

########## IV equation for a singe cell at reference conditions ############

def current_ref(I_guess):

	global V, I_irr_ref, I_0_ref, n_ref, R_p_ref, R_s_ref 		# use the globally defined estimated parameters 
	I_ref = I_guess							# set the current as an unknown and use the initial guess

	eq = -I_ref + N_p * I_irr_ref - N_p * I_0_ref * (m.exp((q * (V + I_ref * N_s_p * R_s_ref) / (N_s * n_ref * k * T_ref))) - 1.0) - \
		(V + I_ref * N_s_p * R_s_ref) / (N_s_p * R_p_ref)
	return eq

initialConditions = [8.12, 6e-9, 1.12, 2.09, 0.0056] 
# print "Initial Conditions: ", initialConditions

beginStopWatch = time.time()

I_irr_ref, I_0_ref, n_ref, R_p_ref, R_s_ref = fsolve(modelling_equations_ref, initialConditions)	# solving the equations for the reference conditions
print "Parameters: ", I_irr_ref, I_0_ref, n_ref, R_p_ref, R_s_ref


############### Computing the IV and PV curve for reference conditions ###############

# I_ref = []			# declare container for the current computed at various voltages
# P_ref = []			# declare container for the power computed at various voltages

# step = 0.001			# set iteration step for the numerical approximation of the solution of the IV equation
# v_min = 0.0			# set the starting value of the voltage
# v_max = V_oc_ref + step	# set the last value for the voltage

# v_range = np.arange(v_min, v_max, step)	

# for V in v_range:						# iterate over all voltages and compute the current at this point
# 	initialConditions = [0.5]					# reasonable initial value for the current in order to accelerate convergence
# 	solution = fsolve(current_ref, initialConditions)	# approximate the solution of the equation above

# 	if(solution[0] <= 0):					# exclude the nonsense 
# 		I_ref.append(0.0)
# 		P_ref.append(0.0)
# 		continue
# 	I_ref.append(solution[0])				# put the result into the container in order to be plotted later
# 	P_ref.append(solution[0] * V)				# compute the power P = I * V


# plt.plot(list(v_range), I_ref, 'r')					# plot the IV curve 
# plt.plot(list(v_range), P_ref, 'b')				# plot the PV curve
plt.show(block = True)


#######################################################################################
###############  PARSE INPUT IRRADIANCE AND TEMPERATURE DATA FROM CSV ###############
#######################################################################################

# T_amb = []		# declare container for the ambient temperature
# G = []			# declare container for the local solar irradiance
# Time = []		# declare container for the time stamp

# pathToInputFiles = "input"									# set the directory from which the input files will be read
# inputFiles = os.listdir(pathToInputFiles)							# open directory and load all files in a list
# for inputFile in inputFiles:
# 	if not(inputFile.endswith(".csv")):							# filter the unwanted files
# 		continue
# 	with open(os.path.join(pathToInputFiles, inputFile), 'rb') as inputData:		# open the data file
# 		reader = csv.reader(inputData, delimiter=';', quoting=csv.QUOTE_NONE)	# assign the opened file to a Reader object
# 		rows = list(reader)								# create a list containing the rows of the parsed file
# 		rows = rows[2:]								# delete the first two elements which are the title of the document and the row 
# 		for row in rows:
# 			T_amb.append(row[22])						# append the next irradiance value to the list 
# 			G.append(row[12])							# append the next ambient temperature value to the list
# 			Time.append(row[0])

# T_amb = map(lambda f: float(f.replace(',' , '.')) + 273.0, T_amb)				# convert string float to real floats
# G = map(lambda f: float(f.replace(',' , '.')), G)							# convert string float to real floats
# Time = [datetime.strptime(t, "%d.%m.%Y %H:%M:%S") for t in Time]				# convert string time to datetime


########################################################################################
############### COMPUTING THE PARAMETERS AT NON-REFERENCE CONDITIONS ###############
########################################################################################

def modelling_equations_nonref(T_amb, G):
	
	T = T_amb + ((NOCT - 20.0) / 0.8) * (G / 1000.0)			# computing the cell temperature at a certain ambient temperature
	# print "T = ", T

	I_irr = (G / G_ref) * (I_irr_ref + t_coeff_isc * (T - T_ref))		# computing the irradiance current at non-reference conditions
	# print "I_irr = ", I_irr

	Eg = (1 - 0.0002677 * (T - T_ref)) / Eg_ref				# the band gap energy should also change a bit with change in temperature

	I_0 = I_0_ref * ((T / T_ref) ** 3) * m.exp((q * Eg_ref) / \
		(n_ref * k * T_ref) - (q * Eg_ref) / (n_ref * k * T))		# computing the diode reverse current at non-reference conditions 
	# print "I_0 = ", I_0

	V_oc_t = (V_oc_ref + t_coeff_voc * (T - T_ref))			# the voltage dependent on the cell temperature 
	# print "V_oc_t = ", V_oc_t

	R_p = (G / G_ref) * R_p_ref						# shunt resistance at non-reference conditions
	# print "R_p = ", R_p

	R_s = R_s_ref								# series resistance at non-reference conditions
	# print "R_s = ", R_s

	n = n_ref * (T / T_ref)							# diode ideality factor at non-reference conditions
	# print "n = ", n

	return (T, I_irr, I_0, n, R_p, R_s)

############# IV equation for a singe cell at arbitrary conditions #############
def current_nonref(I_guess):

	global V, I_irr, I_0, n, R_p, R_s 		# use the globally defined estimated parameters 
	I = I_guess					# set the current as an unknown and use the initial guess

	eq = -I + N_p * I_irr - N_p * I_0 * (m.exp((q * (V + I * N_s_p * R_s) / (N_s * n * k * T))) - 1.0) - (V + I * N_s_p * R_s) / (N_s_p * R_p)
	return eq

### Maximum power point of current and voltage ###
def mpp_v_i(initialConditions):

	V_mpp, I_mpp = initialConditions
	global I_irr, I_0, n, R_p, R_s , T_i		# use the globally defined estimated parameters and various temperature

	eq1 = -I_mpp + N_p * I_irr - N_p * I_0 * (m.exp((q * (V_mpp + I_mpp * N_s_p * R_s)) / (N_s * n * k * T_i)) - 1.0) - \
		(V_mpp + I_mpp * N_s_p * R_s) / (N_s_p * R_p)

	eq2 = -(I_mpp / V_mpp) + (((q * N_p * I_0) / (N_s * n * k * T_i)) * m.exp((q * (V_mpp + I_mpp * N_s_p * R_s)) / (N_s * n * k * T_i)) + 1.0/(N_s_p * R_p)) / \
		(1.0 + (((q * I_0 * R_s) / (n * k * T_i)) * m.exp((q * (V_mpp + I_mpp * N_s_p * R_s)) / (N_s * n * k * T_i))) + (R_s / R_p))
	
	return (eq1, eq2)

############################################################################
################### COMPUTE POWER AND STATISTICAL DATA ###################
############################################################################

# P = []				# declare container for the power computed at maximum power point voltage

# epsilon = 0.5
# initialConditions = [0.5, 0.5]	# initial conditions for the V_mpp and I_mpp

# for T_i, G_i in zip(T_amb, G):							# for every temperature and irradiance value
# 	G_i = G_i + epsilon
# 	T_i, I_irr, I_0, n, R_p, R_s = modelling_equations_nonref(T_i, G_i)	# compute the non-reference parameters
# 	V_mpp, I_mpp = fsolve(mpp_v_i, initialConditions)			# compute the V_mpp and I_mpp
# 	P_i = V_mpp * I_mpp
# 	if (P_i < 0.0):	
# 		print "Warning: Illegal negative values: V_mpp =", V_mpp, "I_mpp =", I_mpp, \
# 			"\n Temperature =", T_i, "Irradiance =", G_i
# 		P_i = 0.0
# 	P.append(P_i)								# and compute the power at these points

# stopStopWatch = time.time()
# elapsedTime = stopStopWatch - beginStopWatch

# print "Elapsed Time of non-linear solving:", repr(elapsedTime)

# cPickle.dump(P, open('power_1y.p', 'wb')) 	
# cPickle.dump(Time, open('time_1y.p', 'wb'))				# store parsed and computed lists and save running time during next simulation :)
# cPickle.dump(T_amb, open('temperature_1y.p', 'wb')) 
# cPickle.dump(G, open('irradiance_1y.p', 'wb')) 

############ IF THE DATA WAS ALREADY PARSED AND COMPUTED, COMMENT EVERYTHING ABOVE TILL THE NEXT MARKER ##########
#
P = cPickle.load(open('power_1y.p', 'rb'))
T_amb = cPickle.load(open('temperature_1y.p', 'rb'))
G = cPickle.load(open('irradiance_1y.p', 'rb'))
Time = cPickle.load(open('time_1y.p', 'rb'))

########## Some simple statistics of the power data ###########

mean = np.mean(P)			# takes the mean value of all the data
print "Mean value of power =", mean

stDev = np.std(P)			# takes the standard deviation of the data set
print "Standard deviation of power =", stDev

takeAvgOfDays = 30 * 24 * 60		# time in minutes over which the average will be computed
PTime_avg = zip(*[(float(sum(P[i: i + takeAvgOfDays])) / takeAvgOfDays, \
	Time[(len(P) - 1) if takeAvgOfDays > (len(P) - i) else int(m.floor(i + takeAvgOfDays / 2))]) \
	for i in xrange(0, len(P), takeAvgOfDays)])						# some obscure list comprehension:
												# compute the average of the selected period, take the time in the middle
												# and unzip the list of all these tuples 	

P_avg =  list(PTime_avg[0])		# take the first part which is the mean point of power
P_avg.insert(0, P[0])			# in order the plot to start from the beginning
# print P_avg

Time_avg = list(PTime_avg[1])	# take the time over all these averages
Time_avg.insert(0, Time[0])		# same reason as above
# print Time_avg

# takeMaxOverTime = 1 * 24 * 60	# time in minutes over which the maximum will be computed
# PTime_max = zip(*[(max(P[i: i + takeMaxOverTime]), \
# 	Time[P.index(max(P[i: i + takeMaxOverTime]))]) \
# 	for i in xrange(0, len(P), takeMaxOverTime)])			# another yet less obscure list comprehension:
# 									# compute the max in the selected period and the respective time to that maximum
# 									# and unzip the list of all these tuples 	

# P_max =  list(PTime_max[0])		# take the first part which is the maximum points of power
# P_max.insert(0, P[0])			# in order the plot to start from the beginning add the very first point
# P_max.append(P[len(P) - 1])
# # print P_max

# Time_max = list(PTime_max[1])	# take the time over all these maximum points
# Time_max.insert(0, Time[0])		# same reason as above
# Time_max.append(Time[len(Time) - 1])
# print Time_max

# plt.plot(Time, G, 'g')
# plt.plot(Time, P, 'b')			# plot the power over time
# plt.plot(Time_avg, P_avg, linewidth=2, linestyle="-", c="red")	# plot the average power over the averaged time
# plt.plot(Time_max, P_max, 'g')	# plot the maximum power points over the corresponding time
plt.show()


#######################################################################################
############## USE MACHINE LEARNING TECHNIQUES FOR PREDICTIONS MODELS ############## 
#######################################################################################

########## Create the dataset for the learning neural network ##########

# dataset = SupervisedDataSet(2, 1) 		# specify its dimensions to 2 inputs (irradiance and temperature) and 1 target (power)
# dataset_testing = SupervisedDataSet(2, 1)	# dimensions of the testing dataset should be the same as the training

# T_amb_training = T_amb#[:len(T_amb)/2]	# use first half of input dataset for training purposes
# T_amb_testing = T_amb#[len(T_amb)/2:]	# use the other half for verifying the goodness of the neural network

# G_training = G#[:len(T_amb)/2]			# the same apply for the irradiance
# G_testing = G#[len(T_amb)/2:]

# P_training = P#[:len(T_amb)/2]			# and the power, aka the target data
# P_testing = P#[len(T_amb)/2:]

# Time_training = Time#[:len(Time)/2]		# used by the plotter 
# Time_testing = Time#[len(Time)/2:]

# ########## Normalize dataset in interval [-1 ,1] ##########
# def normalise_data(data):

# 	minv = min(data)			# collect the minimum and maximum for later use when denormalizing 
# 	maxv = max(data)
# 	for i, x in enumerate(data):		# create and affine function which sends the data in the interval [-1, 1]
# 		data[i] = - float(x - maxv) / (minv - maxv) + float(x - minv) / (maxv - minv)

# 	return (minv, maxv)	

# ########## Denormalize dataset back to original state ##########
# def denormalise_data(data, minv, maxv):

# 	for i, x in enumerate(data):		# using the min and max values obtained before normalizing, compute the inverse value with the affine transformation
# 		data[i] = minv * (x - 1) / (-2) + maxv * (x + 1) / 2
		

# min_t_tr, max_t_tr = normalise_data(T_amb_training)	# normalize the dataset and for training and testing
# min_t_tst, max_t_tst = normalise_data(T_amb_testing)	# and extract the minimum for the later use of the denormalizer

# min_g_tr, max_g_tr = normalise_data(G_training)					
# min_g_tst, max_g_tst = normalise_data(G_testing)		# the same applies here

# min_p_tr, max_p_tr = normalise_data(P_training)		# and here
# min_p_tst, max_p_tst = normalise_data(P_testing)


# for t, g, p in zip(T_amb_training, G_training, P_training):		
# 	dataset.addSample((t, g), (p))				# creating the dataset used for training the neural network
# # print "Dataset size =", len(dataset)

# for t, g, p in zip(T_amb_testing, G_testing, P_testing):
# 	dataset_testing.addSample((t, g), (p))			# creating the dataset used for testing and verification of the network
# print "Dataset size =", len(dataset_testing)

# ########## Creating the Neural Network ##########

# inLayer = LinearLayer(2, name = 'in')				# defining the type of the input layer
# hiddenLayer = SigmoidLayer(4, name = 'hidden')		# the hidden layer 
# outLayer = LinearLayer(1, name = 'out')			# the output layer
# biasLayer = BiasUnit(name = 'bias')				# and the bias if necessary. For this model it turns out that it is.

# network = FeedForwardNetwork()				# type of neural network without any loops

# network.addInputModule(inLayer)				# adding the created layers to the network
# network.addModule(hiddenLayer)				# look above
# network.addOutputModule(outLayer)				# look above
# network.addModule(biasLayer)				# adding bias 

# inToHiddenConnection = FullConnection(inLayer, hiddenLayer)		# every neuron from the input layer is connected with every in the hidden
# hiddenToOutConnection = FullConnection(hiddenLayer, outLayer)		# same for the hidden and the output layer
# biasToHiddenConnection = FullConnection(biasLayer, hiddenLayer)		# connecting the bias to the hidden layer
# biasToOutConnection = FullConnection(biasLayer, outLayer)		# and to the out layer respectively

# network.addConnection(inToHiddenConnection)		# connecting the input and the hidden layer
# network.addConnection(hiddenToOutConnection)		# connecting the middle hidden layer and the output layer
# network.addConnection(biasToHiddenConnection)		# connecting the bias and the hidden layer
# network.addConnection(biasToOutConnection)		# connecting the bias and the out layer

# network.sortModules()						# internal topological initialization 

# trainer = BackpropTrainer(network, learningrate = 0.1, lrdecay =  1.0, momentum = 0.7, weightdecay = 0.0)


# colorMap = mpl.cm.autumn_r						# a color map is used to show the evolution of the neural network over time
# numEpochs = 1							# set the number of times the training data is being fed into the neural network (epochs)

# print "MSE before training =", trainer.testOnData(dataset)		# check out how it is doing with the randomly set weights
# for i in range(0, numEpochs):
# 	trainer.trainOnDataset(dataset, 1)				# train for one epoch 
# 	P_training = [] 							# create the container for the obtained power values from the training
# 	for t, g in zip(T_amb_training, G_training):			# for all temperature and irradiance data from the training data set 
# 		P_training.append(network.activate([t, g]))		# compute the output of the neural network
# 	denormalise_data(P_training, min_p_tst, max_p_tst)		# denormalize from [-1, 1] in order to plot the real power values
# 	if (i % 1 == 0):						# plot the first and every 200th mean squared error and power graph
# 		print "MSE after", i + 1, "epoches of training =", trainer.testOnData(dataset)
# 		plt.plot(Time_training, P_training, color = colorMap(i / float(numEpochs)))


# NetworkWriter.writeToFile(network, 'NNparams100.xml')			# save the weight parameters of the neural network in a file
# beginStopWatch = time.time()

# # network = NetworkReader.readFrom('NNparams2000.xml') 		# rebuild the neural network from file - no further configurations are needed
# P_trained = []								# create the container for the output data from the trained network
# for t, g in zip(T_amb_testing, G_testing):						# for every temperature and irradiance value from the testing data set
# 	P_trained.append(network.activate([t, g]))			# store the output 

# denormalise_data(P_trained, min_p_tst, max_p_tst)			# denormalize for the plot
# denormalise_data(P_testing, min_p_tst, max_p_tst)

# stopStopWatch = time.time()

# elapsedTime = stopStopWatch - beginStopWatch
# print "Elapsed time neural network:", elapsedTime

# totalError = m.sqrt((1.0 / float(len(P_testing))) * sum([((x[0] - x[1])**2) for x in zip(P_testing, P_trained)]))
# print "Total Error =", totalError

#""
# ########## Some plots ###########

# # plt.plot(Time_testing, P_trained, 'r')		# plot the result from the neural network with the testing data

# plt.gcf().autofmt_xdate()		# make time axis look better
# plt.show()

###############################################################################################
##### NAIVE APPROACH IN COMPUTING THE POWER USING THE LINEAR CHARACTER OF THE PROBLEM #####
###############################################################################################

def naive_power(G, wG, maxG, maxP):

	# return wG * (maxP / maxG) *  G 
	return 0.87 * I_irr_ref / G_ref * G * V_mpp_ref

def abs_error_naive(naiveP, realP):
	
	return abs(float(sum(realP) - sum(naiveP)) / len(realP))

P_naive = []		# declaring the container for the naively computed power
wG = 1.0		# setting the initial weights of irradiance
# wT = 0.03		# and temperature
maxG = 0.1		# initializing the max values for irradiance
# maxT = 0.0		# temperature
maxP = 215.0		# and power


for g, p in zip(G, P):
	if (maxG < g):
		maxG = g
	if (maxP < p):
		maxP = p
	# if (abs_error_naive())
	P_naive.append(naive_power(g, wG, maxG, maxP))	

plt.plot(Time, P, 'b')
plt.plot(Time, P_naive, 'r')
plt.gcf().autofmt_xdate()		# make time axis look better
plt.show()

########################################################################################
############## SOME SCRATCHED OUT CODE THAT MIGHT BE USEFUL SOMETIMES ##############
########################################################################################

# Printing the weights of the neural network 
# def print_weights():
# 	for module in network.modules:
# 		print "Module:", module.name
# 		if module.paramdim > 0:
# 	    		print "--parameters:", module.params
# 	  	for conn in network.connections[module]:
# 	    		print "-connection to", conn.outmod.name
# 	    		if conn.paramdim > 0:
# 	       			print "- parameters", conn.params
# 	  	if hasattr(network, "recurrentConns"):
# 	    		print "Recurrent connections"
# 	    		for conn in network.recurrentConns:             
# 	       			print "-", conn.inmod.name, " to", conn.outmod.name
# 	       			if conn.paramdim > 0:
# 	          				print "- parameters", conn.params


