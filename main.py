
# coding: utf-8

# In[5]:

#################################################################################################################

# What is sci-kit learn ?
#              sci-kit provides a range of supervised and unsupervised learning algorithms via a consistent 
# interface in python.
# 
# Extensions or modules for scipy are conventionally name scikits. As such the module that provides the learning 
# algorithms is named sklearn (scikit -learn)
# 
# This library focuses on modelling data. It is not focused on loading, manipulating and summarizing data.
# 
# The main advantage of using this library is that the ease of use, code quality, collaboration, documentation and
# performance increases.
# 
# Numpy is the fundamental package for scientific computing in python. It makes the mathematical computation easier               
##################################################################################################################

from sklearn.cluster import KMeans
import numpy as np
import csv
import math
import matplotlib.pyplot
from matplotlib import pyplot as plt


# In[6]:

 
maxAcc = 0.0
maxIter = 0
C_Lambda = 0.03
TrainingPercent = 80
ValidationPercent = 10
TestPercent = 10
M = 10
PHI = []
IsSynthetic = False


# In[158]:

######################################################################################################################

# The method GetTargetVector is used to extract the values from the Querylevelnorm_t.csv file and appended it to the 
# matrix. At the end of the function, it returns the matrix with labels 0,1,2 appended.

######################################################################################################################


def GetTargetVector(filePath):
    t = []
    with open(filePath, 'rU') as f:
        reader = csv.reader(f)
        for row in reader:  
            t.append(int(row[0]))
    #print("Raw Training Generated..")
    return t

######################################################################################################################

# The method GenerateRawData is used to extract the values from the Querylevelnorm_X.csv and the feature values are 
# appended to the matrix. The method retuns a matrix with the feature values

######################################################################################################################

def GenerateRawData(filePath, IsSynthetic):    
    dataMatrix = [] 
    with open(filePath, 'rU') as fi:
        reader = csv.reader(fi)
        for row in reader:
            dataRow = []
            for column in row:
                dataRow.append(float(column))
            dataMatrix.append(dataRow)   
    
    if IsSynthetic == False :
        dataMatrix = np.delete(dataMatrix, [5,6,7,8,9], axis=1)
    dataMatrix = np.transpose(dataMatrix)     
    #print ("Data Matrix Generated..")
    return dataMatrix

#########################################################################################################################

# The method GenerateTrainingTarget takes the labels of the data set and percentage to be split (Here 80%) and returns 
# the amount of training data based on the training percent. Example: A dataset of 100 values and 80 % as training set 
# percentage will get you a matrix of length 80 with values.

#########################################################################################################################

def GenerateTrainingTarget(rawTraining,TrainingPercent = 80):
    TrainingLen = int(math.ceil(len(rawTraining)*(TrainingPercent*0.01)))
    t           = rawTraining[:TrainingLen]
    #print(str(TrainingPercent) + "% Training Target Generated..")
    return t

#########################################################################################################################

# Similarly, the method GenerateTrainingDataMatrix takes the features of the data set and percentage to be split (Here 80%)
# and returns the no of datas of features based on the training percent. 

#########################################################################################################################

def GenerateTrainingDataMatrix(rawData, TrainingPercent = 80):
    T_len = int(math.ceil(len(rawData[0])*0.01*TrainingPercent))
    d2 = rawData[:,0:T_len]
    #print(str(TrainingPercent) + "% Training Data Generated..")
    return d2

##########################################################################################################################

# The method GenerateValData takes the dataset, validation percentage as main inputs and returns the matrix containing
# labels with a size equal to that of validation percentage of the total data.

# Same method is re used when creating the test data as the percentage of both validation and testing set are the same.

##########################################################################################################################


def GenerateValData(rawData, ValPercent, TrainingCount): 
    valSize = int(math.ceil(len(rawData[0])*ValPercent*0.01))
    V_End = TrainingCount + valSize
    dataMatrix = rawData[:,TrainingCount+1:V_End]
    #print (str(ValPercent) + "% Val Data Generated..")  
    return dataMatrix

##########################################################################################################################

# Similarly, the method GenerateValTargetVector takes the dataset, validation percentage as main inputs and returns the 
# matrix containing the features with the size equal to validation percentage of the total data.

# Same method is re used when creating the test data as the percentage of both validation and testing set are the same.

##########################################################################################################################

def GenerateValTargetVector(rawData, ValPercent, TrainingCount): 
    valSize = int(math.ceil(len(rawData)*ValPercent*0.01))
    V_End = TrainingCount + valSize
    t =rawData[TrainingCount+1:V_End]
    #print (str(ValPercent) + "% Val Target Data Generated..")
    return t

##########################################################################################################################

# Method BigSigma:

#                   Inputs ---> Data, MuMatrix, TrainingPercent, IsSynthetic
#                               Data ---> RawData (values of the Features)
#                               MuMatrix ---> Cluster centers obtained from K-Means algorithm.
#                               Training Percent ----> Amount of data to be used for training
#                               IsSynthetic ----> Global bool function.
#               
#                   Process ---> What does this method do?
#                   
#                                Calculates the variances of the given features and create a diagonal matrix
#                               
#                   Output ----> Returns a diagonal matrix containing the variance of the features        

##########################################################################################################################


def GenerateBigSigma(Data, MuMatrix,TrainingPercent,IsSynthetic):
    BigSigma    = np.zeros((len(Data),len(Data)))
    DataT       = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))        
    varVect     = []
    for i in range(0,len(DataT[0])):
        vct = []
        for j in range(0,int(TrainingLen)):
            vct.append(Data[i][j])    
        varVect.append(np.var(vct))
    
    for j in range(len(Data)):
        BigSigma[j][j] = varVect[j]
    if IsSynthetic == True:
        BigSigma = np.dot(3,BigSigma)
    else:
        BigSigma = np.dot(200,BigSigma)
    ##print ("BigSigma Generated..")
    return BigSigma

#######################################################################################################################

# Method GetScalar: (Called by GetRadialBasisOut)
#
#                   Inputs: DataRow, MuRow, BigSigInv

#                           DataRow ----> Corresponding value of the features
#                           MuRow  -----> Corresponding Center of the clusters obtained from K-Means
#                                                   algorithm
#                           BigSigInv --> Inverse of the BigSigma Matrix(which contains the variances) 
#
#                   Process: What will the method do?
#                           
#                            The method calculates a part of the value(A small portion of equation of closed form) 
#                            which method GetRadialBasisOut finally returns
#
#                   Output:  Returns a matrix 
#

#######################################################################################################################

def GetScalar(DataRow,MuRow, BigSigInv):  
    R = np.subtract(DataRow,MuRow)
    T = np.dot(BigSigInv,np.transpose(R))  
    L = np.dot(R,T)
    return L

###############################################################################################################

# Method GetRadialBasisOut: (called by GetPHIMatrix)
# 
#                            Inputs : DataRow, MuRow , BigSigInv
#                       
#                                     DataRow ----> Corresponding value of the features
#                                     MuRow  -----> Corresponding Center of the clusters obtained from K-Means
#                                                   algorithm
#                                     BigSigInv --> Inverse of the BigSigma Matrix(which contains the variances)      
#
#                            Process: What will the method do?

#                                     The method will calculate the value of phi corresponding to the row using
#                                     the formula based on closed form solution with phi(x) being the Gaussian 
#                                     Radial Function.                       
#                           
#                            Output:  The value of Phi corresponding to each row               

###############################################################################################################

def GetRadialBasisOut(DataRow,MuRow, BigSigInv):    
    phi_x = math.exp(-0.5*GetScalar(DataRow,MuRow,BigSigInv))
    return phi_x

###############################################################################################################
# Method GetPhiMatrix:
# 
#                     Inputs : Data, MuMatrix, BigSigma, Training Percent
# 
#                               Data ---> RawData (values of the Features)
#                               MuMatrix ---> Cluster centers obtained from K-Means algorithm.
#                               BigSigma  -----> Diagonal matrix of varainces obtained from method GetBigSigma        
#                               Training Percent ----> Amount of data to be used for training.  
#
#
#                     Process: What will the method do?

#                              The method will call GetRadialBasisOut method and get the value for corresponding
#                              row and it is iterated over the length of the training data set to get the final 
#                              output as a matrix PHI  
#
#                     Output:  Returns a matric PHI containing the a part of the solutions from applying closed 
#                              loop equation for the linear regression model. 
#


###############################################################################################################


def GetPhiMatrix(Data, MuMatrix, BigSigma, TrainingPercent = 80):
    DataT = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))         
    PHI = np.zeros((int(TrainingLen),len(MuMatrix))) 
    BigSigInv = np.linalg.inv(BigSigma)
    for  C in range(0,len(MuMatrix)):
        for R in range(0,int(TrainingLen)):
            PHI[R][C] = GetRadialBasisOut(DataT[R], MuMatrix[C], BigSigInv)
    #print ("PHI Generated..")
    return PHI

########################################################################################################################

# Method GetWeightsClosedForm: 

#                             Inputs: PHI, T, Lambda
#       
#                                     PHI ----> Matrix containing a part of the solution     
#                                      T  ----> Matrix containing the labels of the dataset 
#                                     Lambda -> A constant value       
#
#
#                             Process: What will this method do?
#        
#                                      This function calculates the output of the closed form equation containing the 
#                                      phi(x) as the Gaussian Radial function.
# 
#                             Output:  Matrix containing the weights i.e telling the relation between the training set 
#                                      and features.     

########################################################################################################################

def GetWeightsClosedForm(PHI, T, Lambda):
    Lambda_I = np.identity(len(PHI[0]))
    for i in range(0,len(PHI[0])):
        Lambda_I[i][i] = Lambda
    PHI_T       = np.transpose(PHI)
    PHI_SQR     = np.dot(PHI_T,PHI)
    PHI_SQR_LI  = np.add(Lambda_I,PHI_SQR)
    PHI_SQR_INV = np.linalg.inv(PHI_SQR_LI)
    INTER       = np.dot(PHI_SQR_INV, PHI_T)
    W           = np.dot(INTER, T)
    ##print ("Training Weights Generated..")
    return W

#########################################################################################################################

# Method GetValTest:
#                    Inputs: VAL_PHI, W   
#               
#                            VAL_PHI -------> Output obtained from calling method GetPHIMatrix
#                               W    -------> Output obtained from calling method GetWeightsClosedForm
#
#                    Process: What is the method doing ?
# 
#                            Multiplies the Weight matrix and the transpose of GetPhiMatrix giving the relation between 
#                            the data and features    
#                   
#                    Output: 
#                            A matrix containing relevant data about the relationship between the feature and inputs.                 
#

#########################################################################################################################

def GetValTest(VAL_PHI,W):
    Y = np.dot(W,np.transpose(VAL_PHI))
    ##print ("Test Out Generated..")
    return Y

##########################################################################################################################

# Method GetErms:

#                Inputs: VAL_TEST_OUT, ValDataAct

#                        VAL_TEST_OUT ----> Output obtained from GetValTest
#                        ValDataAct   ----> Original Dataset (Corresponds to training, testing and validation datasets)
#
#                Process: What will the method do?

#                         The method gives us an accuracy measure by comparing the original dataset and results from the  
#                         linear regression model
#
#                Output:
#                         Returns an accuracy measure.         
#           


##########################################################################################################################



def GetErms(VAL_TEST_OUT,ValDataAct):
    sum = 0.0
    accuracy = 0.0
    counter = 0
    for i in range (0,len(VAL_TEST_OUT)):
        sum = sum + math.pow((ValDataAct[i] - VAL_TEST_OUT[i]),2)
        if(int(np.around(VAL_TEST_OUT[i], 0)) == ValDataAct[i]):
            counter+=1
    accuracy = (float((counter*100))/float(len(VAL_TEST_OUT)))
    ##print ("Accuracy Generated..")
    ##print ("Validation E_RMS : " + str(math.sqrt(sum/len(VAL_TEST_OUT))))
    return (str(accuracy) + ',' +  str(math.sqrt(sum/len(VAL_TEST_OUT))))


# ## Fetch and Prepare Dataset

# In[127]:

######################################################################################################################
# RawTarget -------> is a matrix with the relevance label of the row. It contains either of 0, 1, 2. The larger the
#                    relevance variable, the match between query and label is high. 
#    
# RawData ---------> is a matrix with the values of the features. These features are used to train the data using the 
#                    linear regression model. 
#       
######################################################################################################################

RawTarget = GetTargetVector('Querylevelnorm_t.csv')
RawData   = GenerateRawData('Querylevelnorm_X.csv',IsSynthetic)

# ## Prepare Training Data

# In[128]:

######################################################################################################################

# In general, machines are much faster at processing and storing knowledge compared to humans. But, how can we leverage 
# their speed to create intelligent machines ? The answer is training data.

# Algorithms learn from data. They find relationships, develop understanding , make decisions and evaluate their 
# performance from the training data they are given. 

# Training data: Is a labeled data used to train your machine learning algorithm and increase its accuracy.

# The better the training set, better the machine learning model.

# As given in the project description, we have to split the given data set into three sets. One, Training set (80%).
# Two, validation set (10%) and finally testting set (10%)
 
# The below lines of code (From 224 to 249), creates 80% of the training data, 10 % of validation data and 10% of 
# testing data

# For example, a dataset of 1000 entries is splited into 800 training samples , 100 validation and 100 testing samples

######################################################################################################################


TrainingTarget = np.array(GenerateTrainingTarget(RawTarget,TrainingPercent))
TrainingData   = GenerateTrainingDataMatrix(RawData,TrainingPercent)
print(TrainingTarget.shape)
print(TrainingData.shape)


# ## Prepare Validation Data

# In[129]:


ValDataAct = np.array(GenerateValTargetVector(RawTarget,ValidationPercent, (len(TrainingTarget))))
ValData    = GenerateValData(RawData,ValidationPercent, (len(TrainingTarget)))
print(ValDataAct.shape)
print(ValData.shape)


# ## Prepare Test Data

# In[130]:


TestDataAct = np.array(GenerateValTargetVector(RawTarget,TestPercent, (len(TrainingTarget)+len(ValDataAct))))
TestData = GenerateValData(RawData,TestPercent, (len(TrainingTarget)+len(ValDataAct)))
print(ValDataAct.shape)
print(ValData.shape)


#######################################################################################################################

# K-Means Clustering:
#                     K-Means clustering is a type of unsupervised learning which is used when you have unlabeled data.
#                     The algorithm works iteratively to assign each data point to one of the K groups based on the 
#                     features that are provided.
#   
#                     The results of the K-Means clustering are: 
# 
#                     1. The centroids of the K clusters, which can be used to label data.
#                     2. Labels for the training data.                           
#
#
# We use K-means clustering to get a better accuracy in our final result. After the data set is split into training,
# testing and validation data sets. We take the training data set and apply K-Means Clustering algorithm to partition 
# the observation into clusters.

#######################################################################################################################


#######################################################################################################################
# The basic idea of the project is to compare two forms of linear regressions models and compare its performance.
# The two models of comparision are 
#                                   1. Closed Form Solution
#                                   2. Stiochastic Gradient Descent Solution 
# 
# What is Linear Regression? 
#                           Linear Regression is a type of regression analysis where there is a relationship between 
# the independant and dependant variable. 
#
# Linear Regression using Closed Form Solution aka Normal Equation:
#                                                                   It is a O(n^3) algorithm. The formula are well 
# described in the project. The advantage of this model is that it works really well on small data sets.
# 
# Linear Regression using Stiochastic Gradient Descent:
#                                                       The most commonly used optimization algorithm in the machine
# learning is the Stiochastic Gradient Descent. In SGD, we use one example or one training sample at each iteration 
# instead of using whole data set. SGD is used for larger data sets. It is computationally faster and can be trained 
# parallely                                     

#######################################################################################################################



# ## Closed Form Solution [Finding Weights using Moore- Penrose pseudo- Inverse Matrix]

# In[155]:


ErmsArr = []
AccuracyArr = []

kmeans = KMeans(n_clusters=M, random_state=0).fit(np.transpose(TrainingData))
Mu = kmeans.cluster_centers_

###########################################################################################################################

# BigSigma ------> Generates the diagonal matrix containing the variance elements of the features.

# TRAINING_PHI --> Generates a part of the equation in the closed form.

# W -------------> Generates the weights of the features based on the training set using BigSigma and TRAINING_PHI

# TEST_PHI ------> Generates the matrix Test dataset for comparision against W 

# VAL_PHI  ------> Generates the matriX validation dataset for comparision against W

###########################################################################################################################

BigSigma     = GenerateBigSigma(RawData, Mu, TrainingPercent,IsSynthetic)
TRAINING_PHI = GetPhiMatrix(RawData, Mu, BigSigma, TrainingPercent)
W            = GetWeightsClosedForm(TRAINING_PHI,TrainingTarget,(C_Lambda)) 
TEST_PHI     = GetPhiMatrix(TestData, Mu, BigSigma, 100) 
VAL_PHI      = GetPhiMatrix(ValData, Mu, BigSigma, 100)


# In[156]:

# ## Finding Erms on training, validation and test set 

# In[159]:

#############################################################################################################################

# TR_TEST_OUT ------> The values of relationship between training data set and features.

# VAL_TEST_OUT -----> The values of relationship between validaition data set and features.

# TEST_OUT ---------> The value of relationship between test data and features.

#############################################################################################################################


TR_TEST_OUT  = GetValTest(TRAINING_PHI,W)
VAL_TEST_OUT = GetValTest(VAL_PHI,W)
TEST_OUT     = GetValTest(TEST_PHI,W)


##############################################################################################################################

# Training Accuracy : Checks for the accuracy in the Training data features and ground truth using root mean squares.

# Validation Accuracy : Checks for the accuracy in the validation data features and ground truth using root mean squares.

# Test Accuracy : Check for the accuracy in the test data features and ground truth using root mean squares.

##############################################################################################################################



TrainingAccuracy   = str(GetErms(TR_TEST_OUT,TrainingTarget))
ValidationAccuracy = str(GetErms(VAL_TEST_OUT,ValDataAct))
TestAccuracy       = str(GetErms(TEST_OUT,TestDataAct))


# In[160]:


print ('UBITname      = srivenka')
print ('Person Number = 50288730')
print ('----------------------------------------------------')
print ("------------------LeToR Data------------------------")
print ('----------------------------------------------------')
print ("-------Closed Form with Radial Basis Function-------")
print ('----------------------------------------------------')
print ("M = 10 \nLambda = 0.9")
print ("E_rms Training   = " + str(float(TrainingAccuracy.split(',')[1])))
print ("E_rms Validation = " + str(float(ValidationAccuracy.split(',')[1])))
print ("E_rms Testing    = " + str(float(TestAccuracy.split(',')[1])))





# ## Gradient Descent solution for Linear Regression

# In[138]:


print ('----------------------------------------------------')
print ('--------------Please Wait for 2 mins!----------------')
print ('----------------------------------------------------')


# In[ ]:

###################################################################################################################

# The basic idea of implementing the stiochastic Gradient Descent is almost similar to closed loop solution except
# for the change in formula of application. In this formula we have learning rate as an additional factor which 
# determines the final output. After the calculation of the relation between the dataset and features, it is checked
# for accuracy using the Root mean squares (GetErms) function.

####################################################################################################################

W_Now        = np.dot(220, W)
La           = 2
learningRate = 0.03
L_Erms_Val   = []
L_Erms_TR    = []
L_Erms_Test  = []
W_Mat        = []

for i in range(0,400):
    
    #print ('---------Iteration: ' + str(i) + '--------------')
    Delta_E_D     = -np.dot((TrainingTarget[i] - np.dot(np.transpose(W_Now),TRAINING_PHI[i])),TRAINING_PHI[i])
    La_Delta_E_W  = np.dot(La,W_Now)
    Delta_E       = np.add(Delta_E_D,La_Delta_E_W)    
    Delta_W       = -np.dot(learningRate,Delta_E)
    W_T_Next      = W_Now + Delta_W
    W_Now         = W_T_Next
    
    #-----------------TrainingData Accuracy---------------------#
    TR_TEST_OUT   = GetValTest(TRAINING_PHI,W_T_Next) 
    Erms_TR       = GetErms(TR_TEST_OUT,TrainingTarget)
    L_Erms_TR.append(float(Erms_TR.split(',')[1]))
    
    #-----------------ValidationData Accuracy---------------------#
    VAL_TEST_OUT  = GetValTest(VAL_PHI,W_T_Next) 
    Erms_Val      = GetErms(VAL_TEST_OUT,ValDataAct)
    L_Erms_Val.append(float(Erms_Val.split(',')[1]))
    
    #-----------------TestingData Accuracy---------------------#
    TEST_OUT      = GetValTest(TEST_PHI,W_T_Next) 
    Erms_Test = GetErms(TEST_OUT,TestDataAct)
    L_Erms_Test.append(float(Erms_Test.split(',')[1]))


# In[ ]:


print ('----------Gradient Descent Solution--------------------')
print ("M = 15 \nLambda  = 0.0001\neta=0.01")
print ("E_rms Training   = " + str(np.around(min(L_Erms_TR),5)))
print ("E_rms Validation = " + str(np.around(min(L_Erms_Val),5)))
print ("E_rms Testing    = " + str(np.around(min(L_Erms_Test),5)))

