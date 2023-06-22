import argparse
import os
import subprocess
from shutil import copyfile
import pandas as pd
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.stats import gmean
from sklearn import neighbors
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
import pickle
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import wavfile
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import MidTermFeatures
from sklearn import preprocessing

def normalize_L2(X_train, X_test):
  # we create this function in order to scale the input features to ensure that no feature has an undue influence on the model due to its scale or unit of measurement
  norm = Normalizer(norm='l2')
  X_training_l2 = norm.transform(X_train)
  X_test_l2 = norm.transform(X_test)
  return X_training_l2, X_test_l2

def PCA_decomposition(X_train,X_test):
  # we use this function in order to reduce the number of features to improve computational efficiency, reduce noise, or prevent overfitting.
    pca=PCA()
    pca.fit(X_train)
    X_train_pca=pca.transform(X_train)
    X_test_pca=pca.transform(X_test)
    return X_train_pca,X_test_pca, pca

def Logistic_regresion(X_train, y_train):
  #in this function we classyfy output based on model and data
  Logisricregresion= LogisticRegression()
  Logisricregresion.fit(X_train, y_train.ravel())
  return Logisricregresion

def Bayes_beroulii(X_train, y_train):
    # here we apply Baysien statistic for classification
    Bernie = BernoulliNB()
    Bernie.fit(X_train, y_train.ravel())
    return Bernie

def Neighbour_fitting(X_train,y_train,_algorithm="",_weights="uniform"):
  #we do fitting heighbour to measure instance based on its similarity to other instances in the training dataset.
  #algorithm: - "ball_tree","kd_tree","brute","auto".
  #weights:- "uniform"(default), "distance".

    if(_algorithm==""):
        neighbourd = neighbors.KNeighborsClassifier(11, weights=_weights)
    else:
        neighbourd = neighbors.KNeighborsClassifier(11, algorithm=_algorithm, weights=_weights)
    neighbourd.fit(X_train, y_train.ravel())
    return neighbourd

def SVM_fitting(X_train, y_train, _gamma="auto"):
  # we use this finding the optimal decision boundary that separates the classes
  svmfitt = svm.NuSVC(gamma=_gamma)
  svmfitt.fit(X_train, y_train.ravel())
  return svmfitt

def twoD(logisticReg, X_train, y_train):
   #this is for visualing binary classification
    x_f = X_train[:, 0]
    y_f = X_train[:, 1]
    plt.figure()
    plt.plot(x_f[y_train == 0], y_f[y_train == 0], "or")
    plt.plot(x_f[y_train == 1], y_f[y_train == 1], "og")
    thetaN = logisticReg.coef_
    theta0 = logisticReg.intercept_
    theta1 = thetaN[0][0]
    theta2 = thetaN[0][1]
    x = np.array([-0.9, 0.9])
    y = -((theta0 + theta1) * x) / (theta2)
    plt.plot(x, y)
    plt.show()

def threeD(logreg, X_train, y_train):

  plt.figure()
  plt.subplot(111, projection="3d")
  x_f = X_train[:, 0]
  y_f = X_train[:, 1]
  z_f = X_train[:, 2]
  plt.plot(x_f[y_train == 0], x_f[y_train == 0], z_f[y_train == 0], "or")
  plt.plot(x_f[y_train == 1], y_f[y_train == 1], z_f[y_train == 1], "og")
  thetaN = logreg.coef_
  theta0 = logreg.intercept_
  theta1 = thetaN[0][0]
  theta2 = thetaN[0][1]
  theta3 = thetaN[0][2]
  x = np.array([-0.9, 0.9])
  y = np.arange(-0.9, 0.9)
  x, y = np.meshgrid(x, y)
  z = -(theta0 + theta1 * x + theta2 * y) / (theta3)
  plt.gca().plot_surface(x, y, z, shade=False, color='y')
  plt.show();

def Scoring(lr, x_test, y_test):
    from sklearn.metrics import accuracy_score, confusion_matrix
    p_test = lr.predict(x_test)
    return accuracy_score(y_test, p_test), confusion_matrix(y_test, p_test)
# TODO: Add file to save models in
def One_to_rule_them_all(x_train, y_train, x_test, y_test):
  Model_logistic_regresion(x_train, y_train, x_test, y_test)
  Model_Bayes_bernouli(x_train, y_train, x_test, y_test)
  Model_svm(x_train, y_train, x_test, y_test)
  Model_neighbour(x_train, y_train, x_test, y_test)

def Model_logistic_regresion(x_train, y_train, x_test, y_test):
  logreg = Logistic_regresion(x_train, y_train)
  acc, conf_matrix = Scoring(logreg, x_test, y_test)
  print("LogReg: \t", acc)
  save_model(logreg, "models/Logistic_regresion_model")

def Model_Bayes_bernouli(x_train, y_train, x_test, y_test):
  Bernie = Bayes_beroulii(x_train, y_train)
  acc, conf_matrix = Scoring(Bernie, x_test, y_test)
  print("Bernie: \t", acc)
  save_model(Bernie, "models/Bernouli_model")

def Model_svm(x_train, y_train, x_test, y_test):
  svm = SVM_fitting(x_train, y_train, _gamma="scale")
  acc, conf_matrix = Scoring(svm, x_test, y_test)
  print("SVM: \t", acc)
  save_model(svm, "models/svm_model")

def Model_neighbour(x_train, y_train, x_test, y_test):
  knn = Neighbour_fitting(x_train, y_train, _algorithm="ball_tree", _weights="distance")
  acc, conf_matrix = Scoring(knn, x_test, y_test)
  print("KNN: \t", acc)
  save_model(knn, "models/Neighbour_model")

def save_model(model, filename):
  _filename = filename + '.sav'
  pickle.dump(model, open(_filename, 'wb'))

def use_model(filename):
  loaded_model = pickle.load(open(filename, 'rb'))
  return loaded_model

def Dataset_processing(path):

  #Here we create function that will take path to data set as an input, then we will process this data
  data= pd.read_csv(path)
  #Data shuffling is done before applying statistics to it in order to reduce any bias that may be introduced in the analysis.
  data = shuffle(data)
  # We repalce male to 0 and female to 1
  data["label"] = data["label"].replace("male", 0)
  data["label"] = data["label"].replace("female", 1)
  # get all columns except "label" (gender) we do it so our model will not know what is he analysing
  labels = data.keys()
  # we use drop function in order to remove label
  labels = labels.drop("label")
  # Here we can choose how many lables we can drop, by dropping lablels we increase speed but we reduce accuracy
  #labels = labels.drop("dfrange")
  #labels = labels.drop("mindom")
  #labels = labels.drop("centroid")
  #labels = labels.drop("mode")
  #labels = labels.drop("sfm")
  #labels = labels.drop("IQR")
  # labels = labels.drop("sd")
  # labels = labels.drop("median")
  # labels = labels.drop("meanfun")
  # labels = labels.drop("dfrange")
  # labels = labels.drop("maxdom")
  # labels = labels.drop("meandom")
  # labels = labels.drop("minfun")
  # labels = labels.drop("maxfun")
  # labels = labels.drop("modindx")

  # We apply data split in 2x to 1 x proportion (more on it in appeddix)
  X = data.loc[:, labels]
  y = data['label']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.34)# i applied 42 in order to ensure that my code can be reproduced
  # training set

  return X_train, y_train, X_test, y_test

def file_analysis(path_to_file):
  #description in appendix
  [file_rate, data] = audioBasicIO.read_audio_file(path_to_file)
  #extracted_features, extracted_features_names = audioFeature.feature_extraction.stFeatureExtraction(data, file_rate, 0.050*file_rate, 0.025*file_rate)
  mid_features, short_features, mid_feature_names = MidTermFeatures.mid_feature_extraction(data, file_rate,
                                                                                           int(0.050 * file_rate),
                                                                                           int(0.025 * file_rate),
                                                                                           int(0.050 * file_rate),
                                                                                           int(0.025 * file_rate))

  spec = np.abs(np.fft.rfft(data))
  freq = np.fft.rfftfreq(len(data), d=1 / file_rate)
  peakf = np.argmax(freq) # unusued
  amp = spec / spec.sum()
  mean = (freq * amp).sum()
  sd = np.sqrt(np.sum(amp * ((freq - mean) ** 2)))
  amp_cumsum = np.cumsum(amp)
  median = freq[len(amp_cumsum[amp_cumsum <= 0.5]) + 1]
  mode = freq[amp.argmax()]
  Q25 = freq[len(amp_cumsum[amp_cumsum <= 0.25]) + 1]
  Q75 = freq[len(amp_cumsum[amp_cumsum <= 0.75]) + 1]
  IQR = Q75 - Q25
  z = amp - amp.mean()
  w = amp.std()
  skew = ((z ** 3).sum() / (len(spec) - 1)) / w ** 3
  kurt = ((z ** 4).sum() / (len(spec) - 1)) / w ** 4
  spec_flatness = gmean(spec**2)/np.mean(spec**2)

  result_d = {
    'meanfreq': mean/1000,
    'sd': sd/1000,
    'median': median/1000,
    'Q25': Q25/1000,
    'Q75': Q75/1000,
    'IQR': IQR/1000,
    'skew': skew,
    'kurt': kurt,
    'sp.ent': mid_features[5].mean(),
    'sfm': spec_flatness,
    'mode': mode/1000,
    'centroid':  mid_features[3].mean()/1000,
  }
  return result_d

def test_new_sample(path_to_file):
  new_array_for_samples = []
  #all the custom file_analysis function with the path_to_file variable. This function presumably extracts features from the audio file
  analyis_of_file = file_analysis(path_to_file)
  #Create a new list containing the values of the extracted features from the file.
  new_array_for_samples = [analyis_of_file[x] for x in analyis_of_file]
  #Create a Normalizer object from the sklearn.preprocessing module with L2 normalization. L2 normalization scales the input features so that the Euclidean length (L2-norm) of each feature vector is 1.
  norm = Normalizer(norm='l2')
  new_array_for_samples = norm.transform(np.float64([new_array_for_samples]))
  # Create a PCA (Principal Component Analysis) object from the sklearn.decomposition module. PCA is a technique used to reduce the dimensionality of a dataset by transforming it into a new set of variables called principal components.
  pca=PCA()
  pca.fit(X_train)
  #Apply the PCA transformation to the new_array_for_samples feature vector. The transformed feature vector is stored back in new_array_for_samples. The [0] part is used to extract the first (and only) element of the transformed array since transform() returns a 2D array.
  new_array_for_samples=pca.transform(new_array_for_samples)[0]
  #Here we output our prediction
  print(svm.predict([new_array_for_samples]))
  # print(new_sample)
  return new_array_for_samples

parser = argparse.ArgumentParser(description='speech gender classification')

parser.add_argument("-w", "--wav",        action="store",       dest="wav",   help="Take a sample wav file and classify it", type=str)
parser.add_argument("-i", "--input",      action="store",       dest="inp",   help="Take a sample csv file and classify it", type=str)
parser.add_argument("-r", "--run",        action="store_true",                help="Run the classifier and see the accuracy results")
args = parser.parse_args()

if args.run: # --run or -r
  x_train, y_train, x_test, y_test = Dataset_processing("dataset/voice.csv")
  x_train, x_test      = normalize_L2(x_train, x_test)
  x_train, x_test, pca = PCA_decomposition(x_train, x_test)
  pickle.dump(pca,open("models/pca.sav",'wb'))
  One_to_rule_them_all(x_train, y_train, x_test, y_test)
if args.inp or args.wav:
  # fitting model
  try:
      svm = use_model("models/svm_model.sav")
      logreg = use_model("models/Logistic_regresion_model.sav")
      bernie = use_model("models/Bernouli_model.sav")
      knn = use_model("models/Neighbour_model.sav")
      pca = pickle.load(open("models/pca.sav", 'rb'))
  except:
      x_train, y_train, x_test, y_test = Dataset_processing("dataset/voice.csv")
      x_train, x_test      = normalize_L2(x_train, x_test)
      x_train, x_test, pca = PCA_decomposition(x_train, x_test)
      pickle.dump(pca, open("models_trained/pca.sav",'wb'))
      svm = SVM_fitting(x_train, y_train, _gamma="scale")
      logreg = Logistic_regresion(x_train, y_train)
      bernie = Bayes_beroulii(x_train, y_train)
      knn = Neighbour_fitting(x_train, y_train, _algorithm="ball_tree", _weights="distance")
if args.inp:
  file_path = args.inp

  file_lines = open(file_path, "r").read().split("\n")
  del file_lines[0] # remove first lines
  file_lines.remove('') # remove empty lines

  sample_csv = []
  for f in file_lines:
   # print(f)
    f = f.replace("female", '1')
    f = f.replace("male", '0')
    f = f.replace('"', '')
    #print(f)
    sample_csv.append(f.split(","))

  models = ["SVM", "LogReg", "Bernie", "KNN"]

  x_samples = []
  y_samples = []
  for i in range(len(sample_csv)):
    x_samples.append(sample_csv[i][0:-1])
    y_samples.append(sample_csv[i][-1])

  norm = Normalizer(norm='l2')
  x_samples.append([float(x) for x in sample_csv[i][0:-1]])  # Convert the elements to float
  x_samples = pca.transform(x_samples)

  svm_res   = svm.predict(x_samples),
  logreg_res    = logreg.predict(np.float64(x_samples)),
  bernie_res    = bernie.predict(np.float64(x_samples)),
  knn_res  = knn.predict(np.float64(x_samples)),

  success = [0, 0, 0, 0]
  tot = len(svm_res[0])
  print("SVM \t LR \t NB \t KNN \t label")
  for i in range(tot-1):
    #print(str(svm_res[0][i]) + " \t " + str(logreg_res[0][i]) + " \t " + str(bernie_res[0][i]) + " \t " + str(knn_res[0][i]) + " \t " + y_samples[i])

    success[0] += 1 if int(svm_res[0][i]) == int(y_samples[i]) else 0
    success[1] += 1 if int(logreg_res[0][i]) == int(y_samples[i]) else 0
    success[2] += 1 if int(bernie_res[0][i]) == int(y_samples[i]) else 0
    success[3] += 1 if int(knn_res[0][i]) == int(y_samples[i]) else 0

  print(str(success[0]) + "/" + str(tot) + " \t " +
        str(success[1]) + "/" + str(tot) + " \t " +
        str(success[2]) + "/" + str(tot) + " \t " +
        str(success[3]) + "/" + str(tot))

if args.wav:
  copyfile(args.wav, "voice.wav")

  FNULL = open(os.devnull, 'w')
  subprocess.call(('Rscript', "R/extract_single.r"), stdout=FNULL, stderr=subprocess.STDOUT)

  # read second line of csv file (so exclude header)
  sample = open("dataset/voice.csv", "r").read().split("\n")[1].split(",")
  print(sample[20])
  print(sample[20].find("female"))
  if( sample[20].find("female")==1):
    sample.pop()
    sample.pop()
    sample.append(1)
  else:
    sample.pop()
    sample.pop()
    sample.append(0)

  print(sample)
  sample = np.array(sample, dtype=np.float64).reshape(1, -1)  # Reshape the sample array for normalization
  sample = np.array(sample).reshape(1, -1)  # Reshape the sample array for normalization
  norm = preprocessing.Normalizer(norm='l2')
  sample = norm.transform(sample)
  sample = pca.transform(sample)

  print("male: 0, female: 1")
  print("SVM: \t", svm.predict(sample)[0])
  print("LR: \t", logreg.predict(np.float64(sample))[0])
  print("NB: \t", bernie.predict(np.float64(sample))[0])
  print("KNN: \t", knn.predict(np.float64(sample))[0])