import numpy as np
import matplotlib.pyplot as plt
import sklearn.svm as svm
from sklearn.metrics import confusion_matrix

import torch
import pt_deep
import data


class KSVMWrap():


  def __init__(self,X,Y,param_svm_c = 1, param_svm_gamma = 'auto'):
    self.param_svm_c = param_svm_c
    self.param_svm_gamma = param_svm_gamma
    self.classifier = svm.SVC(C = self.param_svm_c, gamma = self.param_svm_gamma)
    self.classifier.fit(X,Y)

  
  def predict(self,X):
    return self.classifier.predict(X)

  
  def get_scores(self,X):
    return self.classifier.decision_function(X)

  def support(self):
    return self.classifier.support_


def calc_metric(Y,Y_pred):
    """Calcualates performace metrics
        Arguments:
        Y: True labels
        Y_pred: Predicted labels

        Returns: Tuple
         A: class Accuracy vector
         P: class precision vector
         R: class recall vector
         avgP: Average precision
    """
    CM = confusion_matrix(Y,Y_pred)
    cm_diag = np.diag(CM)
    n_classes = CM.shape[0]
    P = np.zeros(n_classes)
    R = np.zeros(n_classes)
    A = np.zeros(n_classes)

    for i in range(n_classes):
        TP = cm_diag[i]
        FP = np.sum(CM[i,:]) - cm_diag[i]
        FN = np.sum(CM[:,i])- cm_diag[i]
        P[i] = TP/(FP + TP)
        R[i] = TP/(FN + TP)
        A[i] = TP/np.sum(CM[:,i])
    
    avgP = np.sum(P)/n_classes
    
    print(f"Class Accuracy: {A}")
    print(f"Class Precision: {P}")
    print(f"Class Recall: {R}")
    print(f"Average Precision: {avgP}")

    return A,P,R,avgP



if __name__ == '__main__':
  np.random.seed(100)

  # get data
  X,Y_ = data.sample_gmm_2d(6, 2, 10)
  # X,Y_ = sample_gauss_2d(2, 100)

  SVM_model = KSVMWrap(X,Y_)
  svm_scores = SVM_model.get_scores(X)
  Y = SVM_model.predict(X)
  calc_metric(Y_,Y)


  deep_model = pt_deep.PTDeep([2,10,2],activation=torch.relu)
  # graph the decision surface
  rect=(np.min(X, axis=0), np.max(X, axis=0))
  data.graph_surface(SVM_model.get_scores, rect, offset=0)
  # graph the data points
  data.graph_data(X, Y_, Y, special=SVM_model.support())
  plt.show()

  sample1 = (X,Y_)
  sample2 = data.sample_gmm_2d(8,2,5)
  sample3 = data.sample_gmm_2d(3,2,30)
  samples = [sample1,sample2,sample3]
  sample_description = ["K = 6 C = 2 N = 10",
  "K = 8 C = 2 N = 5",
  "K = 3 C = 2 N = 30"]

  for i, sample in enumerate(samples):
    X,Y_ = sample
    X_tensor = torch.tensor(X,dtype = torch.float, requires_grad=False)
    Yoh_ = data.class_to_onehot(Y_)
    Yoh_ = torch.tensor(Yoh_)
    svm_model = KSVMWrap(X,Y_)
    pt_deep.train(deep_model,X_tensor,Yoh_,int(1e4),param_delta = 0.1,param_lambda = 1e-4,verbose = False)
    deep_Y = deep_model.classify(X_tensor)
    svm_Y = svm_model.predict(X)
    print("\n \n")

    print("Sample information: "+sample_description[i])
    print("Deep model performance:")
    calc_metric(Y_,deep_Y)
    print("\n")
    print("SVM model performance:")
    calc_metric(Y_,svm_Y)
