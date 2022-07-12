# Short-Term Prediction of Residential Power Energy Consumption via CNN and Multi-Layer Bi-Directional LSTM Networks

Download the dataset from: https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption# (Thanks to [**demmojo et al.**](https://github.com/demmojo))

### Requirements:
Tensorflow: '2.3.1'

Keras:'2.1.5'


******************************************************
### Training:
******************************************************
Run main.py (Make loading_model=False)

For training settings, update can be made in cnn_m_bdlstm.py considering:

Line 51: epoches
Line 52: batchsize
Line 54: datasetpath 

******************************************************
### Testing:
******************************************************

Run main.py (Make loading_model=True) and add complete model path on line 8.

### Statistics (According to 2018):
<p align="center">
  <img src= "https://user-images.githubusercontent.com/43944394/178435770-63abdb44-bcf3-487a-84e5-8bdb104d113d.gif">
</p>
Energy consumption in South Korea on the annual basis illustrates that the large amount of energy is mostly consumed in industrial sector because in industrial sector different heavy machineries are installed for production in the form of electric, light, and water energy. The power energy consumption from resources to industries, business markets, transport and smart homes should be synchronized by strong cooperation. This flow is better shown in the figure.


### Framework:
<p align="center">
  <img src= "https://user-images.githubusercontent.com/43944394/178433703-154f9d4a-b20b-4d1c-8f62-9bbbcd7fb611.png" width="840" height="700">
</p>
Framework of the three steps of the proposed method. Step 1 is based on data acquisition, where the data are collected and preprocessed to refine and remove abnormalities. The refined data sequence is passed to step 2, where a CNN with an M-BDLSTM network is employed. Finally, step 3 provides the final ECP and an evaluation based on error metrics.

### Results:
<p align="center">
  <img src= "https://user-images.githubusercontent.com/43944394/178436918-31c52d51-da5b-46dd-882f-6190cb1f2ab4.gif">
</p>
Comparative analysis of the proposed method with state-of-the-art techniques on the basis of MSE and RMSE except BPTT where only RMSE is calculated.


### If you take help of this code or use it, Please cite the following papers.

#### IEEE Access; 2019
Fath U Min Ullah et al. [**"Short-Term Prediction of Residential Power Energy Consumption via CNN and Multi-Layer Bi-Directional LSTM Networks.**](https://ieeexplore.ieee.org/abstract/document/8945363) 
IEEE Access (2019).

#### MDPI Mathematics; 2021
Fath U Min Ullah et al. [**"Diving Deep into Short-Term Electricity Load Forecasting: Comparative Analysis and a Novel Framework.**](https://www.mdpi.com/2227-7390/9/6/611) 
MDPI Mathematics (2021).


Feel free to contact me at: fath3797@gmail.com

# Notice

The changes are made according to latest development in frameworks (tensorflow, keras, etc)
Next, We are really sorry to have the raw code, but to ensure the availability of an easy and understandable code we need some time. Feel free to contact me at fath3797@gmail.com if you have any queries or you will be okay with raw format code. Thanks
