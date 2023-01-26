# Near-optimal-EM-using-IL
Code for TCAD'22 paper. An imitation learning (IL)-based energy management algorithm. This algorithm provides the energy budget or allocation for each decision interval. At first, oracle policies are designed that optimize the energy allocation of the IoT device to enable self-powered operation while maximizing the utility to the application. Then, the Oracle policy is utilized to train an online policy that performs near-optimal energy allocation at runtime. 


Please cite our work using the following way if you find it motivating to your work. 
```
@article{yamin2022near,
  title={Near-Optimal Energy Management for Energy Harvesting IoT Devices Using Imitation Learning},
  author={Yamin, Nuzhat and Bhat, Ganapati},
  journal={IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems},
  volume={41},
  number={11},
  pages={4551--4562},
  year={2022},
  publisher={IEEE}
}
```
## How to Use
- **Required Dependency:** 
 
 **General Info and Tests:** The data preparation file generates the feature files for train, validation and testing. 
- **Other Function Files:** 
  - [Functions] (https://github.com/nuzhatyamin/Near-optimal-EM-using-IL/tree/main/functions) primarily contain associated functions of the data preparation file.
- **Data Files** 
  - The [Data](https://github.com/nuzhatyamin/Near-optimal-EM-using-IL/tree/main/Data) repository contains sample NREL Energy harvest data(2015-2017) and optimal energy allocation data. The data is used to generate features to train policys.
- **Data Preparation**
  - Run simile_data_preparation.py
    Each episode will be 24 hours and saved as .p files in the Data/Files folder
    Separate the files into Train_files, Valid_files and Test_files folders 
    Create xml files containing corresponding pickle files
    Command: python create_xml.py –file_dir ‘Data/Train_files/’ –out_dir ‘Data’ –filename train.xml
    
 **Reference**
    1. Hoang M. Le, Andrew Kang, Yisong Yue, Peter Carr: Smooth Imitation Learning for Online Sequence Prediction (ICML), 2016 [Link](https://arxiv.org/abs/1606.00968)
    2. Implementation of reference 1 [Link](https://sites.google.com/view/smooth-imitation-learning?pli=1) 
 

