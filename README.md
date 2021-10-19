# Covid-19 Detection Based on Chest X-Ray and CT images
## X-Ray dataset divided
We get 2 X-Ray datasets should be in **/original_data/**, and size of the original data is more than 3.4GB,  
so we just completed and ran the process of **read_xray_once.py** and **build_xray_dataset.py** in advance.  
And the dataset divided is necessary in the **/dataset/xray/**.  So the links of them are below.  
[x_train](https://drive.google.com/file/d/1jXPXpEAWE57HshOx_m9A0bFAWRmyQR1d/view?usp=sharing), 
[y_train](https://drive.google.com/file/d/1QO2KPs0OTrn1Qzp5C7_s1PSWY-wc06q0/view?usp=sharing), 
[x_test](https://drive.google.com/file/d/1Zvln2lbhk6Aov3bg8-dk6aYKOtiFPkIs/view?usp=sharing), 
[y_test](https://drive.google.com/file/d/1O942yv17102st1tdtgjOTErbj_pnHIhh/view?usp=sharing)


## Requirements
Pytorch == 1.6.0  
torchvision == 0.7.0  


## Guide
1. Download the dataset files.
2. Train the Pretrained_res50 and Pretrained_vgg separately,  
   then train the Ensemble. (Models architecture in build_xray_models.py, train process in train_xray.py)
3. Test and classify a X-Ray using trained Ensemble. (An example in xray_application.py)


