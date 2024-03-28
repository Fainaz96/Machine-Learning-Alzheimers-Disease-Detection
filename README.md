# Machine-Learning-Alzheimers-Disease-Detection
Early detection of Alzheimer's disease using  Neuro imaging and Deep learning techniques 

ABSTRACT 
Accurately diagnosing the stages of Alzheimer's disease (AD) using MRI images poses a challenge for human doctors, as even experienced professionals can make mistakes. Traditional solutions heavily rely on various MRI pre-processing techniques to improve diagnostic accuracy. In recent years, deep learning methods, particularly Convolutional Neural Networks (CNNs), have shown promise in AD diagnosis through classification approaches. However, there is still a need to bridge the research gap in utilizing deep learning techniques for object detection in Alzheimer's disease diagnosis. In this research, conduct comprehensive comparisons and explore various techniques to address this gap, aiming to identify the most effective approach for solving this diagnostic challenge. 

Firstly, Introducing  Hyperparameter tuning in our CNN model to identify the best parameters for achieving optimal accuracy, considering execution time. Next, Compare the accuracy and loss metrics of various CNN architectures respective execution times. Additionally, Examine the impact of dataset augmentation techniques and the number of data samples on accuracy, loss, and execution time. Furthermore, Evaluate the effects of different hyperparameter tuning methods, along with their associated execution times. Our proposed CNN model incorporates the latest techniques, featuring the EfficientNetV2S architecture version2. As a result of research, provide a benchmark for the deep learning-based Alzheimer's disease stage detection, categorizing individuals into four stages Mild Demented, Moderate Demented, Non Demented, and Very Mild Demented. The utilization of the dataset we using MRI data from the ADNI, with its extensive 2D sliced MRI data for each stage, enables the extraction of meaningful features and patterns associated with Alzheimer's disease pathology. The outcomes of this study contribute to the advancement of diagnostic tools and therapeutic interventions, enhancing the understanding and management of Alzheimer's disease.

Keywords— Alzheimer's disease (AD), Convolutional Neural Networks (CNN), Deep learning, MRI

I.	INTRODUCTION 
Alzheimer's disease stands as a progressive neurodegenerative condition mainly impacting the elderly demographic. This ailment represents the predominant type of dementia, encompassing around 60-80% of instances of dementia. Its defining features encompass the buildup of anomalous protein formations like beta-amyloid plaques and tau tangles within the brain. Consequently, this process results in the gradual deterioration of cognitive capabilities.
Alzheimer's and dementia-related illnesses contributed to 5,98% of Sri Lanka's 6,939 fatalities in 2020. Due to these issues, Sri Lanka was placed 14th internationally with a death rate of 27.62 per 100,000 by WHO [1]. 
Alzheimer's research is aided by ADNI [2] and OASIS [3], two significant MRI databases. The ADNI provides thorough information on the various phases of disease, including MRI, genetics, biomarkers, and cognition. OASIS offers a variety of cognitive data. Google Colab [4] is our preferred platform for deep learning jobs. GPUs speed up the training of CNNs, and with Colab Pro, Have access to faster GPUs and more RAM for larger experiments and model improvement.
Certainly, CNNs have revolutionized computer vision. Notable designs include LeNet, GoogleNet's Inception modules, ResNet's residual connections, VGG's simplicity, and AlexNet's fame. EfficientNet balances depth, width, and resolution for great performance.
Hyperparameter tuning [5] is crucial for CNN effectiveness. It optimizes learning speed and generalization by adjusting parameters like learning rate, batch size, layers, dropout, and kernel size. Random Search, Genetic Algorithms, Grid Search, and Bayesian Optimization are alternative techniques. The Keras Tuner library aids Random Search. Platforms like scikit-learn, TensorFlow, Optuna, and Ray Tune offer tools for seamless hyperparameter optimization, automating and improving model performance.

II.	RELATED WORK

A.	Deep Learning for Alzheimer’s Disease detection
Various state of the art methods have been employed for the diagnosis of AD in the reviewed papers. These methods primarily leverage advanced technologies such as deep learning, machine learning, neuroimaging, and natural language processing. Some research papers have utilized deep learning algorithms, such as CNNs, to develop accurate classification models for Alzheimer's disease [6], [11], [14], [15], [20], [22], [29], [30], [31]. Others have focused on leveraging multiple data modalities, including MRI, PET, and clinical information, to improve the accuracy of disease diagnosis and prediction [13], [17], [19], [24], [27], [28], [38]. Furthermore, researchers have explored the use of gait analysis, speech data, and genetic factors in combination with deep learning and machine learning approaches to detect early stages of Alzheimer's disease [7], [16], [25]. Additionally, advancements in model architectures, such as the modified EfficientNet [35], and the development of models like DEMNET and EfficientNet  [33], [36], have contributed to accurate and timely diagnosis. Additionally, deep learning detection networks bypass MRI pre-processing steps [8], while others focus on brain segmentation and classification using 3D T1-weighted volumetric images [9]. Spatiotemporal feature extraction and classification [10], multimodal deep learning models [13], [18], [38].
Other approaches include the evaluation of neuroimages [29], the use of ensemble-based algorithms [28], and the prediction of AD progression [25], [27], [33]. these papers collectively demonstrate the effectiveness of deep learning and machine learning approaches in detecting, diagnosing, and predicting AD, and effective management of the disease. showcasing the potential for improved healthcare outcomes in the future.

B.	Accuracy according to techniquies 
Several research have been conducted to detect AD using different techniques and achieving varying levels of accuracy. In the [20], a CNN was used to detect AD from MRI images, achieving an accuracy of 92.5%. Similarly, [27] employed a deep learning model on MRI images and medical data to predict the progression of mild cognitive impairment to AD, achieving an accuracy of 93.1%. An ensemble-based machine learning algorithm was utilized in [28] to predict the conversion from mild cognitive impairment to AD, achieving  accuracy of 89.6%. In [25], Natural Language Processing (NLP) was employed to analyze speech data and identify features indicative of AD, resulting in an accuracy of 88.2%. Furthermore, [26] used machine learning techniques on various features, including patient demographics, clinical data, and imaging data, achieving an accuracy of 87.5%. Another study used a CNN on resting-state fMRIs and reached an accuracy of 94.19% in detecting AD [6]. Similarly, [12] used a CNN on MRIs and obtained an accuracy of 87.7%. [14] utilized a CNN on MRIs and achieved an accuracy of 85.7%. The use of a 3D-CNN in [15] yielded an accuracy of 83.3% in detecting AD from MRIs. Machine learning techniques were also applied in [16] to detect AD, resulting in an accuracy of 82.4%. A cascaded CNN was employed in [17] on MRI images, achieving an accuracy of 80.8%, [13] used a CNN on MRIs, obtaining an accuracy of 80.2%. [19] employed a multi-modal CNN on MRI and PET images, achieving an accuracy of 79.5% in detecting AD.
Finally According to [39], the current trend in Alzheimer's Disease diagnosis involves the use of classification methods coupled with various MRI pre-processing techniques. Notably, Sarraf [40] and [41] achieved a test accuracy of 98.80% and 99.90%, respectively, in the Alzheimer's disease (AD) or is cognitively normal (NC) binary classification category. In this paper, our primary objective is to attain a comparable more test accuracy to Sarraf's findings with and without the need for any MRI pre-processing techniques.

III.	METHODOLOGY
The proposed framework for early detection of Alzheimer's disease is built upon two key components: a Deep Learning detection network and the application of hyperparameter techniques. Let's elaborate on each of these components
 ![image](https://github.com/Fainaz96/Machine-Learning-Alzheimers-Disease-/assets/100863402/99d06fd4-6171-482a-a8c2-d06a20bc1854)
Fig. 1.	Flow chart of overall CNN model. 

A.	Pre-Processing
 Balancing all four classes datasets through preprocessing. This ensures equal data representation, enhancing accuracy and model robustness. Equal sample counts per class yield unbiased learning, improving AD detection reliability.

TABLE I. 	INPUT DATASET
Classes Of Alzheimer's Disease Stages	Source Dataset from ADNI	Pre-Processed Dataset	Total
Mild Demented	896	3104	4000
Moderate Demented	64	3936	4000
Non Demented	3200	800	4000
Very Mild Demented	2240	1760	4000

To tackle class imbalance, we use data augmentation by applying transformations like rotations, flips, and more to existing data. Random noise and brightness adjustments enhance diversity. This combats imbalance, prevents overfitting, and boosts model generalization (images resized to 128x128 pixels).
 ![image](https://github.com/Fainaz96/Machine-Learning-Alzheimers-Disease-/assets/100863402/0272ce27-54b2-41dc-9820-d92d9c11d4ca)

Fig. 2.	Original Vs Augmentation MRI images. 
In this visualization, MRI image visualization of data augmentation: original, rotated, flipped, brightness-adjusted, and zoomed versions. These many viewpoints improve training data for a strong model.
        
![image](https://github.com/Fainaz96/Machine-Learning-Alzheimers-Disease-/assets/100863402/54caee7e-7524-4bad-9ce6-a8600280d079)
![image](https://github.com/Fainaz96/Machine-Learning-Alzheimers-Disease-/assets/100863402/b3b33c19-269d-4833-a9f3-98f78ef9e298)
![image](https://github.com/Fainaz96/Machine-Learning-Alzheimers-Disease-/assets/100863402/09a85f3d-30b0-4acb-8706-158aa3b91afd)
![image](https://github.com/Fainaz96/Machine-Learning-Alzheimers-Disease-/assets/100863402/535b206d-f995-454c-a6d8-3f92a4b30d55)
      
Fig. 3.	Original Vs Enhanced MRI of Alzheimer's disease.	
A visual example of the proposed pre-processing step to enhance the contrast of MRI images, where figure (a) is Mild Demented (b) is Moderated Demented (c) is Non Demented (d) is Very Mild Demented MRI images with noise removed and enhanced.
B.	Convolutional Neural Networks (CNN)
In Convolutional Neural Network (CNN) model, Ithave leveraged the latest powerful feature extraction capabilities of EfficientNetV2S, which serves as the base model for our architecture. The model's input consists of image data, and we proceed with a series of blocks to extract features and classify the images into one of the four classes.
![image](https://github.com/Fainaz96/Machine-Learning-Alzheimers-Disease-/assets/100863402/cf667394-144b-40b5-a51c-fe987bbdadab)

Fig. 4.	Strcucture Of CNN Architecture. 
In Block 1, we begin with a 2D Convolutional layer, where we apply filters ranging from 16 to 128 with a step of 16. The kernel size is set to 3x3, and we use the Rectified Linear Unit (ReLU) activation function to introduce non-linearity. To maintain spatial dimensions, we employ 'same' padding. Block 2 follows with a 2D Max Pooling layer, using a pool size of 2x2. This step helps reduce the spatial dimensions while retaining the essential features. Block 3 incorporates a Dropout layer with a dropout rate ranging from 0.1 to 0.5 and a step of 0.1. The Dropout layer helps prevent overfitting by randomly dropping out a fraction of neurons during training. In Block 4, we apply another 2D Convolutional layer, similar to Block 1, with the same range of filters, kernel size, activation function, and padding. Block 5 repeats the 2D Max Pooling layer used in Block 2. Block 6 introduces a second Dropout layer, applying the same rate range as in Block 3. In Block 7, we employ a Flatten layer, converting the 2D feature maps into a 1D vector, suitable for further processing in dense layers. Blocks 8 and 9 consist of Dense layers. In Block 8, the number of units varies from 64 to 256 with a step of 32, and we apply the ReLU activation function. Block 9 follows a similar pattern, but the number of units ranges from 32 to 128 with a step of 32. Finally, in Block 10, we use a Dense layer with 4 units, representing the four classes in our classification problem. The activation function for this output layer is Softmax, which provides class probabilities for each input image.
The choice of this architecture, combining EfficientNetV2S as the base model with the custom-defined blocks (Build model) for feature extraction and classification, has been optimized to achieve good classification accuracy. By leveraging the powerful feature representation capabilities of EfficientNetV2S and fine-tuning the blocks to suit our specific problem, we aim to achieve accurate and reliable predictions on our image dataset.

C.	Hyperparameter Tuning
Because Random Search from Keras Tuner efficiently samples hyperparameter values over time, we use it. Without performing an extensive search, this approach investigates several combinations. 10 trials will be used to fine-tune important hyperparameters as filter size, layers, dropout, learning rate, and batch size. The design and training of our model will be influenced by the validation data set that performs the best (Trail 10).
TABLE II. 	10 TRIALS OF HYPERPARAMETER TUNING
Parameter	Trial#1	Trial#2	Trial#3	Trial#4	Trial#5	Trial#6	Trial#7	Trial#8	Trial#9	Trial#10
conv1_filters	48	64	32	64	16	32	32	64	48	32
conv1_kernel	3	5	3	3	5	3	5	5	5	3
conv2_filters	64	96	64	128	32	64	64	128	64	128
conv2_kernel	5	3	5	5	5	3	5	3	5	5
conv3_filters	256	64	192	64	192	256	64	128	128	256
conv1_kernel	5	5	5	5	3	3	3	5	5	3
dropout_1	0.3	0.4	0.2	0.2	0.4	0.3	0.4	0.4	0.2	0.3
dropout_2	0.4	0.4	0.3	0.2	0.3	0.4	0.2	0.4	0.4	0.2
dense_units_1	64	64	256	64	128	256	192	256	192	64
dense_units_2	32	96	96	96	128	128	64	128	64	128
Learning rate	0.01	0.01	0.01	0.001	0.0001	0.0001	0.01	0.001	0.0001	0.0001
validation accuracy	50.08	50.08	50.08	50.08	99.84	99.69	99.84	94.68	99.84	100

IV.	RESULTS AND DISCUSSION

A.	Hyperparameter Tuning
Before hyperparameter tuning, the model already exhibited strong performance on all three datasets: test, validation, and training. The initial results showed a test accuracy 99.22%, validation accuracy was 98.59, training accuracy 99.98% of the training data, and it fit the training data closely. After tuning, the model's performance further improved, showcasing the effectiveness of the chosen hyperparameters. The test accuracy increased to 99.84%, validation accuracy 100%, training accuracy 100%, meaning it accurately predicted all the training data, achieving optimal fit to the training set,Here is the table containing the accuracy and losses for the test, training, and validation datasets.

TABLE III. 	ACCURACY AND LOSS FOR CNN MODEL 
	Before Tuned	After Tuned
Test Accuracy	99.22	99.84
Validation Accuracy	98.59	100
Train Accuracy	99.98	100
Test Loss	2.46	0.93
Validation Loss	3.48	0.00
Train Loss	0.07	0.00
The significant improvement in testing, validation, and training metrics after hyperparameter tuning demonstrates the importance of selecting optimal hyperparameters for the model. By fine-tuning the hyperparameters, the model's performance was boosted, leading to better generalization, higher accuracy, and lower losses on unseen and validation data.
![image](https://github.com/Fainaz96/Machine-Learning-Alzheimers-Disease-/assets/100863402/6a1206be-f3df-4b6b-8782-d8eb7bb6a290)

Fig. 5.	Loss and Accuracy plot. (Before Tuned CNN model )
![image](https://github.com/Fainaz96/Machine-Learning-Alzheimers-Disease-/assets/100863402/eaa8d9bd-6f69-47d8-ba61-99f53e48a227)

Fig. 6.	Loss and Accuracy plot. (After Tuned CNN model )
  According to two graphs Stable points of achieving the best accuracy and minimal loss are effectively attained after hyperparameter tuning in the CNN model.

B.	Datasets
TABLE IV. 	ACCURACY AND LOSS FOR CNN MODEL 
Number of MRIs	Without Pre-Processing	With Pre-Processing
Input	Mild Demented	224	448	672	896	4000
	Moderated Demented	16	32	48	64	4000
	Non Demented	800	1600	2400	3200	4000
	Very Mild Demented	560	1120	1680	2240	4000
	Total	1 600 (25%)	3 200 (50%)	4 800 (75%)	6 400 (100%)	16 000
Accuracy	Train accuracy	100	100	99.55	99.89	99.89
	Validation accuracy	88.33	91.21	95.41	98.59	99.84
	Test accuracy	88.76	90.12	95.02	99.22	99.32
![image](https://github.com/Fainaz96/Machine-Learning-Alzheimers-Disease-/assets/100863402/295c145a-6a98-4fe6-911f-e1463d6ceeea)

Fig. 7.	Accuracy Change With Dataset With and Without Pre-Processing
While we Increasing the size of the input dataset to our CNN model improve model accuracy by providing more examples for learning, enhancing generalization to unseen data, reducing biases, and enabling better feature representations. However, the benefits may saturate beyond a certain point, and extremely large datasets may require substantial computational resources. In summary, larger datasets generally lead to better model performance and robustness.
But we Increasing data using pre-processing techniques can initially lead to significant accuracy improvements by providing more diverse examples and improving generalization. However, as the dataset grows larger, the rate of accuracy improvement may decrease since the model becomes more familiar with the existing data and the impact of augmented samples diminishes relative to the original data. Nonetheless, data augmentation remains essential for preventing overfitting and enhancing the model's robustness.

C.	CNN Architecture 
EfficientNetV2S (Base model) is a highly efficient and powerful convolutional neural network architecture, known for its state-of-the-art performance while being computationally efficient. However, in certain scenarios, it might be beneficial to enhance the model further by introducing additional layers (Build model). By adding extra convolutional layers, such as conv1_filters, conv1_kernel, conv2_filters, conv2_kernel, and conv3_filters, we can increase the model's capacity to capture intricate features from the dataset. Furthermore, applying dropout layers after each set of convolutional layers (dropout_1 and dropout_2) helps in regularizing the model, preventing overfitting, and promoting better generalization. Additionally, by adjusting the number of units in the dense layers (dense_units_1 and dense_units_2), we can fine-tune the model's ability to learn complex representations and improve its overall performance.

TABLE V. 	ACCURACY AND LOSS FOR CNN MODEL 
Accuracy/loss	Without additional layers  (Base model)	With additional layers (Build model)
Test Accuracy	85.53	99.84
Validation Accuracy	88.22	100
Train Accuracy	98.98	100
Test Loss	9.46	0.93
Validation Loss	8.48	0.00
Train Loss	0.17	0.00
TABLE VI. 	ACCURACY, LOSS AND EXECUTION TIME CHANGE WITH 
	           CNN ARCHITECTURE 
CNN Architecture	Accuracy	Loss	Execution Time Per Step (ms)
rasnet 101	99.69	0.76	8ms
rasnet 101	99.07	2.75	8ms
LeNet-5	98.75	4.59	8ms
VGG16	99.07	2.37	8ms
VGG19	99.53	1.92	8ms
InceptionV3	99.53	1.72	8ms
InceptionResNetV2	99.38	3.38	8ms
AlexNet	99.22	3.28	8ms
EfficientNetB0	99.69	1.28	12ms
EfficientNetB1	98.75	4.98	12ms
EfficientNetB2	99.53	0.97	1s 12ms
EfficientNetB3	98.91	2.27	1s 14ms
EfficientNetB4	99.38	1.46	13ms
EfficientNetB5	98.91	3.14	13ms
EfficientNetB6	99.69	0.87	12ms
EfficientNetB7	99.53	0.122	8ms
EfficientNetV2B0	99.22	1.37	1s 12ms
EfficientNetV2B1	99.69	1.37	1s 12ms
EfficientNetV2B2	99.38	1.1	1s 14ms
EfficientNetV2B3	99.22	1.81	1s 12ms
EfficientNetV2S	99.84	0.55	12ms
EfficientNetV2M	99.22	2.32	1s 13ms
EfficientNetV2L	99.07	2.09	1s 12ms
![image](https://github.com/Fainaz96/Machine-Learning-Alzheimers-Disease-/assets/100863402/e3efb665-9588-47c7-bdaa-ca8f98f33f56)

Fig. 8.	Accuracy Change With Types of CNN Architecture.
Among popular CNN architectures, there are variations in execution time, accuracy, and loss. However, EfficientNet stands out as the best choice overall for several reasons. Firstly, it incorporates the latest advancements in deep learning, making it highly advanced and effective. Secondly, EfficientNet offers a wide range of model sizes (B0-B7 (V1), B0-B3(V2),V2S,V2M,V2L) that cater to different computational resources, ensuring optimal efficiency. Thirdly, its compound scaling approach efficiently balances accuracy and computational performance.

D.	Epoch
![image](https://github.com/Fainaz96/Machine-Learning-Alzheimers-Disease-/assets/100863402/ea65ac05-6945-4d1a-bc8a-bad3348d3f61)

Fig. 9.	Accuracy Change With Epoch.  
During model training, increasing the number of epochs initially improves accuracy, with a peak of about 100 epochs. This is consistent with learning dynamics: initial growth, optimal generalization at the halfway point, followed by a possible fall due to overfitting.

E.	Batch size
![image](https://github.com/Fainaz96/Machine-Learning-Alzheimers-Disease-/assets/100863402/923aa397-1628-4cf5-972e-a8abcafda666)

Fig. 10.	Accuracy Change With Epoch.
In our analysis of various batch sizes (8, 16, 32, 64, 128, 256, 512, 1024, and 2048) for training our deep learning model, Batch size 64 yielded the highest accuracy in AD detection. This size achieved the optimal balance between stability and adaptability, resulting in the best performance. Additionally, observed that smaller batch sizes led to faster execution times, but they did not perform as well in terms of accuracy. On the other hand, larger batch sizes had longer execution times but offered more stable training updates. Considering both accuracy and execution time, a batch size of 64 emerged as the most suitable choice for our model.

V.	CONCLUSION
AD requires early, accurate diagnosis. Modern methods like deep learning, machine learning, and neuroimaging offer promise. Our research emphasizes deep learning object detection for AD diagnosis, surpassing traditional methods. We extensively compared techniques, including hyperparameter tuning, to find the best approach for this complex diagnostic challenge.
This study introduced hyperparameter tuning to optimize CNN model accuracy and execution time. We evaluated CNN architectures for accuracy, loss, and time. We also studied dataset size, augmentation and batch sizes' effects on accuracy and time. Leveraging EfficientNetV2S architecture, we enhanced Alzheimer's disease stage detection significantly. Our CNN model categorizes four stages using the ADNI dataset via Kaggle, extracting crucial patterns. This research advances diagnostics and interventions, improving Alzheimer's understanding and management.
In conclusion, our research highlights deep learning's potential in Alzheimer's diagnosis. Through hyperparameter tuning and architecture selection, we achieved high accuracy in stage detection. This advancement can aid better diagnostics and treatment strategies, improving patient outcomes. Further research will enhance Alzheimer's management, benefiting affected individuals' quality of life.
VI.	REFERENCES
[1]	 “Alzheimers & Dementia in Sri Lanka,” World Life Expectancy. https://www.worldlifeexpectancy.com/sri-lanka-alzheimers-dementia#:~:text=Sri%20Lanka%3A%20Alzheimers%20%26%20Dementia&text=According%20to%20the%20latest%20WHO.
[2]	R. C. Petersen et al., “Alzheimer’s Disease Neuroimaging Initiative (ADNI),” Neurology, vol. 74, no. 3, pp. 201–209, Jan. 2010, doi: https://doi.org/10.1212/WNL.0b013e3181cb3e25.
[3]	D. S. Marcus, T. H. Wang, J. Parker, J. G. Csernansky, J. C. Morris, and R. L. Buckner, “Open Access Series of Imaging Studies (OASIS): Cross-sectional MRI Data in Young, Middle Aged, Nondemented, and Demented Older Adults,” Journal of Cognitive Neuroscience, vol.19, no.9,pp.14981507,Sep.2007,doi:https://doi.org/10.1162/jocn.2007.19.9.1498.
[4]	T. Carneiro, R. V. Medeiros Da Nobrega, T. Nepomuceno, G.-B. Bian, V. H. C. De Albuquerque, and P. P. R. Filho, “Performance Analysis of Google Colaboratory as a Tool for Accelerating Deep Learning Applications,” IEEE Access, vol. 6, pp. 61677–61685, 2018, doi: https://doi.org/10.1109/access.2018.2874767.
[5]	F. Hutter, L. Kotthoff, and J. Vanschoren, Eds., Automated Machine Learning. Cham: Springer International Publishing, 2019. doi: https://doi.org/10.1007/978-3-030-05318-5.
[6]	N. T. Duc, S. Ryu, M. N. I. Qureshi, M. Choi, K. H. Lee, and B. Lee, “3D-Deep Learning Based Automatic Diagnosis of Alzheimer’s Disease with Joint MMSE Prediction Using Resting-State fMRI,” Neuroinformatics, vol. 18, no. 1, pp. 71–86, May 2019, doi: https://doi.org/10.1007/s12021-019-09419-w.
[7]	B. Ghoraani, L. N. Boettcher, M. D. Hssayeni, A. Rosenfeld, M. I. Tolea, and J. E. Galvin, “Detection of mild cognitive impairment and Alzheimer’s disease using dual-task gait assessments and machine learning,” Biomedical Signal Processing and Control, vol. 64, p. 102249, Feb. 2021, doi: https://doi.org/10.1016/j.bspc.2020.102249.
[8]	Jia Xian Fong, Mohd Ibrahim Shapiai, Yee Wen Tiew, Uzma Batool, and Hilman Fauzi, “Bypassing MRI Pre-processing in Alzheimer’s Disease Diagnosis using Deep Learning Detection Network,” Feb. 2020, doi: https://doi.org/10.1109/cspa48992.2020.9068680.
[9]	Chong Hyun Suh et al., “Development and Validation of a Deep Learning–Based Automatic Brain Segmentation and Classification Algorithm for Alzheimer Disease Using 3D T1-Weighted Volumetric Images,” American Journal of Neuroradiology, vol. 41, no. 12, pp. 2227–2234, Nov. 2020, doi: https://doi.org/10.3174/ajnr.a6848.
[10]	H. Parmar, B. Nutter, R. Long, S. Antani, and S. Mitra, “Spatiotemporal feature extraction and classification of Alzheimer’s disease using deep learning 3D-CNN for fMRI data,” Journal of Medical Imaging, vol. 7, no. 05, Oct. 2020, doi: https://doi.org/10.1117/1.jmi.7.5.056001.
[11]	M. Shahbaz, S. Ali, A. Guergachi, A. Niazi, and A. Umer, “Classification of Alzheimer’s Disease using Machine Learning Techniques,” Proceedings of the 8th International Conference on Data Science, Technology and Applications, 2019, doi: https://doi.org/10.5220/0007949902960303.
[12]	A. W. Salehi, P. Baglat, B. B. Sharma, G. Gupta, and A. Upadhya, “A CNN Model: Earlier Diagnosis and Classification of Alzheimer DiseaseusingMRI,”IEEEXplore,Sep.01,2020.https://ieeexplore.ieee.org/abstract/document/9215402 (accessed Oct. 15, 2021).
[13]	J. Venugopalan, L. Tong, H. R. Hassanzadeh, and M. D. Wang, “Multimodal deep learning models for early detection of Alzheimer’s disease stage,” Scientific Reports, vol. 11, no. 1, p. 3254, Feb. 2021, doi: https://doi.org/10.1038/s41598-020-74399-w.
[14]	L. F. Samhan, A. H. Alfarra, and S. S. Abu-Naser, “Classification of Alzheimer’s Disease Using Convolutional Neural Networks,” International Journal of Academic Information Systems Research (IJAISR),vol.6,no.3,pp.18–23,2022,Available: https://philpapers.org/rec/SAMCOA-4.
[15]	G. Folego, M. Weiler, R. F. Casseb, R. Pires, and A. Rocha, “Alzheimer’s Disease Detection Through Whole-Brain 3D-CNN MRI,” Frontiers in Bioengineering and Biotechnology, vol. 8, Oct. 2020, doi: https://doi.org/10.3389/fbioe.2020.534592.
[16]	M. M. Abd El Hamid, M. S. Mabrouk, and Y. M. K. Omar, “DEVELOPING AN EARLY PREDICTIVE SYSTEM FOR IDENTIFYING GENETIC BIOMARKERS ASSOCIATED TO ALZHEIMER’S DISEASE USING MACHINE LEARNING TECHNIQUES,” Biomedical Engineering: Applications, Basis and Communications, vol. 31, no. 05, p. 1950040, Sep. 2019, doi: https://doi.org/10.4015/s1016237219500406.
[17]	M. Liu, D. Cheng, K. Wang, and Y. Wang, “Multi-Modality Cascaded Convolutional Neural Networks for Alzheimer’s Disease Diagnosis,” Neuroinformatics, vol. 16, no. 3–4, pp. 295–308, Mar. 2018, doi: https://doi.org/10.1007/s12021-018-9370-4.
[18]	J. Wen et al., “Convolutional neural networks for classification of Alzheimer’s disease: Overview and reproducible evaluation,” Medical Image Analysis, vol. 63, p. 101694, Jul. 2020, doi: https://doi.org/10.1016/j.media.2020.101694.
[19]	Y. Huang, J. Xu, Y. Zhou, T. Tong, and X. Zhuang, “Diagnosis of Alzheimer’s Disease via Multi-Modality 3D Convolutional Neural Network,” Frontiers in Neuroscience, vol. 13, May 2019, doi: https://doi.org/10.3389/fnins.2019.00509.
[20]	M. Zaabi, N. Smaoui, H. Derbel, and W. Hariri, “Alzheimer’s disease detection using convolutional neural networks and transfer learning based methods,” 2020 17th International Multi-Conference on Systems, Signals & Devices (SSD), Jul. 2020, doi: https://doi.org/10.1109/ssd49366.2020.9364155.
[21]	L. Yue et al., “Auto-Detection of Alzheimer’s Disease Using Deep ConvolutionalNeuralNetworks,”IEEEXplore,Jul.01,2018.https://ieeexplore.ieee.org/abstract/document/8687207/ (accessed Jul. 08, 2023).
[22]	S. Basaia et al., “Automated classification of Alzheimer’s disease and mild cognitive impairment using a single MRI and deep neural networks,” NeuroImage: Clinical, vol. 21, p. 101645, Jan. 2019, doi: https://doi.org/10.1016/j.nicl.2018.101645.
[23]	H. A. Helaly, M. Badawy, and A. Y. Haikal, “Deep Learning Approach for Early Detection of Alzheimer’s Disease,” Cognitive Computation, Nov. 2021, doi: https://doi.org/10.1007/s12559-021-09946-2.
[24]	T. Jo, K. Nho, and A. J. Saykin, “Deep Learning in Alzheimer’s Disease: Diagnostic Classification and Prognostic Prediction Using Neuroimaging Data,” Frontiers in Aging Neuroscience, vol. 11, Aug. 2019, doi: https://doi.org/10.3389/fnagi.2019.00220.
[25]	N. Clarke, T. R. Barrick, and P. Garrard, “A Comparison of Connected Speech Tasks for Detecting Early Alzheimer’s Disease and Mild Cognitive Impairment Using Natural Language Processing and Machine Learning,” Frontiers in Computer Science, vol. 3, May 2021, doi: https://doi.org/10.3389/fcomp.2021.634360.
[26]	C. Kavitha, V. Mani, S. R. Srividhya, O. I. Khalaf, and C. A. Tavera Romero, “Early-Stage Alzheimer’s Disease Prediction Using Machine Learning Models,” Frontiers in Public Health, vol. 10, Mar. 2022, doi: https://doi.org/10.3389/fpubh.2022.853294.
[27]	B. Y. Lim et al., “Deep Learning Model for Prediction of Progressive Mild Cognitive Impairment to Alzheimer’s Disease Using Structural MRI,” Frontiers in Aging Neuroscience, vol. 14, Jun. 2022, doi: https://doi.org/10.3389/fnagi.2022.876202.
[28]	M. Grassi et al., “A Novel Ensemble-Based Machine Learning Algorithm to Predict the Conversion From Mild Cognitive Impairment to Alzheimer’s Disease Using Socio-Demographic Characteristics, Clinical Information, and Neuropsychological Measures,” Frontiers in Neurology, vol. 10, Jul. 2019, doi: https://doi.org/10.3389/fneur.2019.00756.
[29]	A. A, P. M, M. Hamdi, S. Bourouis, K. Rastislav, and F. Mohmed, “Evaluation of Neuro Images for the Diagnosis of Alzheimer’s Disease Using Deep Learning Neural Network,” Frontiers in Public Health, vol. 10, Feb. 2022, doi: https://doi.org/10.3389/fpubh.2022.834032.
[30]	P. C. Muhammed Raees and V. Thomas, “Automated detection of Alzheimer’s Disease using Deep Learning in MRI,” Journal of Physics: Conference Series, vol. 1921, p. 012024, May 2021, doi: https://doi.org/10.1088/1742-6596/1921/1/012024.
[31]	D. Manzak, G. Çetinel, and A. Manzak, “Automated Classification of Alzheimer’s Disease using Deep Neural Network (DNN) by Random ForestFeatureElimination,”IEEEXplore,Aug.01,2019.https://ieeexplore.ieee.org/document/8845325 (accessed Mar. 24, 2023).
[32]	H. Ji, Z. Liu, W. Q. Yan, and R. Klette, “Early Diagnosis of Alzheimer’s Disease Using Deep Learning,” Proceedings of the 2nd International Conference on Control and Computer Vision - ICCCV 2019, 2019, doi: https://doi.org/10.1145/3341016.3341024.
[33]	S. Murugan et al., “DEMNET: A Deep Learning Model for Early Diagnosis of Alzheimer Diseases and Dementia From MR Images,” IEEE Access, vol. 9, pp. 90319–90329, 2021, doi: https://doi.org/10.1109/ACCESS.2021.3090474.
[34]	E. Altinkaya, K. Polat, and B. Barakli, “Detection of Alzheimer’s Disease and Dementia States Based on Deep Learning from MRI Images: A Comprehensive Review,” Journal of the Institute of Electronics and Computer, vol. 1, pp. 39–53, 2019, doi: https://doi.org/10.33969/JIEC.2019.11005.
[35]	B. Zheng, A. Gao, X. Huang, Y. Li, D. Liang, and X. Long, “A modified 3D EfficientNet for the classification of Alzheimer’s disease using structural magnetic resonance images,” vol. 17, no. 1, pp. 77–87, Aug. 2022, doi: https://doi.org/10.1049/ipr2.12618.
[36]	M. Tan and Q. Le, “EfficientNetV2: Smaller Models and Faster Training,”proceedings.mlr.press,Jul.01,2021.http://proceedings.mlr.press/v139/tan21a.html.
[37]	M. B. T. Noor, N. Z. Zenia, M. S. Kaiser, S. A. Mamun, and M. Mahmud,“Application of deep learning in detecting neurological disorders from magnetic resonance images: a survey on the detection of Alzheimer’s disease, Parkinson’s disease and schizophrenia,” Brain Informatics,vol.7,no.1,Oct. 2020, doi: https://doi.org/10.1186/s40708-020-00112-2.
[38]	J. Venugopalan, L. Tong, H. R. Hassanzadeh, and M. D. Wang, “Multimodal deep learning models for early detection of Alzheimer’s disease stage,” Scientific Reports, vol. 11, no. 1, p. 3254, Feb. 2021, doi: https://doi.org/10.1038/s41598-020-74399-w.
[39]	A. Khvostikov, K. Aderghal, J. Benois-Pineau, A. Krylov, and G. Catheline, “3D CNN-based classification using sMRI and MD-DTI images for Alzheimer disease studies,” arXiv:1801.05968 [cs], Jan. 2018, Available: https://arxiv.org/abs/1801.05968.
[40]	S. Sarraf and G. Tofighi, “Classification of Alzheimer’s Disease using fMRI Data and Deep Learning Convolutional Neural Networks,” arXiv:1603.08631[cs],Mar.2016,Available:https://arxiv.org/abs/1603.08631.
[41]	S. Sarraf, D. D. DeSouza, J. Anderson, and G. Tofighi, “DeepAD: Alzheimer’s Disease Classification via Deep Convolutional Neural Networks using MRI and fMRI,” Aug. 2016, doi: https://doi.org/10.1101/07044. 


