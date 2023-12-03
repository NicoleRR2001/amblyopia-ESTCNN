# amblyopia-ESTCNN
This repo is an implementation to classify amblyopia using ESTCNN model.
The dataset is obtained from the hospital and it is private. The task is a 4-class classification task, including N(normal),L(Left),R(right),B(binoculus). The sample rate is fixed and it is 500hz. 

We dived into twe series of experiments. 
One of them is the impact of window size(the length of the signal) on the model. We seperate data into sis lengths, 500-3000.

The other is the impact of the order of electrodes on the model, including five orders: order based on Shannon Entropy, order based on 10-20 system, order based on electrodes of left and right brains and order based on the power of electrodes. 

<a href = "https://doi.org/10.1109/TNNLS.2018.2886414">ESTCNN</a> is a CNN model to extract features of EEG data.
