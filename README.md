# amblyopia-ESTCNN
This repo is an implementation to classify amblyopia using ESTCNN model.
The dataset is obtained from the hospital and it is private. The task is a 4-class classification task, including N(normal),L(Left),R(right),B(binoculus)

We dived into twe series of experiments. One of them is the impact of the order of electrodes on the model, including five orders: order based on Shannon Entropy, order based on 10-20 system, order based on electrodes

<a href = "https://doi.org/10.1109/TNNLS.2018.2886414">ESTCNN</a> is a CNN model to extract features of EEG data.
We analyze the performance of 