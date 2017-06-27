# Project Work Group 4 - DLCV
This is the project repository for the group 4 at the DLCV. The Team is made up by:

| <img src="https://github.com/AdriaGS/AdriaGS.github.io/blob/master/Images/IMG_5420.jpg" width="250"> | <img src="https://github.com/AdriaGS/AdriaGS.github.io/blob/master/Images/IMG_8642.jpg" width="250"> | <img src="https://github.com/AdriaGS/AdriaGS.github.io/blob/master/Images/Luis.jpg" width="250"> |
| :---: | :---: | :---: |
| Adrià Gil Sorribes | Genís Floriach Pigem | Luis Esteve Elfau |

It is going to be explained below what has been done during the [Deep Learning for Computer Vison](https://telecombcn-dl.github.io/2017-dlcv/) course at UPC at Summer 2017.

<img src="https://github.com/AdriaGS/AdriaGS.github.io/blob/master/Images/UPC_ETSETB.jpg" width="600">

# Task 1 - CNN Cifar10

The main objective of this second task is to explore the whole recipe of the CIFAR dataset. The main idea is to design a Convolutional Neural Network (CNN) and train it with the CIFAR dataset.

- Task 1.1 Architecture of the CNN
    
    In this subtask the main goal is to design the architecture of the CNN: Neurons per layer. Amount of layers. Types of layers (Fully-connected, Convolutional) as well as nonlinearities (ReLu, Sigmoid,...)
    
    We have developed 2 architectures:
   
   <img src="https://github.com/AdriaGS/AdriaGS.github.io/blob/master/Images/Task_1_1v1.png" width="400" > 

    And
    
   <img src="https://github.com/AdriaGS/AdriaGS.github.io/blob/master/Images/Task_1_1v2.png" width="450" >
   
- Task 1.2 Training the CNN

    In this second subtask the idea is train the CNN, exploring some improvements: Data augmentation, batch size, amount of epochs, regularization (drop outs), learning rate or optimizers. 
  
    We have experimented with the number of epochs as well as regularization (adding drop outs) in oder to improve the performance of the CNN.


- Task 1.3 Visualization
    
    In this third task we have to provide some kind of visualization of the processes of the CNN: Visualize filter responses (from our network or from other pre-trained network), t-SNE or Off-the-shelf AlexNet (Visualize local classification over a small set of images)

# Task 2 - Training a softmax classifier

The main objective of this second task is to do some transfer learning. The main idea is to adapt some pre-trained neural network models to classify images on a new dataset which is composed by images of buildings of Terrassa. This task is divided in two subtasks:

- CNN trained with CIFAR10 + Softmax layer on top of CNN
    
   In this subtask the main goal is to use the previously implemented and trained CNN with the CIFAR10 dataset in order to classify the    images of the new database (TerrassaBuildings). For this reason we will load the model, we will extract the top layers and we will      train a new top_model (fully connected layers) softmax classifier with the new database.
   
   <img src="https://github.com/AdriaGS/AdriaGS.github.io/blob/master/Images/Task21.PNG" >

- Off the Shelf VGG16 + Softmax layer on top of the CNN

  In this second subtask the idea is the same as in the previous but this time loading an Imagenet pre-trained model such as the VGG16.   This model has been trained with the imagenet database which is layer and far wider than the CIFAR10. With this configuration we         expect better results.
  
  <img src="https://github.com/AdriaGS/AdriaGS.github.io/blob/master/Images/Task22.PNG" >


# Task 3 - Fine-tuning

In this task we fine tune the last convolutional layers of the VGG16 network. To do so, we start from the network obtained at the previous task after training the last convolutional layers. It is important to start the fine-tuning with all the layers previously trained. One should not add a randomly initialized fully-connected network on top of a pre-trained convolutional base. This is because the large gradient updates triggered by the randomly initialized weights would wreck the learned weights in the convolutional base.

We only fine-tune the last convolutional block  in order to prevent overfitting, since the entire network have a very large capacity and thus a strong tendency to overfit with such a small dataset. The features learned by low-level convolutional blocks are more general, less abstract than those found higher-up, so it is sensible to keep the first few blocks fixed (more general features) and only fine-tune the last one (more specialized features).

The traininig is done with SGD and a very slow learning rate. We do it in that way to make sure the magnitude of the updates stays very small, so as not to wreck the previously learned features.

# Task 4 - Improve if possible

To improve the performance of our classifier we considered the idea of using an external dataset of buildings to fine tune more layers of the network. By doing that, we aim to train a network more specialized in buildings so that when we perform the transfer learning for the Terrassa Building we get better results. 

Making a quick search we have found the following datasets:

- Places dataset (10M images) 
- Oxford Buildings dataset (5K images)  
- Paris Buildings dataset (6K images)

We discarded the first option because we could not find any pretrained network with this dataset written in keras. Besides, it is not a dataset of buildings and many of the classes would not have been helpful for the task. That's why we decided to go with the Oxford Building Dataset. However, after taking a look at the images we found out that many of them were not buildings. 
<img src="https://github.com/AdriaGS/AdriaGS.github.io/blob/master/Images/OxfordimgKK.png" width="500">


# Task 5 - Cycle Generative Adversari Network (GAN)
