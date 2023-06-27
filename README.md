# CTransCNN: Combining Transformer and CNN in Multi-Label Medical Image Classification

![demo](./picture/demo.gif)

## model
![model](./picture/model.png)

![reslut](./picture/result.png)

## Abstract

Multi-label image classification aims to assign images to multiple possible labels. In this task, each image may have multiple labels, making it more challenging than single-label classification problems. For example, Convolutional Neural Networks (CNNs) have not met performance requirement in utilizing statistical dependencies between labels in this study. Additionally, data imbalance is a common problem in machine learning that needs to be considered for multi-label medical image classification. Furthermore, the disadvantage of concatenating CNN and Transformer is the lack of direct interaction and information exchange between the two models. We propose a novel hybrid deep learning model (CTransCNN). It consists of three main parts in CNN branch and Transformer branch: a Multi-label Multi-head Attention Enhanced Feature module (MMAEF), a Multi-Branch Residual module (MBR), and an Information Interaction module (IIM). We adopt the MMAEF to explore the implicit correlations between labels, the MBR for model optimization, and the IIM for feature transmission and increasing nonlinearity between the two branches to help accomplish the multi-label medical image classification task. The publicly available datasets, namely the ChestX-ray11 and NIH ChestX-ray14, along with our self-constructed Traditional Chinese Medicine (TCM) tongue dataset (TCMTD) were utilized for extensive multi-label image classification experiments, in which we compared our approach with state-of-the-art methods. In the experimental results, the commonly used average AUC scores of the CTransCNN model on the ChestX-ray11, NIH ChestX-ray14 and TCMTD reached 83.37%, 78.47% and 84.56%, respectively. In addition, we provide visual explanations using the gradient-based localization method. Compared with previous research, the experimental results show that the framework we developed has strong competitiveness. CTransCNN can be more effective in identifying multi-label images and more detailed information, and achieve the best results on the indicators of the three studied data sets, indicating that the network has a relatively good performance in medical multi-label image classification. Strong generalization ability can be further applied to other medical multi-label image classification tasks.

## Dataset

1. [ChestX-ray11](kaggle.com/competitions/ranzcr-clip-catheter-line-classification/data): In this study, 30,083 CXR image training data were used for multi-label sample classification due to computer limitations.

2. [NIH ChestX-ray14](nihcc.app.box.com/v/ChestXray-NIHCC):  Due to computational limitations, this study performed multi-label sample classification on 51759 CXR images, named NIH ChestX-ray14, rather than the complete set of 112120 frontal images. 

   **ChestX-ray11** and **NIH ChestX-ray14** are shown in folders chest11 and chest14.

3. The TCMTD is a multi-label classification task for 9 different TCM pathologies, conditions viz. ‘Qixu’ (qi deficiency), ‘Qiyu’ (qi stagnation), ‘Shire’ (damp heat), ‘Tanshi’ (phlegm damp), ‘Tebing’ (idiosyncratic), ‘Xueyu’ (blood stagnation), ‘Yinxu’ (yin deficiency), ‘Pinghe’ (balanced), and ‘Yangxu’ (yang deficiency), which is a multi-label classification task. **Due to ethical and personal privacy issues, this dataset is not publicly available.**

## Citation

We now have a related paper that you can cite in this repository 🤗 .

## Thanks

We implemented our models using the open source computer vision library [MMCV](github.com/open-mmlab/mmcv), which was developed by OpenMMLab. We would like to express our gratitude to the developers for their valuable contributions to the research community.
