# Single Image Inference Demo

## Introduction

This demo illustrates how to perform inference on a single image using a deep learning model. 

## Steps

1. **Prepare Image**: First, prepare an image for inference. It can be any image of your interest, such as a picture of a cat or a landscape photo.

2. **Load Model**: Next, load the pre-trained deep learning model. The project uses onnx format, including `CTransCNN_Chest11.onnx`, `CTransCNN_Chest14.onnx` and `CTransCNN_Tongue.onnx`. [Download](https://pan.baidu.com/s/1ofBbLmROo3cLUAnEHpHBig?pwd=pk7s) 

3. **Image Preprocessing**: Before passing the image to the model, preprocessing operations are often required, such as resizing, normalization, or cropping. This ensures that the image conforms to the input requirements of the model.

4. **Inference**: Now, input the preprocessed image into the model for inference. The model will output class labels prediction results.

5. **Result Presentation**: Finally, present the inference results. 