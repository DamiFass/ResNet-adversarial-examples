# Project Title

Small Python project to easily generate adversarial examples for the [ResNet50](https://arxiv.org/pdf/1512.03385.pdf)

## Description

There are several methods to create adversarial images for image recognition deep learning models. This project implements on of the most famous approaches, the Fast Gradient Sign Method (FSGM), first published by [Goodfellow et al.](https://arxiv.org/abs/1412.6572).

The idea behind this method is to use the backpropagated gradients to modify the input image, with the target of maximising the loss (instead of using the gradient to modify the network weights with the target of minimizing the loss).

In this project we implemented an iterative version of the FGSM method, where essentially the algorithm is repeated for a predefined number of steps, initializing the adversarial image as the original image and taking a small step at each iteration along the direction of the gradient.

## Dependencies

All it is needed are the following 3 libraries:

```
numpy 
pillow
torchvision
```

## Execution

* Just run the ```main.py``` script - it will automatically run with an example input.

* The ```main.py``` script takes in input 3 arguments:
    * img-path: the path to the image you want to use. The default value is set to one of the examples in the data folder, an image of an espresso coffe. 
    * target_index_class: the index of the target class. We want our model to classify the adversarial image as this class. This should be a number between 0 and 999, because the ResNet50 model we are attacking has been trained on the ImageNet1K_v1 dataset, which has 1000 classes. You can select any number, as long as it is different from the index of the correct class. The default is set to 9, the index of the 'ostrich' class.
    * epsilon: this is the magnitude of the perturbation. Bigger values are more likely to successfully fool the model, but they are also more likey to visibly modify the input image. The default is 0.25.

* Example usege from Linux command line:
```python3 main.py --img_path=data/ostrich.jpg --target_class_index=421```

* The script will display the original image, the perturbation and the adversarial image, along with the predicted classes and the probabilities with which the model has assigned each image its class.
