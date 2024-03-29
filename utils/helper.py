import numpy as np
import io
import torch
import torchvision
import collections
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Constants
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
NUM_STEPS = 5
ALPHA = 0.025

def get_all_labels():
    """Return all the possible labels on which ResNet was trained on."""
    
    weights = torchvision.models.ResNet50_Weights.DEFAULT
    labels = weights.meta["categories"]
    
    return labels

def load_model():
    """Load and return the pre-trained ResNet50 model."""
    
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    model.eval()
    
    return model


def preprocess_image(image_path, mean=MEAN, std=STD):
    """Preprocess the input image."""
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    img = Image.open(image_path)
    input_image = preprocess(img)
    input_image = input_image.unsqueeze(0)
    input_image.requires_grad = True
    
    return input_image


def predict_class(model, input_image):
    """Predict the class of the input image."""
    
    output = model(input_image)
    label_index = torch.max(output, 1).indices[0].item()
    output_probs = F.softmax(output, dim=1)
    label_prob = np.round(torch.max(output_probs, 1)[0][0].item() * 100, 2)
    
    return label_index, label_prob


def get_class_name(label_index):
    """Get dictionary to match indices with the classes names."""
    
    labels = get_all_labels()
    return labels[label_index]
    

def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, collections.abc.Iterable):
        for elem in x:
            zero_gradients(elem)


def perform_attack(input_image, model, target_class, epsilon, num_steps=NUM_STEPS, alpha=ALPHA):
    """Perform adversarial attack on the input image."""
    
    for _ in range(num_steps):
        zero_gradients(input_image.grad)
        output = model(input_image)
        loss = torch.nn.CrossEntropyLoss()
        loss_val = loss(output, torch.LongTensor([target_class]))
        loss_val.backward()
        x_grad = alpha * torch.sign(input_image.grad.data)
        adv_temp = input_image.data - x_grad
        total_grad = adv_temp - input_image
        total_grad = torch.clamp(total_grad, -epsilon, epsilon)
        x_adv = input_image + total_grad
        input_image.data = x_adv
            
    return input_image, total_grad


def visualize(x, x_adv, x_grad, epsilon, clean_pred, adv_pred, clean_prob, adv_prob):

    x = x.squeeze(0)     #remove batch dimension # B X C H X W ==> C X H X W
    x = x.mul(torch.FloatTensor(STD).view(3,1,1)).add(torch.FloatTensor(MEAN).view(3,1,1)).detach().numpy()#reverse of normalization op- "unnormalize"
    x = np.transpose( x , (1,2,0))   # C X H X W  ==>   H X W X C
    x = np.clip(x, 0, 1)

    x_adv = x_adv.squeeze(0)
    x_adv = x_adv.mul(torch.FloatTensor(STD).view(3,1,1)).add(torch.FloatTensor(MEAN).view(3,1,1)).detach().numpy()#reverse of normalization op
    x_adv = np.transpose( x_adv , (1,2,0))   # C X H X W  ==>   H X W X C
    x_adv = np.clip(x_adv, 0, 1)

    x_grad = x_grad.squeeze(0).detach().numpy()
    x_grad = np.transpose(x_grad, (1,2,0))
    x_grad = np.clip(x_grad, 0, 1)

    figure, ax = plt.subplots(1,3, figsize=(18,8))
    ax[0].imshow(x)
    ax[0].set_title('Clean Example', fontsize=20)


    ax[1].imshow(x_grad)
    ax[1].set_title('Perturbation', fontsize=20)
    ax[1].set_yticklabels([])
    ax[1].set_xticklabels([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])


    ax[2].imshow(x_adv)
    ax[2].set_title('Adversarial Example', fontsize=20)

    ax[0].axis('off')
    ax[2].axis('off')

    ax[0].text(1.1,0.5, "+{}*".format(round(epsilon,3)), size=15, ha="center",
             transform=ax[0].transAxes)

    ax[0].text(0.5,-0.17, "Prediction: {}\n Probability: {}".format(clean_pred, clean_prob), size=15, ha="center",
         transform=ax[0].transAxes)

    ax[1].text(1.1,0.5, " = ", size=15, ha="center", transform=ax[1].transAxes)

    ax[2].text(0.5,-0.17, "Prediction: {}\n Probability: {}".format(adv_pred, adv_prob), size=15, ha="center",
         transform=ax[2].transAxes)

    plt.show()