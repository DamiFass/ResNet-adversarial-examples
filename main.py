import argparse
import yaml
import os

from utils.helper import (get_all_labels, 
                          load_model,
                          preprocess_image,
                          predict_class,
                          get_class_name,
                          zero_gradients,
                          perform_attack,
                          visualize)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--img_path', type=str, required=True, help='path to your image')
    parser.add_argument('--target_class_index', type=int, required=True, help='index of your target class')
    
    args = parser.parse_args()
    
    # Load pretrained model
    model = load_model()
    
    # Load and preprocess input image
    image_path = "your_image_path.jpg"
    input_image = preprocess_image(image_path)
    
    # Get initial prediction
    initial_label_index, initial_label_prob = predict_class(model, input_image)
    print(f'Initial prediction: {classes[initial_label_index]} class, with {initial_label_prob}% probability')
    
    # Choose target class
    target_index = 9
    
    # Perform adversarial attack
    adversarial_image = attack_image(input_image, model, target_index)
    
    # Get prediction on adversarial image
    adversarial_label_index, adversarial_label_prob = predict_class(model, adversarial_image)
    print(f'Adversarial prediction: {classes[adversarial_label_index]} class, with {adversarial_label_prob}% probability')
    
