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
    
    parser.add_argument('--img_path', default='data/espresso.jpg', type=str, required=False, help='path to your image')
    parser.add_argument('--target_class_index', default=9, type=int, required=False, help='index of your target class - chose between 0 and 999')
    parser.add_argument('--epsilon', default=0.25, type=float, required=False, help='magnitude of modification to the inital image')
    args = parser.parse_args()
        
    # Load pretrained model
    model = load_model()
    
    # Load and preprocess input image
    image_path = args.img_path
    input_image = preprocess_image(image_path)
    
    # Get initial prediction
    initial_label_index, initial_label_prob = predict_class(model, input_image)
    initial_label = get_class_name(initial_label_index)
    print('\n')
    print(f'The model has labelled the INPUT IMAGE as {initial_label.upper()}, with {initial_label_prob}% probability')
    
    # Choose target class
    target_index = args.target_class_index
    target_label = get_class_name(target_index)
    
    # Perform adversarial attack
    adversarial_image, gradient = perform_attack(input_image, model, target_index, args.epsilon)
    
    # Get prediction on adversarial image
    adversarial_label_index, adversarial_label_prob = predict_class(model, adversarial_image)
    adv_label = get_class_name(adversarial_label_index)
    print('\n')
    print(f'The ADVERSARIAL IMAGE with target class {target_label.upper()} was predicted by the model as {adv_label.upper()}, with {adversarial_label_prob}% probability')
    print('\n')
    
    # Visualize output:
    visualize(input_image, adversarial_image, gradient, args.epsilon, initial_label, adv_label, initial_label_prob, adversarial_label_prob)
    
