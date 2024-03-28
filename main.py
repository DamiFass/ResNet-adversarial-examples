




def main():
    
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
    

if __name__ == "__main__":
    run(main)