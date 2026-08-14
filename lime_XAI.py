# LIME Explanation
def lime_explanation(image_path, predicted_class):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, (224, 224)).astype(np.float32) / 255.0  # Normalize to [0, 1]

    explainer = lime_image.LimeImageExplainer()

    def batch_predict(images):
        images = torch.stack([transforms.ToTensor()(img).to(device).float() for img in images])
        with torch.no_grad():
            outputs = model(images)
        return outputs.cpu().numpy()

    explanation = explainer.explain_instance(
        image_resized, batch_predict, top_labels=1, hide_color=0, num_samples=1000
    )

    top_label = explanation.top_labels[0]  # Get the top predicted label by LIME
    temp, mask = explanation.get_image_and_mask(
        top_label, positive_only=True, num_features=10, hide_rest=False
    )

    # Normalize the mask to [0, 255] for saving as an image
    mask = (mask * 255).astype(np.uint8)
    return image, mask, explanation, top_label




