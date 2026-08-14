# Grad-CAM Hook
class GetFeatures:
    features = None

    @staticmethod
    def hook_fn(module, input, output):
        GetFeatures.features = output

model.base_model.layer4[-1].register_forward_hook(GetFeatures.hook_fn)

# Generate Grad-CAM Heatmap
def generate_heatmap(image_tensor, pred):
    model(image_tensor)
    gradients = torch.autograd.grad(
        outputs=model(image_tensor)[:, pred], inputs=GetFeatures.features, create_graph=True
    )[0]
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    for i in range(len(pooled_gradients)):
        GetFeatures.features[0, i, :, :] *= pooled_gradients[i]
    heatmap = torch.mean(GetFeatures.features, dim=1).squeeze().detach().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max()
    return heatmap

# Overlay Heatmap on Original Image
def overlay_heatmap(heatmap, original_image):
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(original_image, 0.5, heatmap_colored, 0.5, 0)

# Save Image with Prediction Label
def save_image_with_label(image, label, output_path):
    # Ensure the image is in RGB format before overlaying the text (OpenCV uses BGR by default)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Define the font, position, and other parameters for overlaying the text
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"Prediction: {label}"
    position = (10, 30)  # Position of the text on the image
    font_scale = 1
    font_color = (0, 255, 0)  # Green color for the text
    thickness = 2
    line_type = cv2.LINE_AA

    # Overlay the text on the image
    cv2.putText(image_rgb, text, position, font, font_scale, font_color, thickness, line_type)

    # Save the image with the prediction label
    cv2.imwrite(output_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))  # Convert back to BGR for OpenCV saving
