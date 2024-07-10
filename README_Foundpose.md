Input image must be divided by 255.0
Then tranform the image by     
    rgb_normalize = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
Put it thorugh models
The feature afterwards, no need to do the inverse_transformation
Jut need to normalize it and multiply it back to 255
    normalized_image = ((top3_pca - np.min(top3_pca)) / (np.max(top3_pca) - np.min(top3_pca))* 255).astype(np.uint8)
        Actually no need to multiply with 255 and convert to unit8. Normalize it is enough