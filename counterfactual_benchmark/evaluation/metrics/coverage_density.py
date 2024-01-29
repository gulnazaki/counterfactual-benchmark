from embeddings.vgg import vgg

def coverage_density(real_images, generated_images, embedding_fn=vgg):
    real_features = [vgg(image) for image in real_images]
    generated_features = [vgg(image) for image in generated_images]

    # blabla