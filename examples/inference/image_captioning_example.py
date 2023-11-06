from hezar import Model

if __name__ == '__main__':

    # model = Model.load("hezarai/vit-roberta-fa-image-captioning-flickr30k")
    model = Model.load("hezarai/vit-gpt2-fa-image-captioning-flickr30k")
    texts = model.predict("../assets/image_captioning_example.jpg")
    print(texts)
