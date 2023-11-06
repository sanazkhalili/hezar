from hezar import Model


# model = Model.load("hezarai/trocr-base-fa-v1")  # TrOCR
if __name__ == '__main__':
    model = Model.load("hezarai/crnn-base-fa-64x256")  # CRNN
    text = model.predict("../assets/ocr_example.jpg")
    print(text)
