from hezar import Model

if __name__ == '__main__':
    hub_path = "hezarai/distilbert-fa-sentiment-dksf"
    model = Model.load(hub_path, device="cpu")
    inputs = ["کتابخانه هزار، بهترین کتابخانه هوش مصنوعیه"]
    model_outputs = model.predict(inputs)
    print(model_outputs)
