from ultralytics import YOLO

def train(model_config, data_config, epochs):
    # Load a model
    model = YOLO(model_config, task="detect")  # build a new model from scratch

    # Train the model
    model.train(data=data_config, epochs=epochs)