from deepface import DeepFace

# 1. Load the DeepFace wrapper client
arcface_client = DeepFace.build_model("ArcFace")

# 2. Access the raw underlying Keras neural network and print the summary
arcface_client.model.summary()