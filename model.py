from transformers import ViTForImageClassification, ViTFeatureExtractor

def pretrained(MODEL_DIR):
    model = ViTForImageClassification.from_pretrained(MODEL_DIR)
    feature_extractor = ViTFeatureExtractor.from_pretrained(MODEL_DIR)
    model.config.output_hidden_states = True
    return model, feature_extractor