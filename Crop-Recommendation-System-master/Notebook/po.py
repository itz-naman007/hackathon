import gradio as gr
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load models
catboost_model = joblib.load(r"C:\Users\naman\Downloads\Crop-Recommendation-System-master\Crop-Recommendation-System-master\Notebook\catboost_model.pkl")
random_forest_model = joblib.load(r"C:\Users\naman\Downloads\Crop-Recommendation-System-master\Crop-Recommendation-System-master\Notebook\random_forest_model.pkl")
decision_tree_model = joblib.load(r"C:\Users\naman\Downloads\Crop-Recommendation-System-master\Crop-Recommendation-System-master\Notebook\decision_tree_model.pkl")
meta_model = joblib.load(r"C:\Users\naman\Downloads\Crop-Recommendation-System-master\Crop-Recommendation-System-master\Notebook\meta_model.pkl")

# Load label encoder classes
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load(r"C:\Users\naman\Downloads\Crop-Recommendation-System-master\Crop-Recommendation-System-master\Notebook\label_classes.npy", allow_pickle=True)

# Prediction function
def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    input_data = [[N, P, K, temperature, humidity, ph, rainfall]]

    catboost_pred = catboost_model.predict(input_data)
    rf_pred = random_forest_model.predict(input_data)
    dt_pred = decision_tree_model.predict(input_data)

    meta_input = [[catboost_pred[0], rf_pred[0], dt_pred[0]]]
    final_encoded = meta_model.predict(meta_input)[0]
    final_label = label_encoder.inverse_transform([final_encoded])[0]

    return final_label


html_css = """
<style>
    .gradio-container {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 10px;
    }
    .gr-button {
        background-color: #4CAF50;
        color: white;
    }
</style>
"""

# Define the interface
with gr.Blocks() as interface:
    gr.HTML(html_css)
    gr.Markdown("## 🌾 Crop Recommendation System with Meta Model")
    gr.Markdown("Enter soil and climate details to get the best crop suggestion.")

    with gr.Row():
        N = gr.Number(label="Nitrogen (N)")
        P = gr.Number(label="Phosphorous (P)")
        K = gr.Number(label="Potassium (K)")
    with gr.Row():
        temp = gr.Number(label="Temperature (°C)")
        hum = gr.Number(label="Humidity (%)")
        ph = gr.Number(label="pH")
        rain = gr.Number(label="Rainfall (mm)")
    
    output = gr.Textbox(label="🌱 Recommended Crop")

    submit_btn = gr.Button("Recommend Crop")
    submit_btn.click(fn=predict_crop, inputs=[N, P, K, temp, hum, ph, rain], outputs=output)

# Launch
interface.launch(share=True)