import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

MODEL_PATH = 'model/final_CNN_model.h5'

@st.cache_resource
def load_cnn_model(path):
    try:

        model = load_model(path, custom_objects={'softmax_v2': tf.nn.softmax})
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        model = None
    return model

def main():
    st.title("Adivinhar Número")
    st.markdown("---")
    
    model = load_cnn_model(MODEL_PATH)
    st.subheader("Carregar imagem para adivinhar")
    
    uploaded_file = st.file_uploader(
        "Escolha uma imagem de dígito(PNG, JGP, JGEP): ", 
        type=["png", "jpg", "jpeg"]
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        st.image(image, caption='Imagem de entrada.', width=150)
        
        try:
            img_resized = image.convert('L').resize((28, 28))
            
            img_array = np.array(img_resized, dtype=np.float32) 
            
            img_array = img_array / 255.0
            
            img_input = img_array.reshape(1, 28, 28, 1)
            prediction = model.predict(img_input)
            predicted_class = np.argmax(prediction, axis=1)[0]
            confidence = np.max(prediction) * 100
            
            st.success(f" O digito provavelmente é:: **{predicted_class}**")
            st.info(f"Confiança: **{confidence:.2f}%**")

            st.markdown("---")
            st.subheader("Distribuição de Probabilidades")
            
            df_prob = pd.DataFrame(prediction[0].reshape(1, 10), 
                                    columns=[str(i) for i in range(10)], 
                                    index=['Probabilidade'])
            st.bar_chart(df_prob.T) 
            
        except Exception as e:
            st.error(f"Erro ao processar a imagem: {e}")
            st.warning("Não é um numero")

if __name__ == "__main__":
    main()