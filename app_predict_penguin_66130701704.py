import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# ฟังก์ชันสำหรับโหลดโมเดลและตัวแปลงที่ใช้ในการฝึก
@st.cache
def load_model():
    with open('model_penguin.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('island_encoder.pkl', 'rb') as f:
        island_encoder = pickle.load(f)
    with open('sex_encoder.pkl', 'rb') as f:
        sex_encoder = pickle.load(f)
    with open('species_encoder.pkl', 'rb') as f:
        species_encoder = pickle.load(f)
    return model, island_encoder, sex_encoder, species_encoder

# ฟังก์ชันสำหรับแปลงข้อมูลและทำนาย
def predict_penguin_species(model, island_encoder, sex_encoder, species_encoder, island, sex, culmen_length_mm, culmen_depth_mm, flipper_length_mm, body_mass_g):
    # สร้าง DataFrame สำหรับข้อมูลที่กรอก
    x_new = pd.DataFrame({
        'island': [island],
        'culmen_length_mm': [culmen_length_mm],
        'culmen_depth_mm': [culmen_depth_mm],
        'flipper_length_mm': [flipper_length_mm],
        'body_mass_g': [body_mass_g],
        'sex': [sex]
    })

    # แปลงข้อมูล categorical โดยใช้ LabelEncoder
    x_new['island'] = island_encoder.transform(x_new['island'])
    x_new['sex'] = sex_encoder.transform(x_new['sex'])

    # ทำนายผลลัพธ์จากโมเดล
    prediction = model.predict(x_new)

    # แปลงผลลัพธ์กลับเป็นชื่อสายพันธุ์
    predicted_species = species_encoder.inverse_transform(prediction)
    
    return predicted_species[0]

# การสร้างแอปด้วย Streamlit
def app():
    # โหลดโมเดลและตัวแปลงที่ใช้
    model, island_encoder, sex_encoder, species_encoder = load_model()

    # ตั้งชื่อและคำอธิบายแอป
    st.title("Penguin Species Prediction")
    st.write("This app predicts the species of a penguin based on physical characteristics.")

    # ฟอร์มให้ผู้ใช้กรอกข้อมูล
    st.sidebar.header("Input Features")

    island = st.sidebar.selectbox("Island", ["Torgersen", "Biscoe", "Dream"])
    sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
    culmen_length_mm = st.sidebar.slider("Culmen Length (mm)", 30.0, 70.0, 45.0)
    culmen_depth_mm = st.sidebar.slider("Culmen Depth (mm)", 10.0, 25.0, 15.0)
    flipper_length_mm = st.sidebar.slider("Flipper Length (mm)", 150.0, 250.0, 200.0)
    body_mass_g = st.sidebar.slider("Body Mass (g)", 2500, 6500, 4000)

    # ปุ่มทำนาย
    if st.sidebar.button("Predict"):
        # ทำการทำนาย
        predicted_species = predict_penguin_species(
            model, island_encoder, sex_encoder, species_encoder,
            island, sex, culmen_length_mm, culmen_depth_mm, flipper_length_mm, body_mass_g
        )

        # แสดงผลการทำนาย
        st.write(f"The predicted penguin species is **{predicted_species}**.")

        # แสดงผลการประเมินโมเดล (ถ้ามี)
        if st.checkbox("Show Model Evaluation Report"):
            # โหลดข้อมูล y_test และ X_test (ต้องจัดเตรียมไฟล์)
            y_test = pd.read_csv("y_test.csv")  # แทนที่ด้วยข้อมูลจริง
            X_test = pd.read_csv("X_test.csv")  # แทนที่ด้วยข้อมูลจริง
            from sklearn.metrics import classification_report
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred)
            st.text(report)

# รันแอป
if __name__ == "__main__":
    app()


