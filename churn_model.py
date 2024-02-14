import pypickle
import streamlit as st
import numpy as np
churn_data = pypickle.load("churn_model2.pkl")

st.title("CHURN PREDICTION MODEL")
montant = st.number_input("MONTANT")
st.write(montant)
frequence_rech = st.number_input("FREQUENCE_RECH")
st.write(frequence_rech)
revenue = st.number_input("REVENUE")
st.write(revenue)
arpu_segment = st.number_input("ARPU_SEGMENT")
st.write(arpu_segment)
frequence = st.number_input("FREQUENCE")
st.write(frequence)
data_volume = st.number_input("DATA_VOLUME")
st.write(data_volume)
on_net = st.number_input("ON_NET")
st.write(on_net)
orange = st.number_input("ORANGE")
st.write(orange)
tigo = st.number_input("TIGO")
st.write(tigo)
zone_one = st.number_input("ZONE1")
st.write(zone_one)
zone_two = st.number_input("ZONE2")
st.write(zone_two)
regularity = st.number_input("REGULARITY")
st.write(regularity)


freq_top_pack = st.number_input("FREQ_TOP_PACK")
st.write(freq_top_pack)

a = np.array([montant, frequence_rech, revenue, arpu_segment, frequence, data_volume,on_net, orange, tigo, zone_one, zone_two, regularity, freq_top_pack])
x = a.reshape(1,-1)
if st.button("PREDICT"):
    predicted = churn_data.predict(x)
    st.success(f"CHURN = {predicted}")

