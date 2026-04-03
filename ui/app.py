# ui/app.py
import streamlit as st
import requests
from PIL import Image
import io

FASTAPI_URL = "http://localhost:8000"

# ── Página ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Animal CNN", page_icon="🐾", layout="wide")

st.title("🐾 Clasificador de Animales con CNN")
st.caption("Red Neuronal Convolucional · 15 clases · PyTorch")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📊 Info del Modelo")
    st.markdown("""
| Campo | Valor |
|---|---|
| **Arquitectura** | CNN 3 bloques Conv2D |
| **Clases** | 15 animales |
| **Input** | 128x128 RGB |
| **Framework** | PyTorch |
""")
    st.divider()
    st.markdown("**Animales detectables:**")
    st.markdown("""
    🐻 Bear · 🐦 Bird · 🐱 Cat · 🐄 Cow · 🦌 Deer
    🐶 Dog · 🐬 Dolphin · 🐘 Elephant · 🦒 Giraffe · 🐴 Horse
    🦘 Kangaroo · 🦁 Lion · 🐼 Panda · 🐯 Tiger · 🦓 Zebra
    """)

st.divider()

# ── Controles principales ─────────────────────────────────────────────────────
mode = st.radio("Modo", ["🏋️ Entrenar", "🔍 Predecir"], horizontal=True)

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# MODO ENTRENAR
# ════════════════════════════════════════════════════════════════════════════
if mode == "🏋️ Entrenar":
    st.subheader("🏋️ Entrenamiento del Modelo")
    st.info("El dataset ya está descargado en `data/animal_data/`")

    col1, col2, col3 = st.columns(3)
    with col1:
        epochs = st.slider("Épocas", 1, 50, 10)
    with col2:
        batch_size = st.slider("Batch Size", 8, 64, 32)
    with col3:
        lr = st.selectbox("Learning Rate", [0.001, 0.0001, 0.01], index=0)

    st.warning("⚠️ El entrenamiento puede tardar varios minutos dependiendo de tu PC.")

    if st.button("🚀 Iniciar Entrenamiento", type="primary"):
        data_form = {
            "epochs":     str(epochs),
            "batch_size": str(batch_size),
            "lr":         str(lr)
        }
        with st.spinner("Entrenando la CNN... por favor espera."):
            try:
                res = requests.post(
                    f"{FASTAPI_URL}/train/",
                    data=data_form,
                    timeout=3600
                )
                if res.ok and res.json().get("status") == "ok":
                    st.success("✅ Modelo entrenado y guardado correctamente")
                    st.json(res.json())
                else:
                    st.error("❌ Error durante el entrenamiento")
                    st.code(res.text)
            except requests.exceptions.ConnectionError:
                st.error("❌ No se pudo conectar a FastAPI.")
                st.code("python -m uvicorn app.main:app --reload --port 8000")

# ════════════════════════════════════════════════════════════════════════════
# MODO PREDECIR
# ════════════════════════════════════════════════════════════════════════════
elif mode == "🔍 Predecir":
    st.subheader("🔍 Clasificar una Imagen")

    uploaded = st.file_uploader(
        "Sube una imagen de un animal",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded:
        # Mostrar imagen
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded, caption="Imagen subida", use_container_width=True)

        if st.button("🔬 Clasificar", type="primary"):
            with st.spinner("Analizando imagen..."):
                try:
                    files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                    res   = requests.post(
                        f"{FASTAPI_URL}/predict/",
                        files=files,
                        timeout=60
                    )
                    if res.ok:
                        result = res.json()

                        if result.get("status") == "error":
                            st.error(f"❌ {result['message']}")
                        else:
                            with col2:
                                st.success("✅ Clasificación completada")
                                st.metric(
                                    "Animal detectado",
                                    result["prediccion"],
                                    f"{result['confianza']}% confianza"
                                )
                                st.divider()
                                st.markdown("**Top 3 predicciones:**")
                                for i, pred in enumerate(result["top3"]):
                                    emoji = ["🥇", "🥈", "🥉"][i]
                                    st.progress(
                                        pred["probabilidad"] / 100,
                                        text=f"{emoji} {pred['animal']}: {pred['probabilidad']}%"
                                    )
                    else:
                        st.error(f"❌ Error {res.status_code}")
                        st.code(res.text)

                except requests.exceptions.ConnectionError:
                    st.error("❌ No se pudo conectar a FastAPI.")
                    st.code("python -m uvicorn app.main:app --reload --port 8000")