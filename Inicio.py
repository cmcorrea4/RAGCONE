import streamlit as st
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings  # Actualizado
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores.pinecone import Pinecone as LangchainPinecone  # Actualizado
from langchain_community.callbacks import get_openai_callback  # Actualizado
from gtts import gTTS
import base64
import os
from tempfile import NamedTemporaryFile
import re


# Configuración de la página
st.set_page_config(page_title="Consulta de Base de Datos Vectorial", layout="wide")
st.title("🔍 Sistema de Consulta Inteligente con Pinecone")

# Función para obtener índices de Pinecone
def get_pinecone_indexes(api_key):
    try:
        pc = Pinecone(api_key=api_key)
        current_indexes = pc.list_indexes().names()
        return list(current_indexes)
    except Exception as e:
        st.error(f"Error al obtener índices: {str(e)}")
        return []

# Función para limpiar estados
def clear_all_states():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.experimental_rerun()

# Inicialización de estados
if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False

# Sidebar para configuración
with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    
    # Campo para OpenAI API Key
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Introduce tu API key de OpenAI"
    )
    
    # Campo para Pinecone API Key
    pinecone_api_key = st.text_input(
        "Pinecone API Key",
        type="password",
        help="Introduce tu API key de Pinecone"
    )
    
    # Selector de modelo LLM
    llm_model = st.selectbox(
        "Modelo LLM",
        options=["gpt-4", "gpt-3.5-turbo"],
        help="Selecciona el modelo de lenguaje a utilizar"
    )
    
    # Selector de idioma para TTS
    tts_lang = st.selectbox(
        "Idioma para Text-to-Speech",
        options=["es", "en", "fr", "de", "it", "pt"],
        format_func=lambda x: {
            "es": "Español", "en": "English", "fr": "Français",
            "de": "Deutsch", "it": "Italiano", "pt": "Português"
        }[x],
        help="Selecciona el idioma para la conversión de texto a voz"
    )
    
    # Temperatura del modelo
    temperature = st.slider(
        "Temperatura",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        help="Controla la creatividad de las respuestas"
    )
    
    # Botones de control
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Actualizar"):
            st.experimental_rerun()
    with col2:
        if st.button("🗑️ Limpiar"):
            clear_all_states()
    
    # Verificar conexión con Pinecone
    if pinecone_api_key:
        try:
            st.markdown("### 📊 Estado")
            available_indexes = get_pinecone_indexes(pinecone_api_key)
            
            if available_indexes:
                st.success("✅ Conectado a Pinecone")
                
                # Selector de índice
                selected_index = st.selectbox(
                    "Selecciona un índice",
                    options=available_indexes
                )
                
                # Mostrar información del índice seleccionado
                if selected_index:
                    pc = Pinecone(api_key=pinecone_api_key)
                    index = pc.Index(selected_index)
                    stats = index.describe_index_stats()
                    
                    # Mostrar estadísticas básicas
                    st.markdown("#### 📈 Estadísticas")
                    total_vectors = stats.get('total_vector_count', 0)
                    st.metric("Total de vectores", total_vectors)
                    
                    # Mostrar namespaces disponibles
                    if 'namespaces' in stats:
                        st.markdown("#### 🏷️ Namespaces")
                        namespaces = list(stats['namespaces'].keys())
                        if namespaces:
                            selected_namespace = st.selectbox(
                                "Selecciona un namespace",
                                options=namespaces
                            )
                            st.session_state.namespace = selected_namespace
                        else:
                            st.info("No hay namespaces disponibles")
                            st.session_state.namespace = ""
            else:
                st.warning("⚠️ No hay índices disponibles")
                selected_index = None
                
        except Exception as e:
            st.error(f"❌ Error de conexión: {str(e)}")
            selected_index = None
    else:
        selected_index = None

def autoplay_audio(audio_data):
    """Función para reproducir audio automáticamente en Streamlit."""
    b64 = base64.b64encode(audio_data).decode()
    md = f"""
        <audio autoplay>
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """
    st.markdown(md, unsafe_allow_html=True)

def text_to_speech(text, lang="es"):
    """Convierte texto a voz usando Google Text-to-Speech."""
    try:
        texto_limpio = re.sub(r"[\*\#]", "", text)
        tts = gTTS(text=texto_limpio, lang=lang, slow=False)
        
        with NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tts.save(tmp_file.name)
            with open(tmp_file.name, "rb") as audio_file:
                audio_bytes = audio_file.read()
            os.unlink(tmp_file.name)
            return audio_bytes
    except Exception as e:
        st.error(f"Error en la generación de audio: {str(e)}")
        return None

def query_pinecone(query_text, namespace, k=5):
    try:
        # Inicializar embeddings y LLM
        embedding_model = OpenAIEmbeddings(
            openai_api_key=openai_api_key,
            model="text-embedding-ada-002"
        )
        
        llm = ChatOpenAI(
            temperature=temperature,
            model_name=llm_model,
            openai_api_key=openai_api_key
        )
        
        # Inicializar Pinecone
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index(selected_index)
        
        # Crear vector store
        vectorstore = LangchainPinecone(
            index=index,
            embedding=embedding_model,
            text_key="text",
            namespace=namespace
        )
        
        # Crear cadena de QA con recuperación
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": k}),
            return_source_documents=True
        )
        
        # Obtener respuesta de la cadena
        with get_openai_callback() as cb:
            response = qa_chain.invoke({"query": query_text})
            print(f"Uso de tokens: {cb}")
        
        # Extraer respuesta y documentos fuente
        answer = response["result"]
        source_docs = response["source_documents"]
        
        # Convertir documentos fuente al formato de resultados
        results = {
            "matches": [
                {
                    "metadata": doc.metadata,
                    "score": doc.metadata.get("score", 0.0) if hasattr(doc, "metadata") else 0.0,
                    "text": doc.page_content
                }
                for doc in source_docs
            ]
        }
        
        return type('Results', (), results)(), answer
        
    except Exception as e:
        st.error(f"Error en la consulta: {str(e)}")
        return None, None

# Interfaz principal
if openai_api_key and pinecone_api_key and selected_index:
    st.markdown("### 🔍 Realizar Consulta")
    
    # Parámetros de búsqueda
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("💭 ¿Qué deseas consultar?")
    with col2:
        k = st.number_input("Número de resultados", min_value=1, max_value=10, value=3)
    
    # Botón de búsqueda
    if st.button("🔍 Buscar"):
        if query:
            with st.spinner("🔄 Buscando y procesando..."):
                results, answer = query_pinecone(
                    query,
                    namespace=getattr(st.session_state, 'namespace', ''),
                    k=k
                )
                
                if results and hasattr(results, 'matches'):
                    # Mostrar respuesta generada
                    st.markdown("### 🤖 Respuesta:")
                    st.write(answer)
                    
                    # Generar y mostrar audio
                    st.markdown("### 🔊 Escuchar Respuesta")
                    audio_data = text_to_speech(answer, tts_lang)
                    if audio_data:
                        st.audio(audio_data, format="audio/mp3")
                        st.download_button(
                            label="⬇️ Descargar Audio",
                            data=audio_data,
                            file_name="respuesta.mp3",
                            mime="audio/mp3"
                        )
                    
                    # Mostrar fuentes
                    st.markdown("### 📚 Fuentes Consultadas:")
                    
                    for i, match in enumerate(results.matches, 1):
                        score = match.get('score', 0)
                        similarity = round((1 - (1 - score)) * 100, 2) if score else 0
                        
                        with st.expander(f"📍 Fuente {i} - Similitud: {similarity}%"):
                            if 'text' in match:
                                st.write(match['text'])
                            else:
                                st.write("No se encontró texto en los metadatos")
                            
                            # Mostrar metadatos adicionales
                            other_metadata = {k:v for k,v in match['metadata'].items() if k != 'text'}
                            if other_metadata:
                                st.markdown("##### Metadatos adicionales:")
                                st.json(other_metadata)
                else:
                    st.warning("No se encontraron resultados")
        else:
            st.warning("⚠️ Por favor, ingresa una consulta")
else:
    st.info("👈 Por favor, configura las credenciales en el panel lateral para comenzar")

# Información en el sidebar
with st.sidebar:
    with st.expander("### ℹ️ Acerca de esta aplicación"):
        st.markdown("---")
        st.markdown("### ℹ️ Características")
        st.write("""
        Esta aplicación te permite realizar consultas semánticas mejoradas con IA en bases de datos
        vectoriales existentes en Pinecone.
        
        Características:
        - Búsqueda semántica utilizando documentos similares
        - Respuestas generadas solo con la información recuperada
        - Conexión directa a índices de Pinecone
        - Soporte para múltiples namespaces
        - Visualización de similitud y fuentes
        - Reproducción de audio de respuesta
        """)
