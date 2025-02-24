import streamlit as st
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

# Configura os parâmetros na barra lateral
st.sidebar.header("Parâmetros para Modelagem de Tópicos")
min_topic_size = st.sidebar.number_input("Tamanho mínimo do tópico", value=2, min_value=1)
#ngram_start = st.sidebar.number_input("Início do n-gram", value=1, min_value=1)
ngram_start = 1
#ngram_end = st.sidebar.number_input("Fim do n-gram", value=1, min_value=ngram_start)
ngram_end = 1
nr_topics_input = st.sidebar.number_input("Número de tópicos (-1 para não reduzir)", value=-1, step=1)
nr_topics = None if nr_topics_input == -1 else nr_topics_input

# Título da aplicação
st.title("Modelagem de Tópicos com BERTopic")

# Upload do arquivo de texto
uploaded_file = st.file_uploader("Envie um arquivo de texto (.txt)", type=["txt"])

if uploaded_file is not None:
    # Leitura do arquivo: cada linha é tratada como um documento
    file_content = uploaded_file.read().decode("utf-8")
    documents = [linha for linha in file_content.splitlines() if linha.strip()]
    st.write(f"Número de documentos encontrados: {len(documents)}")

    # Botão para executar a modelagem
    if st.button("Executar Modelagem de Tópicos"):
        # Criação do vectorizer com intervalo de n-gram e remoção das stopwords em inglês
        vectorizer_model = CountVectorizer(ngram_range=(ngram_start, ngram_end), stop_words='english')

        # Criação e ajuste do modelo BERTopic com os parâmetros definidos
        topic_model = BERTopic(min_topic_size=min_topic_size, vectorizer_model=vectorizer_model, nr_topics=nr_topics)
        topics, probs = topic_model.fit_transform(documents)

        st.write("Informações dos Tópicos:")
        st.dataframe(topic_model.get_topic_info())

        st.header("Visualizações dos Tópicos")

        st.subheader("Heatmap")
        fig_heatmap = topic_model.visualize_heatmap()
        st.plotly_chart(fig_heatmap)

        st.subheader("Bar Chart")
        fig_barchart = topic_model.visualize_barchart()
        st.plotly_chart(fig_barchart)

        st.subheader("Visualização de Documentos")
        # Utilizando o parâmetro correto "docs" na chamada da função
        fig_documents = topic_model.visualize_documents(docs=documents, topics=topics)
        st.plotly_chart(fig_documents)