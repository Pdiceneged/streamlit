import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import base64

st.set_page_config(
    page_title="Similarity ODS",
    page_icon="🌎"
)
@st.cache_data()
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64("fundoesg4k.png")
img2 = get_img_as_base64("esgfundo1.png")

page_bg_img = f"""
<style>
header, footer {{
    visibility: hidden !important;
}}

#MainMenu {{
    visibility: visible !important;
    color: #F44D00;
}}

[data-testid="stAppViewContainer"] > .main {{
    background-image: url("data:fundoesg4k/png;base64,{img}");
    background-size: cover; 
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}
[data-testid="stSidebar"] > div:first-child {{
    background-image: url("data:esgfundo1/png;base64,{img2}");
    background-position: center; 
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}

[data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
    right: 2rem;
}}

.stTextInput>div>div>input[type="text"] {{
    background-color: #CFE6D2; 
    color: #000; 
    border-radius: 7px; 
    border: 2px solid #000010; 
    padding: 5px; 
    width: 500; 
}}

@media (max-width: 360px) {{
    [data-testid="stAppViewContainer"] > .main, [data-testid="stSidebar"] > div:first-child {{
        background-size: auto;
    }}
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)
st.sidebar.image("logoesg.png", width=250)


ods_categoria = {
    1: "Erradicação da pobreza" " (Social)",
    2: "Fome zero e agricultura sustentável" " (Social)",
    3: "Saúde e Bem-Estar" " (Social)",
    4: "Educação de qualidade" " (Social)",
    5: "Igualdade de gênero" " (Social)",
    6: "Água potável e saneamento" " (Ambiental)",
    7: "Energia limpa e acessível" " (Ambiental)",
    8: "Trabalho decente e crescimento econômico" " (Governança)",
    9: "Indústria, inovação e infraestrutura" " (Ambiental)",
    10: "Redução das desigualdades" " (Social)",
    11: "Cidades e comunidades sustentáveis" " (Ambiental)",
    12: "Consumo e produção responsáveis" " (Ambiental)",
    13: "Ação contra a mudança global do clima" " (Ambiental)",
    14: "Vida na água" " (Ambiental)",
    15: "Vida terrestre" " (Ambiental)",
    16: "Paz, Justiça e Instituições Eficazes" " (Governança)",
    17: "Parcerias e meios de implementação" " (Governança)",
}

ods_texts = [

    "Pobreza extrema, inclusão social, desigualdade econômica, fome, vulnerabilidade, acesso à educação, emprego digno, sustentabilidade social, igualdade de oportunidades, resiliência financeira, proteção social, microfinanças, redução da pobreza, empoderamento econômico, assistência social, fornecimento de recursos, renda mínima, justiça social, desenvolvimento inclusivo, apoio comunitário.",
    "Segurança alimentar, agricultura sustentável, desnutrição, agricultura familiar, produção de alimentos, soberania alimentar, desperdício de alimentos, seguro agrícola, acesso à água para irrigação, diversidade alimentar, infraestrutura agrícola, tecnologia agrícola, fome oculta, agricultura de subsistência, comércio justo, políticas alimentares, desenvolvimento rural, agricultura de precisão, sistemas agroflorestais, resiliência agrícola.",
    "nascimento familia, filhos mães pais famillia saude bem estar, familiar cuidado Saúde pública, prevenção de doenças, acesso à saúde, bem-estar mental, vacinação, saneamento básico, educação em saúde, atenção primária, qualidade de vida, cobertura universal de saúde, doenças transmissíveis, saúde materna, saúde infantil, tratamento médico acessível, pesquisa biomédica, infraestrutura de saúde, promoção da saúde, tecnologia médica, assistência médica preventiva, medicina personalizada.",
    "aprendizes  desenvolvimento webnar webinar aprender aprendizado Inclusão educacional, alfabetização, acesso à educação, ensino de qualidade, equidade educacional, educação inclusiva, desenvolvimento de habilidades, tecnologias educacionais, aprendizado ao longo da vida, parcerias educacionais, infraestrutura escolar, qualificação de professores, educação STEM Ciência, Tecnologia, Engenharia e Matemática, educação para a cidadania, avaliação educacional, inovação pedagógica, educação não formal, educação à distância, educação para a sustentabilidade, educação de base comunitária.",
    "Empoderamento feminino, igualdade salarial, participação política das mulheres, direitos reprodutivos, violência de gênero, educação de gênero, equidade de oportunidades, liderança feminina, eliminação de estereótipos de gênero, maternidade segura, saúde sexual e reprodutiva, paridade de gênero, discriminação de gênero, igualdade no local de trabalho, acesso a recursos para mulheres, direitos das mulheres, mulheres na ciência e tecnologia, empreendedorismo feminino, emprego digno para mulheres, participação igualitária em esportes.",
    "Acesso à água potável, saneamento básico, gestão sustentável da água, higiene adequada, tratamento de água, abastecimento de água seguro, infraestrutura hídrica, qualidade da água, uso eficiente da água, reuso de água, saneamento rural, sistemas de esgoto, água e saneamento em situações de emergência, proteção de fontes de água, tecnologias de saneamento inovadoras, promoção da saúde por meio da água limpa, participação comunitária em projetos hídricos, educação sobre recursos hídricos, água e desenvolvimento sustentável, equidade no acesso à água e saneamento.",
    "Energias renováveis, acesso à eletricidade, eficiência energética, energia sustentável, fontes de energia limpa, inovação em energia, eletrificação rural, desenvolvimento de tecnologias verdes, energia solar, energia eólica, biomassa, energia hidrelétrica, hidrogenio verde , infraestrutura energética, acesso a tecnologias de energia, democratização da energia, combate à pobreza energética, educação energética, parcerias para o acesso à energia, investimento em infraestrutura energética, transição para uma matriz energética sustentável.",
    "Emprego digno, crescimento econômico inclusivo, redução do desemprego, igualdade salarial, empreendedorismo, inovação econômica, desenvolvimento de habilidades profissionais, ambientes de trabalho seguros, erradicação do trabalho infantil, trabalho decente para todos, proteção social, emprego sustentável, formalização do trabalho, desenvolvimento de pequenas e médias empresas, bancarização, inclusão financeira, responsabilidade social corporativa, economia verde, comércio justo, redução da desigualdade de renda.",
    "Inovação tecnológica, infraestrutura sustentável, desenvolvimento de tecnologias verdes, industrialização, acesso à tecnologia da informação, pesquisa e desenvolvimento, infraestrutura de transporte, inclusão digital, desenvolvimento de habilidades tecnológicas, desenvolvimento de infraestrutura industrial, investimento em infraestrutura, conectividade global, inovação em energias limpas, desenvolvimento de zonas industriais sustentáveis, acesso a serviços de telecomunicação, modernização da infraestrutura, fomento à inovação, desenvolvimento de parcerias público-privadas, desenvolvimento de infraestrutura rural, promoção da pesquisa aplicada.",
    "Equidade social, distribuição de renda, inclusão social, oportunidades iguais, participação cidadã, desenvolvimento inclusivo, acesso à educação e saúde, mobilidade social, combate à discriminação, proteção dos direitos humanos, políticas de inclusão, empoderamento de comunidades marginalizadas, igualdade de oportunidades econômicas, participação política equitativa, cooperação internacional para redução de desigualdades, acesso a serviços básicos para todos, promoção da diversidade, medidas contra a discriminação de gênero, raça e orientação sexual, acesso a recursos para grupos vulneráveis, emprego e oportunidades justas para todos.",
    "reclica reciclagem reusa reduzir reutilização coleta lixo papel plastico papelão vidro organico Urbanização sustentável, planejamento urbano, habitação acessível, mobilidade sustentável, infraestrutura resiliente, gestão de resíduos sólidos, desenvolvimento de áreas verdes, acesso a serviços básicos nas cidades, participação comunitária, inclusão social urbana, preservação do patrimônio cultural, redução da poluição do ar e sonora, segurança urbana, construção sustentável, eficiência energética em edificações, planejamento de uso do solo, transporte público eficiente, resiliência a desastres naturais, desenvolvimento de tecnologias urbanas sustentáveis, promoção da cultura local nas cidades.",
    "Consumo consciente, produção sustentável, desperdício zero, ciclo de vida dos produtos, eficiência no uso de recursos, economia circular, compras éticas, redução da pegada de carbono, consumo de energia responsável, certificações ambientais, ecoeficiência, incentivo ao consumo local, reutilização de produtos, redução do uso de plástico, educação para o consumo sustentável, responsabilidade das empresas na cadeia de produção, inovação em processos sustentáveis, gestão responsável de resíduos, agricultura sustentável, redução da emissão de poluentes.",
    "Mitigação das mudanças climáticas, adaptação às mudanças climáticas, energias renováveis, redução das emissões de gases de efeito estufa, transição para uma economia de baixo carbono, preservação de ecossistemas cruciais para o clima, conscientização ambiental, medidas para a eficiência energética, planejamento urbano sustentável, tecnologias limpas, resiliência climática, preservação da biodiversidade, agricultura de baixa emissão de carbono, reflorestamento e conservação florestal, cooperação internacional em mudanças climáticas, regulamentação ambiental eficaz, educação sobre mudanças climáticas, financiamento para ações climáticas, monitoramento e relatórios climáticos, participação cidadã na mitigação climática.",
    "agua praia praiano praias acquatica peixe  aquatica agua Conservação marinha, pesca sustentável, preservação de ecossistemas aquáticos, proteção da biodiversidade marinha, combate à poluição dos oceanos, áreas marinhas protegidas, aquicultura sustentável, monitoramento dos oceanos, redução do descarte de plásticos no mar mares praia praias, recuperação de habitats marinhos, conscientização sobre a vida marinha, combate à pesca ilegal, conservação de espécies ameaçadas aquáticas, oceanografia, ecoturismo marinho responsável, restauração de recifes de coral, gestão sustentável de recursos marinhos, ecologia marinha, inovações para a conservação marinha, cooperação internacional em questões oceânicas.praias praia",
    "conserva floresta Conservação da biodiversidade, reflorestamento, combate à desertificação, proteção de ecossistemas terrestres, manejo sustentável de florestas, prevenção da extinção de espécies, conservação de habitats naturais, restauração de ecossistemas degradados, biodiversidade em áreas urbanas, proteção de áreas de importância ecológica, redução da perda de solo, combate à caça ilegal, conservação de áreas de importância cultural, monitoramento de espécies ameaçadas, educação ambiental para a vida terrestre, uso sustentável da terra, proteção contra invasões biológicas, incentivo à agricultura sustentável, medidas para a preservação de fauna e flora, desenvolvimento de tecnologias para a conservação.",
    "Estado de direito, justiça social, paz duradoura, direitos humanos, acesso à justiça, combate à corrupção, participação cidadã, igualdade perante a lei, instituições transparentes, redução da violência, proteção de vítimas de crimes, combate ao tráfico humano, promoção da não discriminação, resolução pacífica de conflitos, construção de capacidades institucionais, desenvolvimento de sistemas judiciais eficazes, promoção da verdade e reconciliação, combate à impunidade, cooperação internacional em questões de justiça, educação para a paz.",
    "Cooperação internacional, parcerias público privada, desenvolvimento sustentável, engajamento global, financiamento para o desenvolvimento, compartilhamento de conhecimento, colaboração entre setores, inovação social, tecnologias para o desenvolvimento, capacitação de comunidades locais, transferência de tecnologia, desenvolvimento de capacidades institucionais, mobilização de recursos, desenvolvimento de parcerias multissetoriais, advocacia para os ODS, promoção do voluntariado, cooperação Sul-Sul, monitoramento e avaliação conjunta, participação da sociedade civil, cooperação triangular (países desenvolvidos, em desenvolvimento e instituições internacionais).",
]

palavras_indesejadas = [
    "a", "e", "o", "que", "de", "do", "da", "em", "um", "para", "com", "não", "uma", "os", "no", "se", ",", "projeto",
    "ação",
    "na", "por", "mais", "as", "dos", "como", "mas", "ao", "ele", "das", "à", "seu", "sua", "ou", "ser",
    "quando", "muito", "há", "nos", "já", "está", "eu", "também", "só", "pelo", "pela", "até", "isso",
    "ela", "entre", "depois", "sem", "mesmo", "aos", "seus", "quem", "nas", "me", "esse", "eles", "você",
    "essa", "num", "nem", "suas", "meu", "às", "minha", "têm", "numa", "pelos", "elas", "havia", "seja",
    "qual", "será", "nós", "tenho", "lhe", "deles", "essas", "esses", "pelas", "este", "fosse", "dela",
    "tu", "te", "vocês", "vos", "lhes", "meus", "minhas", "teu", "tua", "teus", "tuas", "nosso", "nossa",
    "nossos", "nossas", "dela", "delas", "esta", "estes", "estas", "aquele", "aquela", "aqueles", "aquelas",
    "isto", "aquilo", "estou", "está", "estamos", "estão", "estive", "esteve", "estivemos", "estiveram",
    "estava", "estávamos", "estavam", "estivera", "estivéramos", "esteja", "estejamos", "estejam", "estivesse",
    "estivéssemos", "estivessem", "estiver", "estivermos", "estiverem", "hei", "há", "havemos", "hão", "houve",
    "houvemos", "houveram", "houvera", "houvéramos", "haja", "hajamos", "hajam", "houvesse", "houvéssemos",
    "houvessem", "houver", "houvermos", "houverem", "houverei", "houverá", "houveremos", "houverão", "houveria",
    "houveríamos", "houveriam", "sou", "somos", "são", "era", "éramos", "eram", "fui", "foi", "fomos", "foram",
    "fora", "fôramos", "seja", "sejamos", "sejam", "fosse", "fôssemos", "fossem", "for", "formos", "forem",
    "serei", "será", "seremos", "serão", "seria", "seríamos", "seriam", "tenho", "tem", "temos", "tém", "tinha",
    "tínhamos", "tinham", "tive", "teve", "tivemos", "tiveram", "tivera", "tivéramos", "tenha", "tenhamos",
    "tenham", "tivesse", "tivéssemos", "tivessem", "tiver", "tivermos", "tiverem", "terei", "terá", "teremos",
    "terão", "teria", "teríamos", "teriam",
]

def remover_palavras_indesejadas(texto, palavras_indesejadas):
    for palavra in palavras_indesejadas:
        texto = texto.replace(palavra, "")
    return texto

def obter_categoria_ods(ods):
    return ods_categoria.get(ods, "Categoria não definida")

def calcular_similaridade(frase, ods_texts, palavras_indesejadas):
    tfidf_vectorizer = TfidfVectorizer()

    frase_processada = remover_palavras_indesejadas(frase, palavras_indesejadas)
    ods_texts_processados = [remover_palavras_indesejadas(texto, palavras_indesejadas) for texto in ods_texts]

    ods_vectors = tfidf_vectorizer.fit_transform(ods_texts_processados + [frase_processada])
    similarity_matrix = cosine_similarity(ods_vectors[-1], ods_vectors[:-1])

    return similarity_matrix.flatten()

def encontrar_todas_ods_similares(frase, ods_texts, palavras_indesejadas):
    similaridades = calcular_similaridade(frase, ods_texts, palavras_indesejadas)
    resultados = [(i + 1, similaridade) for i, similaridade in enumerate(similaridades) if similaridade > 0.05]
    return resultados

st.title("Análise de Similaridade com Objetivos de Desenvolvimento Sustentável (ODS)")

frase_usuario = st.text_area("Digite a descrição OU objetivo da sua iniciativa:", height=150)

if st.button("Analisar Similaridade"):
    if frase_usuario:
        resultados = encontrar_todas_ods_similares(frase_usuario, ods_texts, palavras_indesejadas)

        if resultados:
            for ods, similaridade in resultados:
                categoria_ods = obter_categoria_ods(ods)
                st.write(f"A iniciativa é similar à ODS {ods} - {categoria_ods} com uma similaridade de {similaridade:.2%}.")
        else:
            st.write("Não há similaridade com nenhuma ODS.")
    else:
        st.warning("Por favor, insira uma descrição para a análise.")
        
st.sidebar.markdown("---")
st.sidebar.markdown("Desenvolvido por [PedroFS](https://linktr.ee/Pedrofsf)")
