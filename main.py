# NLTK
import nltk
# Corpus
nltk.download('machado')
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.corpus import machado, stopwords
# Tokenizer
from nltk.tokenize import word_tokenize, sent_tokenize
# Stemming
from nltk.stem import SnowballStemmer
# Part-of-Speech Tagging
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# SpaCy
import spacy

if __name__ == "__main__":

    print('\n#### NLTK\n')
    print("\nApresentação do copus e das stopwords\n")
    print(f"Corpora machado (amostra): {machado.fileids()}")
    print(f"Exemplo de corpus (contos/macn001.txt): {machado.raw('contos/macn001.txt')}")
    print(f"Número de stopwords: {len(stopwords.words('portuguese'))}")
    print(f"Lista de stopwords (amostra): {stopwords.words('portuguese')[:5]}")

    print("\nApresentação dos tokenizadores\n")
    texto = 'O aprendizado automático (português brasileiro) ou a aprendizagem automática (português europeu) ou também aprendizado de máquina (português brasileiro) ou aprendizagem de máquina (português europeu) (em inglês: machine learning) é um subcampo da Engenharia e da ciência da computação que evoluiu do estudo de reconhecimento de padrões e da teoria do aprendizado computacional em inteligência artificial. Em 1959, Arthur Samuel definiu aprendizado de máquina como o "campo de estudo que dá aos computadores a habilidade de aprender sem serem explicitamente programados" (livre tradução). O aprendizado automático explora o estudo e construção de algoritmos que podem aprender de seus erros e fazer previsões sobre dados. Tais algoritmos operam construindo um modelo a partir de inputs amostrais a fim de fazer previsões ou decisões guiadas pelos dados ao invés de simplesmente seguindo inflexíveis e estáticas instruções programadas. Enquanto que na inteligência artificial existem dois tipos de raciocínio (o indutivo, que extrai regras e padrões de grandes conjuntos de dados, e o dedutivo), o aprendizado de máquina só se preocupa com o indutivo.'
    print(f"Tokenizando sentenças como unidades: {sent_tokenize(texto)}")
    print(f"Tokenizaando palavras como unidades: {word_tokenize(texto)}")

    print('\nApresentação da Lematização (Stemming)\n')

    print(f"Línguas diponíveis no SnowballStemmer: {SnowballStemmer.languages}")
    snowballStemmer = SnowballStemmer('portuguese') # objeto lematizador para o português
    print(f"Exemplo de lematização ('computação'): {snowballStemmer.stem('computação')}")
    print(f"Exemplo de lematização ('computador'): {snowballStemmer.stem('computador')}")
    print(f"Exemplo de lematização ('computando'): {snowballStemmer.stem('computando')}")

    print('\nApresentação da Part-of-Speech Tagging (POS Tagging)\n')

    texto2 = 'Dentro da teoria das probabilidades, um processo estocástico é uma família de variáveis aleatórias representando a evolução de um sistema de valores com o tempo. É a contraparte probabilística de um processo determinístico. Ao invés de um processo que possui um único modo de evoluir, como nas soluções de equações diferenciais ordinárias, por exemplo, em um processo estocástico há uma indeterminação: mesmo que se conheça a condição inicial, existem várias, por vezes infinitas, direções nas quais o processo pode evoluir.'
    print(f"{nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(texto2)))}")

    print('\n#### SPACY\n')
    nlp = spacy.load("pt_core_news_sm")
    doc = nlp(texto)
    print(f"\nTokenizando palavras como unidades:\n")
    for token in doc:
        print(token.text)
    print(f"\nTokenizando sentenças como unidades:\n")
    for sent in doc.sents:
        print(sent.text)

    print(f"\nIdentificando as stopwords:\n")
    for token in doc:
        print(f'{token.text} - {token.is_stop}')
    print(f"\nIdentificando os alfanuméricos:\n")
    for token in doc:
        print(f'{token.text} - {token.is_alpha}')
