# Text Improvement Engine

This tool analyzes a given text and suggests improvements based on the similarity to a list of standard phrases. It aims to align the input text closer to predefined standards related to business and communication. The tool uses a pre-trained language model to identify phrases in the input text that are semantically similar to the standard phrases, providing suggestions for replacements.

## Project Structure

* **TextImprovementEngine.py**: Python script for text analysis.

* **sample_phrases.csv**: File containing the standard phrases.

* **sample_text.txt**: File containing the sample text.

* **README.md**: Documentation file explaining setup, technologies used, and results analysis.

* **requirements.txt**: List of Python dependencies. 

## Setup Process

1. Clone the repository:

    ```bash
    git clone https://github.com/TTapa6ola/Text-Improving-Engine.git
    cd text-improvement-engine
    ```

2. Install dependencies:

    ```bash   
    pip install -r requirements.txt
    ```

3. Run the tool:

    ```bash
    python TextImprovementEngine.py
    ```

4. Follow the prompts to input the text you wish to analyze.

## Technologies Used

* Python 3.x
* sentence_transformers for state-of-the-art sentence embeddings and pretrained models
* Cosine similarity for semantic similarity
* Git for version control

## Rationale

This tool uses the SentenceTransformers library and the pre-trained 'all-MiniLM-L6-v2' model to generate semantic embeddings for the input text and standard phrases.

SentenceTransformers provides an easy way to leverage state-of-the-art models for text matching without having to train custom models from scratch. The 'all-MiniLM-L6-v2' model was selected as a compact yet powerful model suitable for embedding short texts.

After encoding the text, cosine similarity is used to efficiently compare the similarity of the input text embeddings versus the standard phrase embeddings.

The text is split into sentences using regular expressions to allow for more granular comparisons, rather than considering the full paragraph text as a whole. This provides improved suggestions by matching on a sentence-level basis.

Overall, this combination of pre-trained embeddings, cosine similarity, and sentence-level matching provides an performant out-of-box solution while avoiding the need to train custom neural networks. The modular design allows for swapping model and similarity metric if needed.

## Results Analysis

The tool successfully identifies phrases in the input text that are similar to the standard phrases. However, some improvements could be made to enhance accuracy. For example, considering the context of the surrounding words and phrases could provide more accurate results. With more time, additional techniques like contextual embeddings and a larger training dataset could be explored for better performance.
