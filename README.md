# RAGS
# Chatbot with Multiple Outputs

This repository contains a chatbot that generates responses based on various similarity measures. The chatbot utilizes Euclidean distance, cosine similarity, sentence transformer similarity, and a MultiQueryRetriever method to provide diverse and comprehensive responses to user queries.

## Features

- Generates responses based on Euclidean distance.
- Generates responses based on cosine similarity.
- Generates responses based on sentence transformer similarity.
- MultiQueryRetriever method to generate multiple query variations and responses.
  

## Setup and Installation

### Requirements

- Python 3.7 or later
- CUDA-compatible GPU (optional but recommended for faster processing)

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/AiHunter27/Rags.git
    cd Rags
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Download the pre-trained models and place them in the specified directories:

    - Hugging Face Embedding Model: `/content/drive/MyDrive/HuggingFaceModels/`
    - Hugging Face LLM Model: `/content/drive/MyDrive/HuggingFaceLLM/`

4. Run the script to launch the Gradio interface:

    ```bash
    python app.py
    ```


### Files

- `app.py`: Main application script.
- `README.md`: This readme file.
- `requirements.txt`: List of required packages.



### Acknowledgements

- Hugging Face for providing pre-trained models.
- Gradio for the interactive interface.


### Contact

For any questions or inquiries, please contact @Aswath at https://www.linkedin.com/in/sai-aswath-993b61a9/.

### support
https://buymeacoffee.com/aswath
