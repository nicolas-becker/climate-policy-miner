# Climate Policy Miner 🌍⛏️

Welcome to **Climate Policy Miner**, a powerful tool for analyzing and extracting insights from climate policy documents. This repository is designed to streamline the processing of complex textual data, making it easier to uncover key information for research, decision-making, and advocacy. 🌱📜

## What is Climate Policy Miner?

Climate Policy Miner leverages cutting-edge natural language processing (NLP) tools and APIs to help you:  
- 🔍 Analyze climate policy documents efficiently.  
- ✨ Highlight relevant text passages.  
- 📊 Generate structured outputs for further analysis.

Whether you're a researcher, policymaker, or climate advocate, this tool is here to simplify your workflow and empower data-driven decisions. 

Let’s get started! 🚀


## Prerequisites
Before you can run the application, you need to have the following installed on your system:

#### Git:
1. Go to the [Git for Windows download page](https://gitforwindows.org/).
2. Click on the "Download" button.
3. Once the installer has downloaded, open it to start the installation process.

#### Anaconda:
1. Go to the [Anaconda Distribution page](https://www.anaconda.com/products/distribution).
2. Download the installer for your operating system (Windows, macOS, or Linux).
3. Run the installer and follow the on-screen instructions.

## Step-by-Step Guide

### 1. Clone the Repository
First, you need to clone the repository to your local machine. Open the "Anaconda Prompt" and run the following command:

```sh
git clone https://github.com/nicolas-becker/climate-policy-miner.git
```

### 2. Set Up the Environment
Navigate to the root directory of the cloned repository:

```sh
cd climate-policy-miner
```

Create a new conda environment using the environment.yml file included in the repository:

```sh
conda env create -f environment.yml
```

### 3. Activate the Environment
Activate the newly created conda environment by running:

```sh
conda activate textmining_venv
```

### 4. Gather API Keys for .env file
Gather the necessary API Keys for

- [Unstructure.io](https://unstructured.io/)
- [Pinecone.io](https://www.pinecone.io/)
- [OpenAI.com](https://www.pinecone.io/) or via Azure
  
and insert them in the specified attributes of the .env file.

### 5. Run the Application
With the environment activated, you can now run the application by executing the pipeline.py script:

```sh
python -i src/pipeline.py
```

### 6. Follow the instructions in the CLI 

1. **File Selection:**
   - When prompted:  
     **"Paste the path or URL to the file you would like to be analyzed"**
   - **Action:**  
     Insert the path to the policy document located in your local repository.  
     Alternatively, you can try pasting a URL to the document, but this option is **unstable** at the moment.

2. **Pre-processing Step:**
   - When prompted:  
     **"Is the file already pre-processed? [y/n]"**
   - **Action:**  
     Type `n` and press `ENTER` if it's your first time analyzing the document.

3. **Troubleshooting:**
   - If the application **stops after the pre-processing step** without displaying results:
     - **Action:**  
       Type `quit()` to exit the CLI.
     - Then, return to **Step 5** and try again.
     - This is a known issue, and in such cases, you may type `y` when prompted if the file was already pre-processed.

4. **Output:**
   - Once the analysis is complete:
     - A folder will be created in your repository, named after your document.
     - Inside this folder, you will find the results in a subfolder called **"output"**:
       - `.csv` and `.xlsx` files containing the retrieved data.
       - A `.pdf` file containing highlighted text passages.

## Demo

https://github.com/user-attachments/assets/3b93bfcd-c9d5-439d-a0f7-328873f9a26f

## API Usage

### 🔧 Unstructured.io API
- **Purpose:** Pre-processes and extracts clean, structured text from raw files (PDFs, Word documents, etc.).  
- **Usage Tip:** Ensure that your documents are in formats supported by the API. Refer to the [Unstructured.io documentation](https://unstructured.io/) for more details. 
- **Key Note:** For large files or complex documents, processing times may vary. This projects applies the Free API. Please refer to [Free Unstructured API](https://docs.unstructured.io/api-reference/api-services/free-api) for futher information on API access and limitations.

### 🤖 ChatGPT (OpenAI) API
- **Purpose:** Performs advanced natural language processing tasks, including summarization and extracting key insights from text.  
- **Usage Tip:** Use this API to tailor analyses to specific questions or objectives. For example, you can extract sections of text related to "emissions targets" or "policy impacts."  
- **Key Note:** Keep track of your token usage when using the OpenAI API, especially for large-scale analyses. Refer to the [OpenAI documentation](https://platform.openai.com/docs/) for managing your API calls effectively. Please refer to [OpenAI Pricing](https://openai.com/api/pricing/) or [Azure OpenAI Service](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/) for pricing information.


