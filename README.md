## Prerequisites
Before you can run the application, you need to have the following installed on your system:
- Git
- Anaconda

## Step-by-Step Guide

### 1. Clone the Repository
First, you need to clone the repository to your local machine. Open your terminal or command prompt and run the following command:

```sh
git clone https://github.com/nicolas-becker/climate-policy-miner.git
```

### 2. Install Anaconda
1. Go to the [Anaconda Distribution page](https://www.anaconda.com/products/distribution).
2. Download the installer for your operating system (Windows, macOS, or Linux).
3. Run the installer and follow the on-screen instructions.

### 3. Set Up the Environment
Navigate to the root directory of the cloned repository:

```sh
cd your-repository
```

Create a new conda environment using the environment.yml file included in the repository:

```sh
conda env create -f environment.yml
```

### 4. Activate the Environment
Activate the newly created conda environment by running:

```sh
conda activate your-environment-name
```

### 5. Run the Application
With the environment activated, you can now run the application by executing the pipeline.py script:

```sh
python -i src/pipeline.py
```