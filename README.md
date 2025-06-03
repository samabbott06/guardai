# General Project Information
## Group Members
- Emery Lear - leari@wwu.edu
- Nick Houlding - houldin@wwu.edu
- Rob Jesionowski - jesionr@wwu.edu
- Sam Abbott - abbotts9@wwu.edu

## Resources In Use
- Cyber Range GitLab Repository
- NLTK (Natural Language Toolkit)
- PyTorch Deep Learning Framework
- BERT Language Model
- Docker Containerization

## Project Outcome
Identify, analyze, and propose mitigation strategies for the various cyber attack surfaces associated with Artificial Intelligence (AI) systems. This includes understanding how AI systems can be exploited and developing robust defenses to protect against such vulnerabilities.

# Project Background & Motivations
## Previous Projects
This project is the first of several, and serves as the foundation for continuation projects by future student groups at WWU for the NUWC.

## Project Focus: Prompt Injection
- Motivating factors
    - Develop methods to identify and classify potential AI prompt injection attacks
    - Create a framework for generating and testing adversarial prompts
    - Build tools for analyzing AI system vulnerabilities systematically
- Goals of the project
    - Create a scalable text classification system for identifying malicious prompts
    - Implement a GAN-based system for generating test cases
    - Provide a foundation for future research in AI security testing

## Vision Statement
To develop a comprehensive toolkit for analyzing, identifying, and testing AI system vulnerabilities through automated classification and adversarial prompt generation.

# Deliverables & Outcome
## Technology Utilized
- Text Classification System using NLTK and potentially other libraries (see `data/text_classifier.py`)
- BERT-based Text Embedding Pipeline (see `data/masterlist_to_gan_pipeline.py`)
- GAN Implementation using PyTorch (see `api/models.py` for definitions, `models/` for trained checkpoints)
- Data Processing Utilities (see scripts in `data/`)
- Containerized Docker Development Environment (`Dockerfile`, `build_and_run.sh`, `entrypoint.sh`)
- CSV-based Data Management System (see `data/training_sets/`)
- Python API for GAN models (`api/GanAPI.py`, `api/setup.py`)
- Example Test Applications (see `test_env/`)

## Major Features
1.  **Text Classification System (`data/text_classifier.py`)**
    *   Processes and classifies text data based on predefined categories.
    *   Outputs classified data into structured directories.
2.  **Data Processing Pipeline (`data/masterlist_to_gan_pipeline.py`, `data/data_processing.py`)**
    *   Generates BERT embeddings for text data.
    *   Manages data chunking and storage of embeddings (`data/embeddings/`).
    *   Prepares data loaders for model training (`data/train_loaders/`, `data/test_loaders/`).
3.  **GAN Implementation**
    *   Defines Generator and Discriminator neural networks (likely in `api/models.py` or related files).
    *   Training script (`train.sh`) to train models.
    *   Stores trained model checkpoints (`models/`).
4.  **GAN API (`api/`)**
    *   Provides a Python API (`api/GanAPI.py`) to interact with the trained GAN models.
    *   Includes setup script (`api/setup.py`) for potential packaging.
5.  **Test Environment (`test_env/`)**
    *   Contains various example applications (e.g., `document_summarizer_app.py`, `email_manager_app.py`) likely used for testing or demonstrating the capabilities of the generated data or models.
6.  **Containerized Environment**
    *   `Dockerfile` and associated scripts (`build_and_run.sh`, `entrypoint.sh`, `install_nvidia_docker.sh`) for building and running the project in a Docker container.

## Project Architecture
The system consists of several key directories and scripts:

1.  **`api/`**: Contains the source code for the GAN API, including the main API class (`GanAPI.py`), model definitions (likely Pydantic models in `models.py`), utility functions (`api_utils.py`), and packaging setup (`setup.py`).
2.  **`data/`**: Houses scripts for data processing (`data_processing.py`, `masterlist_to_gan_pipeline.py`, `segment_csv.py`) and text classification (`text_classifier.py`). It also stores raw datasets (`training_sets/`), generated embeddings (`embeddings/`), and data loaders (`train_loaders/`, `test_loaders/`).
3.  **`models/`**: Stores the trained PyTorch GAN model checkpoints (`.pt` files) categorized by attack type.
4.  **`test_env/`**: Includes example Python applications that might utilize the GANs or demonstrate potential use cases/testing scenarios.
5.  **Root Directory**: Contains core scripts for managing the environment and execution:
    *   `Dockerfile`, `entrypoint.sh`: Define the Docker container environment.
    *   `build_and_run.sh`: Script to build the Docker image and run the container.
    *   `train.sh`: Script to initiate the GAN model training process.
    *   `install_nvidia_docker.sh`: Utility script for setting up Nvidia Docker.
    *   `requirements.txt`: Lists Python package dependencies.
    *   `README.md`: This file.

## Project Achievements
- Implemented a scalable text classification system
- Created a robust data processing pipeline
- Developed a working GAN-based generation system
- Built a containerized development environment
- Established a foundation for future AI cybersecurity research

## Areas for Future Work
1. Classification System Improvements
   - Additional classification categories
   - Enhanced feature extraction
   - Alternative classification algorithms

2. GAN Enhancements
   - Mapping GAN embedding generations to plaintext
   - Improved data generation quality through training fine-tuning
   - Advanced architecture exploration
   - Improved training stability
   - Output quality metrics

3. System Integration
   - Web-based interface
   - Real-time processing capabilities
   - API extension

4. Testing and Validation
   - Comprehensive test suite
   - Performance benchmarking
   - Security validation framework

5. Synthetic LLM-Integrated Applications (Continued)
   - Email summarization exploitation
   - Chatbot vulnerability testing

## Helpful Notes
   - Run `docker system prune -a` periodically to remove unused containers and images to prevent disk space issues.
   - We encountered some issues pulling large files through the VPN, so in the case of an error similar to the following:

        > error: RPC failed; curl 18 transfer closed with outstanding read data remaining
        >
        > error: 47 bytes of body are still expected
        >
        > fetch-pack: unexpected disconnect while reading sideband packet
        >
        > fatal: protocol error: bad pack header

        One can pull blob-wise via:

        > git config remote.origin.promisor true
        >
        > git config remote.origin.partialCloneFilter "blob:none"
        >
        > git fetch --filter=blob:none --deepen 1 (at the problem commit)
        >
        > git fetch --filter=blob:none --unshallow

        This is not a catch-all solution. YMMV.
