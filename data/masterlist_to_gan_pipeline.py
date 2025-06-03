import concurrent.futures
import multiprocessing
import pandas as pd
import numpy as np
import threading
import signal
import torch
import glob
import time
import sys
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
from sklearn.decomposition import PCA
from tqdm import tqdm

# Set to 'spawn' for CUDA compatibility.
# This MUST happen before any multiprocessing.
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

# GLOBALS
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32 if torch.cuda.is_available() else 16
NUM_WORKERS = os.cpu_count() or 4
USE_CASES = ["info_gathering", "intrusion", "malware", "manipulated_content"]

# Track active threads/processes for graceful shutdown.
shutdown_flag = threading.Event()
active_executors = []

def signal_handler(sig, frame):
    """
    Handle keyboard interrupts by setting the shutdown flag and exiting gracefully
    """
    print("\n\nInterrupt received, shutting down gracefully...")
    shutdown_flag.set()
    
    # Shutdown all active executors.
    for executor in active_executors:
        if executor is not None and not executor._shutdown:
            executor.shutdown(wait=False, cancel_futures=True)
    
    print("All threads/processes have been terminated. Exiting...")
    sys.exit(0)

# Register the signal handler for keyboard interrupts:
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset from CSV and inspect its structure.

    Args:
        file_path (str): The file path to the dataset CSV.

    Returns:
        pd.DataFrame: The dataset as a pandas DataFrame.
    
    Raises:
        FileNotFoundError: If the file path does not exist.
    """
    df = pd.read_csv(file_path, nrows=128)

    print("Dataset Head:\n", df.head())
    print("\nDataset Info:\n", df.info())
    print("\nMissing Values:\n", df.isnull().sum())

    return df

def get_bert_embeddings(text_list: list, 
                        chunk_index: int,
                        use_case: str
                        ) -> np.array:
    """
    Get BERT embeddings for a list of text prompts, utilizing GPU when available.

    Args:
        text_list (list): A list of text prompts.
        chunk_index (int): The index of the current chunk.
        use_case (str): The use case for organizing embeddings.
    
    Returns:
        np.array: BERT embeddings for each prompt.
    
    Raises:
        FileNotFoundError: If the BERT embeddings cache file does not exist.
    """
    # Create use case directory if it doesn't exist
    embed_dir = os.path.join("data/embeddings", use_case)
    os.makedirs(embed_dir, exist_ok=True)
    
    cache_file = os.path.join(embed_dir, f"bert_embeddings_chunk_{chunk_index}.npy")
    
    if os.path.exists(cache_file):
        print(f"Loading cached BERT embeddings for {use_case}...")
        return np.load(cache_file)
    
    # Check if shutdown signal has been received
    if shutdown_flag.is_set():
        raise KeyboardInterrupt("Process interrupted by user")
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model = model.to(DEVICE)  # Move model to GPU if available
    model.eval()  # Set to evaluation mode for inference
    
    embeddings = []
    batch_size = 32
    
    # Track progress:
    for i in tqdm(range(0, len(text_list), batch_size), 
                  desc=f"BERT Processing Chunk {chunk_index}"):
        # Check if shutdown signal has been received:
        if shutdown_flag.is_set():
            raise KeyboardInterrupt("Process interrupted by user")
            
        batch_texts = text_list[i:i + batch_size]
        
        tokens = tokenizer(batch_texts, 
                           return_tensors="pt", 
                           padding=True, 
                           truncation=True,
                           max_length=512)
        
        tokens = {key: val.to(DEVICE) for key, val in tokens.items()}
        
        with torch.no_grad():
            output = model(**tokens)
        
        # Move embeddings back to CPU for numpy conversion:
        batch_embeddings = output.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.append(batch_embeddings)
    
    # Concatenate all batch embeddings:
    embeddings = np.vstack(embeddings)
    
    # Only save if not shutting down:
    if not shutdown_flag.is_set():
        # Save embeddings in separate thread to avoid blocking:
        threading.Thread(
            target=lambda: np.save(cache_file, embeddings)
        ).start()

        print(f"BERT Embeddings were cached for future use in {embed_dir}!")
    
    return embeddings

def ensure_formatting(embeddings: np.array) -> np.array:
    """
    Ensure that the embeddings are in 8-dimensional format.

    Args:
        embeddings (np.array): The original high-dimensional embeddings.
    
    Returns:
        np.array: The 8-dimensional feature vectors.
    
    Raises:
        ValueError: If the embeddings are not in the correct format.
    """
    _, num_features = embeddings.shape

    if num_features == 8:
        print("Data already in 8d, skipping PCA.")
        return embeddings

    if num_features < 8:
        print(f"Padding Dataset to 8d with zeros")

        pad_width = ((0,0), (0, 8 - num_features))
        embeddings = np.pad(embeddings, 
                            pad_width, 
                            mode='constant', 
                            constant_values=0)

    if num_features > 8:
        print(f"Reducing Dimensionality to 8d using PCA...")

        pca = PCA(n_components=8)
        embeddings = pca.fit_transform(embeddings)

    return embeddings

def split(features: np.array, 
          labels: np.array
          ) -> tuple:
    """
    Split the features and labels into training and testing sets.

    Args:
        features (np.array): Input feature vectors.
        labels (np.array): Corresponding labels.
    
    Returns:
        tuple: Train/test splits of the features and labels
    
    Raises:
        ValueError: If the features and labels have different lengths.
    """
    return train_test_split(features, 
                            labels, 
                            test_size=0.2, 
                            random_state=42)

def create_dataloader(train_data: tuple, 
                      test_data: tuple, 
                      batch_size=BATCH_SIZE
                      ) -> tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoader objects for training and testing data.

    Args:
        train_data (tuple): Training features and labels.
        test_data (tuple): Testing features and labels.
        batch_size (int): Batch size for the DataLoader.
    
    Returns:
        tuple: PyTorch DataLoader objects for training and testing data.
    
    Raises:
        ValueError: If the input data is not in the correct format.
    """
    train_features, train_labels = train_data
    test_features, test_labels = test_data

    train_tensor = torch.tensor(train_features, 
                                dtype=torch.float32)
    test_tensor = torch.tensor(test_features, 
                               dtype=torch.float32)

    train_labels_tensor = torch.tensor(train_labels, 
                                       dtype=torch.float32
                                       ).unsqueeze(1)
    test_labels_tensor = torch.tensor(test_labels, 
                                      dtype=torch.float32
                                      ).unsqueeze(1)

    train_dataset = TensorDataset(train_tensor, 
                                  train_labels_tensor)
    test_dataset = TensorDataset(test_tensor, 
                                 test_labels_tensor)

    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True,
                              num_workers=NUM_WORKERS,
                              pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, 
                             batch_size=batch_size, 
                             shuffle=False,
                             num_workers=NUM_WORKERS,
                             pin_memory=torch.cuda.is_available())

    return train_loader, test_loader

def process_chunk(chunk_data: tuple) -> tuple:
    """
    Process a single data chunk in parallel.
    
    Args:
        chunk_data (tuple): Contains chunk, chunk_index, use_case
        
    Returns:
        tuple: Processed features and labels
    """
    # Check if shutdown signal has been received:
    if shutdown_flag.is_set():
        return None
        
    chunk, chunk_index, use_case = chunk_data
    
    try:
        # Filter out invalid rows:
        valid_texts = []
        valid_labels = []
        for text, label in zip(chunk["text"], chunk["label"]):
            if isinstance(text, str):
                text = text.strip()
                if text and text.lower() not in ["n/a", "null", "none"]:
                    valid_texts.append(text)
                    valid_labels.append(label)
        
        rows_removed = len(chunk) - len(valid_texts)
        
        if rows_removed > 0:
            print(f"Skipped {rows_removed} invalid samples in chunk {chunk_index}.")

        # Get embeddings and process features:
        text_embeddings = get_bert_embeddings(valid_texts, 
                                              chunk_index,
                                              use_case)
        features = ensure_formatting(text_embeddings)
        labels = np.array(valid_labels)
        
        # Split into train/test sets:
        train_features, test_features, train_labels, test_labels = split(features, labels)
        
        # Create intermediate dataloaders:
        train_dir = os.path.join("data/train_loaders", use_case)
        test_dir = os.path.join("data/test_loaders", use_case)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        chunk_train_loader, chunk_test_loader = create_dataloader(
            (train_features, train_labels), 
            (test_features, test_labels)
        )
        
        # Save intermediate loaders asynchronously only if not shutting down
        if not shutdown_flag.is_set():
            chunk_train_loader_path = os.path.join(train_dir, f"chunk_{chunk_index}.pt")
            chunk_test_loader_path = os.path.join(test_dir, f"chunk_{chunk_index}.pt")
            
            def save_loaders():
                try:
                    torch.save(chunk_train_loader, 
                               chunk_train_loader_path)
                    torch.save(chunk_test_loader, 
                               chunk_test_loader_path)
                except Exception as e:
                    if not shutdown_flag.is_set():
                        print(f"Error saving loaders: {e}")
            
            save_thread = threading.Thread(target=save_loaders)
            save_thread.daemon = True
            save_thread.start()
        
        return (train_features, 
                test_features, 
                train_labels, 
                test_labels)
    
    except KeyboardInterrupt:
        print(f"Chunk {chunk_index} processing interrupted")
        return None
    except Exception as e:
        print(f"Error processing chunk {chunk_index}: {e}")
        return None

def main(csv_path: str, 
         use_case: str,
         chunk_size=5000,
         max_workers=None
         ) -> tuple[DataLoader, DataLoader]:
    """
    Main function to process the dataset and create DataLoader objects.

    Args:
        csv_path (str): The file path to the dataset CSV.
        use_case (str): The use case for organizing outputs.
        chunk_size (int): The number of rows to process at a time.
        max_workers (int): Maximum number of workers for parallel processing.
    
    Returns:
        tuple: PyTorch DataLoader objects for training and testing data.

    Raises:
        FileNotFoundError: If the dataset CSV file does not exist.
    """
    start_time = time.time()
    print(f"Processing dataset for {use_case} in chunks using {DEVICE} device...")
    print(f"Using {NUM_WORKERS} worker threads for data loading")

    os.makedirs("data", exist_ok=True)
    os.makedirs("data/embeddings", exist_ok=True)
    train_dir = os.path.join("data/train_loaders", use_case)
    test_dir = os.path.join("data/test_loaders", use_case)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    try:
        max_rows = sum(1 for _ in open(csv_path)) - 1
        total_rows_processed = 0
        
        df_reader = pd.read_csv(csv_path, chunksize=chunk_size)        
        all_results = []
        
        if max_workers is None:
            max_workers = min(NUM_WORKERS, 4)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            active_executors.append(executor)
            
            chunk_args = []
            for chunk_index, chunk in enumerate(df_reader):
                if total_rows_processed >= max_rows or shutdown_flag.is_set():
                    break
                
                chunk_args.append((chunk, chunk_index, use_case))
                total_rows_processed += len(chunk)
            
            futures = [executor.submit(process_chunk, args) 
                       for args in chunk_args]
            
            for future in tqdm(concurrent.futures.as_completed(futures), 
                              total=len(futures), 
                              desc=f"Processing {use_case} chunks"):
                
                if shutdown_flag.is_set():
                    break
                    
                try:
                    result = future.result()
                    if result is not None:
                        all_results.append(result)
                
                except Exception as e:
                    print(f"Error processing chunk: {e}")
            
            active_executors.remove(executor)
        
        if shutdown_flag.is_set():
            print(f"Processing of {use_case} interrupted by user")
            return None, None
        
        print("\nConsolidating all chunks into a single dataloader...")
        
        if all_results:
            # Unpack and consolidate all results:
            all_train_features = [r[0] for r in all_results]
            all_test_features = [r[1] for r in all_results]
            all_train_labels = [r[2] for r in all_results]
            all_test_labels = [r[3] for r in all_results]
            
            consolidated_train_features = np.concatenate(all_train_features, axis=0)
            consolidated_test_features = np.concatenate(all_test_features, axis=0)
            consolidated_train_labels = np.concatenate(all_train_labels, axis=0)
            consolidated_test_labels = np.concatenate(all_test_labels, axis=0)
            
            # Create consolidated dataloaders with optimized batch size
            final_batch_size = BATCH_SIZE * 2  # Larger batch size for consolidated dataset
            final_train_loader, final_test_loader = create_dataloader(
                (consolidated_train_features, consolidated_train_labels), 
                (consolidated_test_features, consolidated_test_labels),
                batch_size=final_batch_size
            )
            
            consolidated_train_loader_path = os.path.join(train_dir, f"{use_case}_train.pt")
            consolidated_test_loader_path = os.path.join(test_dir, f"{use_case}_test.pt")
            torch.save(final_train_loader, consolidated_train_loader_path)
            torch.save(final_test_loader, consolidated_test_loader_path)
            
            elapsed_time = time.time() - start_time
            print(f"Consolidated dataloaders saved for {use_case}!")
            print(f"Total processing time: {elapsed_time:.2f} seconds")
            
            return final_train_loader, final_test_loader
        else:
            print("No data processed!")
            return None, None
            
    except KeyboardInterrupt:
        print(f"\nProcessing of {use_case} interrupted by user")
        return None, None

# Entry point
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    print(f"Available CPU cores: {NUM_WORKERS}")
    print("Press Ctrl+C at any time to gracefully stop processing")
    
    for use_case in USE_CASES:
        os.makedirs(os.path.join("data/embeddings", use_case), 
                    exist_ok=True)
        os.makedirs(os.path.join("data/train_loaders", use_case), 
                    exist_ok=True)
        os.makedirs(os.path.join("data/test_loaders", use_case), 
                    exist_ok=True)
    
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            active_executors.append(executor)
            """
            We use max_workers=1 for the process pool because 
            each process will use multiple threads internally, 
            and BERT/GPU operations are already parallelized.
            """
            futures = {}
            
            for use_case in USE_CASES:
                if shutdown_flag.is_set():
                    break
                    
                # Look for CSV files in training_sets/classified directory
                csv_files = glob.glob(os.path.join("training_sets/classified", f"{use_case}.csv"))
                
                if csv_files:
                    print(f"\n==> Processing {use_case} dataset...")
                    csv_file = csv_files[0]
                    futures[executor.submit(main, csv_file, use_case)] = use_case
                else:
                    print(f"No CSV file found for use case: {use_case}")
            
            # Wait for all futures to complete:
            for future in concurrent.futures.as_completed(futures):
                if shutdown_flag.is_set():
                    break
                    
                use_case = futures[future]
                try:
                    future.result()
                    print(f"Completed processing {use_case} dataset!")
                
                except Exception as e:
                    if not isinstance(e, KeyboardInterrupt):
                        print(f"Error processing {use_case}: {e}")
            
            active_executors.remove(executor)
            
        if not shutdown_flag.is_set():
            print("\nAll use cases processed successfully!")
        else:
            print("\nProcessing interrupted by user. Some datasets may be incomplete.")
            
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        shutdown_flag.set()