import threading
import random
import nltk
import csv
import os
from sklearn.metrics import classification_report, confusion_matrix
from nltk.classify.naivebayes import NaiveBayesClassifier
from concurrent.futures import ThreadPoolExecutor
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from typing import Generator

# Gets nltk data files if needed.
try:
    nltk.data.find('corpora/punkt')
    nltk.data.find('corpora/punkt')
    nltk.data.find('corpora/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')

# GLOBALS
# These are used in the tokenize_row() function, but they need to be initializes here or it breaks:
stop_words = set(stopwords.words("english"))
stop_chars = set(["'", ".", "?", ","])

class FileLockManager:
    """
    File locks, prevent multiple threads from writing to same file simultaneously, avoids race conditions
    """
    def __init__(self):
        self._locks = {}
        self._manager_lock = threading.Lock()  # Protects access to _locks

    def get_lock(self, filepath: str):
        """
        Gets or creates a lock for a specific file path to prevent concurrent writes.
        
        Args:
            filepath (string): Path to the file that needs a lock
        
        Returns:
            threading.Lock: A lock object specific to the requested file
        """ 
        with self._manager_lock:
            if filepath not in self._locks:
                self._locks[filepath] = threading.Lock()
            return self._locks[filepath]

def chunk_csv(file_path: str, 
              chunk_size=10000
              ) -> Generator:
    """
    Chunks CSV, chunk size can be played around with
    
    Args:
        file_path (string): Path to the CSV file to be read
        chunk_size (int): Maximum number of rows to include in each chunk
    
    Yields:
        list: Chunks of CSV rows as dictionaries, each chunk containing up to chunk_size rows
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        chunk = []
        for row in reader:
            chunk.append(row)
            if len(chunk) >= chunk_size:
                yield chunk                 # Once chunk_size reached, yield the current batch of rows
                chunk = []
        if chunk:
            yield chunk                     # last partial chunk if it exists

def tokenize_row(row, stops):
    """
    Tokenizes a single row. 
    We may want to add a tokenize_chunk function as well.
    
    Args:
        row (dict): Dictionary containing 'text' and 'label' fields
        stops: boolean switch, if True then stopchars and stopwords will be removed, if false then row will be tokenized as is
    
    Returns:
        tuple: Pair containing tokenized text in format ({"words": [tokens]}, label)
    """
    def remove_stop_chars(tokens):
        sanitized_tokens = list()

        for token in tokens:
            #if token not in stop_words and token.isalnum():            # filters out any non alpha numeric characters
            if token not in stop_words and token not in stop_chars:     # only filters out non alpha numeric characters in the stop_chars list
                sanitized_tokens.append(token)

        return sanitized_tokens

    text = row['text']
    tokens = word_tokenize(text.lower())

    if stops == True:
        row['words'] = remove_stop_chars(tokens)
    elif stops == False:
        row['words'] = tokens

    return ({"words": row['words']}, row['label'])

def tokenize_data(file_path, stops, chunk_size=1000):
    """
    Tokenizes all text entries via tokenize_row. Adjust max_workers based on CPU.
    Also could look into using ThreadPoolExecutor, which bypasses GIL which ntlk is beholden to.
    
    Args:
        file_path (string): Path to CSV file containing text data
        stops: boolean switch, if True then stopchars and stopwords will be removed, if false then row will be tokenized as is
        chunk_size (int): Number of rows to process in each concurrent batch
    
    Returns:
        list: Collection of tokenized entries in the format [({"words": [tokens]}, label), ...]
    
    Sample data: list of tuples (dict, string), each dict contains a tokenized text, and the string represents its label
    [
        ({"words": ["the", "quick", "brown", "fox"]}, "positive"),
        ({"words": ["lazy", "old", "dog"]}, "negative"),
        ({"words": ["happy", "bright", "day"]}, "positive"),
        ({"words": ["dark", "gloomy", "night"]}, "negative"),
    ]
    """
    tokenized_data = []
    for chunk in chunk_csv(file_path, chunk_size):
        with ThreadPoolExecutor(max_workers=8) as executor:
            chunk_tokenized = list(executor.map(lambda x: tokenize_row(x, stops), chunk))
            tokenized_data.extend(chunk_tokenized)
    return tokenized_data


'''
Abstract: detokenizes one element(row) of data.
Literal: takes a dictionary, concatonates the values into a string, returns string (keys are not returned)
    args:
        data: dict{"words": [word]...}
    returns:
        detokenized_string: string
    
'''
def detokenize_data(data):
    detokenized_string = ""

    for (key, value) in data.items():
        detokenized_string += " ".join(value)

    return detokenized_string

'''
takes tokenized data as input, and returns word features
    Args:
        data: list[tuple(dict{"words": []}, string)]
    Returns:
        wordlist.keys(): dict_keys(every individual word contained in the tokenized data set)
    
'''
def get_word_features(data):
    all_words = []
    for (text, label) in data:
        all_words.extend(text['words'])
    wordlist = nltk.FreqDist(all_words) 
    return wordlist.keys()

def create_extract_features(word_features):
    """
    Uses closure pattern to bind the word_features to the returned function. Avoids the use of global variables for get_word_features
    
    Args:
        word_features (dict_keys): Collection of all unique words from training corpus
    
    Returns:
        function: Feature extractor that converts tokenized text into feature dictionaries
    """
    def extract_features(text):
        text_words = set(text["words"])
        features = {}
        for word in word_features:
            features[f'contains({word})'] = (word in text_words)
        return features
    return extract_features

'''
Takes a classifier as input and returns the filepath of that classifiers csv file
    Args:
        classifier_string: string
    Returns:
        string (filepath to the dataset corresponding with said classifier)
'''
def get_classifier_filepath(classifier_string):

    os.makedirs('training_sets/classified', exist_ok=True)
    match classifier_string:
        case "malware":
            return "training_sets/classified/malware.csv"
        case "info_gathering":
            return "training_sets/classified/info_gathering.csv"
        case "intrusion":
            return "training_sets/classified/intrusion.csv"
        case "manipulated_content":
            return "training_sets/classified/manipulated_content.csv"
        case _:
            return "Default action"

def process_entry(entry, classifier, extract_features_func, lock_manager):
    """
    Processes single data entry by classifying it and writing to the correct output file.
    Called concurrently from multiple threads.
    
    Args:
        entry (tuple): A (text, value) pair where text contains tokenized words
        classifier (NaiveBayesClassifier): Trained classifier for determining text category
        extract_features_func (function): Function to convert tokenized text to features
        lock_manager (FileLockManager): Manager for thread-safe file access
    """
    text, value = entry
    filepath = get_classifier_filepath(classifier.classify(extract_features_func(text))) # Extract features from text, classify it, and determine which file to write to
    row = [detokenize_data(text), value]                                                 # Convert back to string form and pair with original label
    file_lock = lock_manager.get_lock(filepath)
    with file_lock:
        with open(filepath, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)

'''
Takes tokenized data and a classifier, and writes the data into predefined csv file for their respective categories based off of the classifier
    Args:
        data: list[tuple(dict{"words": []}, string)]
        classifer: label_probdist, feature_probdist
            https://www.nltk.org/_modules/nltk/classify/naivebayes.html
    Returns:
        nothing
'''

def classify_data(data, classifier, extract_features_func, lock_manager):
    """
    Distributes classifying all data entries across multiple threads.
    ThreadPoolExecutor for threading and FileLockManager to avoid race conditions
    
    Args:
        data (list): Collection of (text, label) pairs to be classified
        classifier (NaiveBayesClassifier): Trained classifier
        extract_features_func (function): Function to convert text to feature vectors (pass extract_features in main)
        lock_manager (FileLockManager): Manager for thread-safe file operations
    """
    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(
            lambda entry: process_entry(entry, classifier, extract_features_func, lock_manager),
            data
        )
"""
This function was determined to have no effect on accuracy, 
keeping it for nostalgic purposes

def balance_dataset(data):

    from collections import defaultdict
    class_data = defaultdict(list)
    for item in data:
        class_data[item[1]].append(item)
    # Find the smallest class size among the malicious categories
    min_count = min(len(items) for items in class_data.values())
    balanced = []
    for label, items in class_data.items():
        balanced.extend(random.sample(items, min_count))
    return balanced
"""

def split_dataset(data, test_size=0.2):
    """
    Partitions dataset into test/train for eval purposes. Default of 80/20 split.

    Args:
        data (list): Tokenized training data
        test_size (float, optional): Proportion of dataset to test. Defaults to 0.2.

    Returns:
        list, list: train, test datasets
    """
    random.shuffle(data)
    split_point = int(len(data) * (1 - test_size))
    return data[:split_point], data[split_point:]
        
def evaluate_classifier(classifier, test_data, extract_features_func):
    """_summary_

    Args:
        classifier (NaiveBayesClassifier): Trained classifier
        test_data (list): Tokenized test split
        extract_features_func (function -> dict): function used to extract text features
    """
    test_set = nltk.classify.apply_features(extract_features_func, test_data)
    accuracy = nltk.classify.accuracy(classifier, test_set)
    
    y_true = []
    y_pred = []
    for (text, label) in test_data:
        features = extract_features_func(text)
        prediction = classifier.classify(features)
        y_true.append(label)
        y_pred.append(prediction)
    
    print(f"evaluation accuracy: {accuracy:.4f}")
    print("classification report:")
    print(classification_report(y_true, y_pred))
    print("confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

def augment_training_data(training_data, augmentation_factor=2):
    """
    Augments training data to create prompt variations.

    Techniques used:
    1. Word shuffling - Preserves semantic meaning while changing word order
    2. Synonym replacement - Maintains intent while using different vocabulary
    3. Random word deletion - Simulates incomplete or truncated prompts

    Any number of techniques can be added/deleted to determine isolated effects.
    
    Args:
        training_data (list): Original training data in format [({"words": [tokens]}, label), ...]
        augmentation_factor (int): Target multiplication factor for dataset size
        
    Returns:
        list: Augmented training data
    """
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    
    augmented_data = list(training_data)
    
    class_counts = {}
    for _, label in training_data:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    print(f"original class distribution: {class_counts}")
    
    # Get synonyms for a word
    def get_synonyms(word):
        synonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                syn_word = lemma.name().replace('_', ' ')
                if syn_word != word and syn_word not in synonyms:
                    synonyms.append(syn_word)
        return synonyms[:3]
    
    # Augmentation methods
    for text_dict, label in training_data:
        words = text_dict['words'].copy()
        
        if len(words) < 4:
            continue
        
        # Word shuffling (all entries)
        if len(words) > 3:
            first_word = words[0]
            last_word = words[-1]
            middle_words = words[1:-1]
            random.shuffle(middle_words)
            shuffled_words = [first_word] + middle_words + [last_word]
            augmented_data.append(({"words": shuffled_words}, label))

        # Synonym replacement (some entries)
        if random.random() < 0.7:  # 70% chance to occur
            synonym_words = words.copy()
            # Replace 10-30% of words with synonyms
            num_to_replace = max(1, int(random.uniform(0.1, 0.3) * len(words)))
            replace_indices = random.sample(range(len(words)), min(num_to_replace, len(words)))
            
            for idx in replace_indices:
                synonyms = get_synonyms(words[idx])
                if synonyms:
                    synonym_words[idx] = random.choice(synonyms)
            
            augmented_data.append(({"words": synonym_words}, label))
        
        # Random word deletion (some entries)
        if random.random() < 0.8:  # 80% chance to occur
            deletion_words = words.copy()
            
            # Delete 10-25% of words based on prompt length
            delete_ratio = random.uniform(0.1, min(0.25, 3.0/len(words)))
            num_to_delete = max(1, int(delete_ratio * len(words)))
            
            # Dont delete too many words from short prompts
            if len(deletion_words) - num_to_delete < 3:
                num_to_delete = len(deletion_words) - 3
            
            if num_to_delete > 0:
                delete_indices = random.sample(range(len(deletion_words)), num_to_delete)
                deletion_words = [word for i, word in enumerate(deletion_words) if i not in delete_indices]
                augmented_data.append(({"words": deletion_words}, label))

    new_class_counts = {}
    for _, label in augmented_data:
        new_class_counts[label] = new_class_counts.get(label, 0) + 1
    
    print(f"augmented class distribution: {new_class_counts}")
    
    random.shuffle(augmented_data)
    
    return augmented_data


########
# MAIN #
########
def main():
    directory = os.path.join(os.path.dirname(__file__), "training_sets/master_plus_imoxto")
    for filename in os.scandir(directory):
        with open(os.path.join(directory, filename)) as f:
            print("tokenizing training data")
            training_data = tokenize_data("training_sets/classifier_training_set.csv", True)

            print(f"tokenized {len(training_data)} training entries") # Training data will contain prompt injection text and classifier
            print("tokenizing data")
            data = tokenize_data("training_sets/Master.csv", False) # actual data will contain prompt injection text and a 1 or 0 to represent malicous or benign
            print(f"tokenized {len(data)} entries")


            train_orig, test_data = split_dataset(training_data, test_size=0.2)
            augmented_train = augment_training_data(train_orig, augmentation_factor=2)


            print("extracting word features")
            word_features = get_word_features(augmented_train)
            print(f"found {len(word_features)} unique words") # Needs to be global so the extract features function can properly access it (don't think there is a way around this)
            extract_features = create_extract_features(word_features) # Prepare word features

            print("prepping training data")
            #training_set = [(extract_features(text), label) for (text, label) in train_data]
            training_set = nltk.classify.apply_features(extract_features, training_data) # Prepare training data (this function is kinda of a maze. Read extract_features() comment for more info)
            
            # training_set is a nltk.collections.LazyMap
            print("training classifier...")
            classifier = NaiveBayesClassifier.train(training_set) # Train the Naive Bayes classifier (this is non-deterministic                                                                        
            
            # ie: different outputs each time it's run, even on the same dataset)
            print("classifier trained")

            print(classifier.show_most_informative_features(10))      
            evaluate_classifier(classifier, test_data, extract_features)

            # Show informative features
            lock_manager = FileLockManager()
            classify_data(data, classifier, extract_features, lock_manager) # Sorts data entries into their catagories based on the classifier 
            print("classification complete B-)")


# Entry point
if __name__ == '__main__':
    main()