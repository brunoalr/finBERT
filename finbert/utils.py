# The classes used for data processing and convert_examples_to_features are very similar versions of the ones \
# found in Hugging Face's scripts in the transformers library. For more BERT or similar language model implementation \
# examples, we would highly recommend checking that library as well.


from __future__ import absolute_import, division, print_function

import csv
import sys
import os
import torch

import numpy as np
import logging

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

if not DATASETS_AVAILABLE:
    logger.warning("datasets library not available. Hugging Face dataset loading will be disabled.")


# Classes regarding input and data handling

WEIGHTS_NAME = 'pytorch_model.bin'
CONFIG_NAME = 'config.json'

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None, agree=None):
        """
        Constructs an InputExample
        Parameters
        ----------
        guid: str
            Unique id for the examples
        text: str
            Text for the first sequence.
        label: str, optional
            Label for the example.
        agree: str, optional
            For FinBERT , inter-annotator agreement level.
        """
        self.guid = guid
        self.text = text
        self.label = label
        self.agree = agree


class InputFeatures(object):
    """
    A single set of features for the data.
    """

    def __init__(self, input_ids, attention_mask, token_type_ids, label_id, agree=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_id = label_id
        self.agree = agree


class DataProcessor(object):
    """Base class to read data files."""

    @classmethod
    def _read_tsv(cls, input_file):
        """Reads a tab separated value file or CSV file."""
        # Auto-detect delimiter
        with open(input_file, "r", encoding='utf-8') as f:
            first_line = f.readline()
            delimiter = "\t" if first_line.count('\t') > first_line.count(',') else ","
            f.seek(0)

            reader = csv.reader(f, delimiter=delimiter, quotechar='"')
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    try:
                        unicode_type = unicode  # noqa: F821
                    except NameError:
                        unicode_type = str
                    line = [unicode_type(cell, 'utf-8') for cell in line]
                else:
                    line = [str(cell) for cell in line]
                lines.append(line)
        return lines


class FinSentProcessor(DataProcessor):
    """
    Data processor for FinBERT.
    """

    def _is_hf_dataset_id(self, data_dir):
        """
        Check if data_dir is a Hugging Face dataset identifier.
        
        Parameters
        ----------
        data_dir: str or Path
            Path or dataset identifier to check.
            
        Returns
        -------
        bool
            True if data_dir appears to be a HF dataset ID, False otherwise.
        """
        if not DATASETS_AVAILABLE:
            return False
        
        # Convert Path objects to string
        data_dir_str = str(data_dir)
        
        # Check if it contains a slash (format: "org/dataset") and doesn't exist as a local path
        if '/' in data_dir_str and not os.path.exists(data_dir_str) and not os.path.isdir(data_dir_str):
            # Additional check: doesn't look like a relative path
            if not data_dir_str.startswith('.') and not os.path.isabs(data_dir_str):
                return True
        return False

    def _load_hf_dataset(self, dataset_id, phase):
        """
        Load a Hugging Face dataset and convert it to InputExample format.
        
        Parameters
        ----------
        dataset_id: str
            Hugging Face dataset identifier (e.g., "NOSIBLE/financial-sentiment").
        phase: str
            Dataset split name (e.g., "train", "validation", "test").
            
        Returns
        -------
        list
            List of InputExample objects.
        """
        if not DATASETS_AVAILABLE:
            raise ImportError("datasets library is required to load Hugging Face datasets. Install it with: pip install datasets")
        
        # Map phase names to common HF dataset split names
        split_mapping = {
            'train': 'train',
            'validation': 'validation',
            'val': 'validation',
            'test': 'test',
            'eval': 'test'
        }
        
        hf_split = split_mapping.get(phase.lower(), phase)
        
        try:
            # Try to load the dataset with the mapped split name
            try:
                dataset = load_dataset(dataset_id, split=hf_split)
            except ValueError:
                # If that fails, try without specifying split (some datasets have different split names)
                full_dataset = load_dataset(dataset_id)
                # Try to find a matching split
                available_splits = list(full_dataset.keys())
                logger.info(f"Available splits: {available_splits}")
                
                # Try exact match first
                if hf_split in available_splits:
                    dataset = full_dataset[hf_split]
                elif phase.lower() in available_splits:
                    dataset = full_dataset[phase.lower()]
                else:
                    # If validation split is requested but not available, raise a specific error
                    # The caller should handle splitting the training data
                    if hf_split == 'validation' and 'train' in available_splits:
                        validation_error = ValueError(
                            f"Validation split not found in dataset '{dataset_id}'. "
                            f"Available splits: {available_splits}."
                        )
                        # Store a flag so we can identify this specific error
                        validation_error._is_validation_split_error = True
                        raise validation_error
                    # Use the first available split as fallback for other cases
                    logger.warning(f"Split '{hf_split}' not found. Using '{available_splits[0]}' instead.")
                    dataset = full_dataset[available_splits[0]]
            
            # Get column names
            column_names = dataset.column_names
            logger.info(f"Dataset columns: {column_names}")
            
            # Auto-detect text and label columns
            text_col = None
            label_col = None
            
            # Find text column (common names: text, sentence, content, headline)
            text_candidates = ['text', 'sentence', 'content', 'headline', 'news', 'statement']
            for col in column_names:
                if col.lower() in text_candidates:
                    text_col = col
                    break
            
            # Find label column (common names: label, sentiment, target, class)
            label_candidates = ['label', 'sentiment', 'target', 'class', 'sentiment_label']
            for col in column_names:
                if col.lower() in label_candidates:
                    label_col = col
                    break
            
            if text_col is None:
                # If no text column found, use the first non-label column
                text_col = [c for c in column_names if c != label_col][0] if label_col else column_names[0]
                logger.warning(f"Text column not found, using '{text_col}'")
            
            if label_col is None:
                # If no label column found, use the last column or second column
                if len(column_names) > 1:
                    label_col = column_names[-1] if column_names[-1] != text_col else column_names[1]
                    logger.warning(f"Label column not found, using '{label_col}'")
                else:
                    label_col = None
                    logger.warning("No label column found. Labels will be set to None.")
            
            # Convert dataset to InputExample objects
            examples = []
            for i, example in enumerate(dataset):
                guid = f"{phase}-{i+1}"
                text = str(example[text_col]) if text_col else ""
                label = str(example[label_col]) if label_col and label_col in example else None
                agree = None  # HF datasets typically don't have agreement levels
                
                if not text:
                    logger.warning(f"Skipping example {i+1}: empty text")
                    continue
                
                examples.append(InputExample(guid=guid, text=text.strip(), label=label, agree=agree))
            
            logger.info(f"Loaded {len(examples)} examples from Hugging Face dataset '{dataset_id}' (split: {hf_split})")
            return examples
            
        except ValueError as e:
            # Re-raise validation split errors as-is so they can be caught by the caller
            if hasattr(e, '_is_validation_split_error'):
                raise
            # Wrap other ValueErrors
            raise ValueError(f"Failed to load Hugging Face dataset '{dataset_id}': {str(e)}")
        except Exception as e:
            raise ValueError(f"Failed to load Hugging Face dataset '{dataset_id}': {str(e)}")

    def get_examples(self, data_dir, phase):
        """
        Get examples from the data directory or Hugging Face dataset.

        Parameters
        ----------
        data_dir: str
            Path for the data directory or Hugging Face dataset identifier.
        phase: str
            Name of the .csv file to be loaded or dataset split name.
        """
        if self._is_hf_dataset_id(data_dir):
            return self._load_hf_dataset(data_dir, phase)
        else:
            return self._create_examples(self._read_tsv(os.path.join(data_dir, (phase + ".csv"))), phase)

    def get_labels(self):
        return ["positive", "negative", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []

        # Detect format from header
        format_type = None
        if lines:
            header = [str(col).lower() for col in lines[0]]
            if len(header) == 2:
                if header[0] in ['id', 'index'] and header[1] == 'text':
                    format_type = 'id_text'
                elif any('text' in col or 'sentence' in col for col in header):
                    format_type = 'text_label'

        for (i, line) in enumerate(lines[1:], start=1):  # Skip header
            if not line:
                logger.warning(f"Skipping line {i+1}: empty line")
                continue

            guid = "%s-%s" % (set_type, str(i))

            # Parse columns based on format
            if format_type == 'id_text':
                if len(line) < 2:
                    logger.warning(f"Skipping line {i+1}: insufficient columns")
                    continue
                text, label, agree = line[1].strip(), None, None
            elif len(line) == 2:
                # Auto-detect: if first col looks like ID, treat as id_text
                col0, col1 = line[0].strip(), line[1].strip()
                if (col0.isdigit() or len(col0) < 10) and len(col1) > len(col0):
                    text, label, agree = col1, None, None
                else:
                    text, label, agree = col0, col1, None
            else:  # 3+ columns: standard format
                if len(line) < 3:
                    logger.warning(f"Skipping line {i+1}: insufficient columns")
                    continue
                text = line[1].strip()
                label = line[2].strip() if len(line) > 2 else None
                agree = line[3] if len(line) > 3 else None

            if not text:
                logger.warning(f"Skipping line {i+1}: empty text")
                continue

            if label is not None and not label:
                logger.warning(f"Skipping line {i+1}: empty label")
                continue

            examples.append(InputExample(guid=guid, text=text, label=label, agree=agree))
        return examples


def normalize_label(label, label_list):
    """
    Normalize and match a label to the label_list (case-insensitive).
    Handles numeric labels, common variations, and case-insensitive matching.

    Returns:
        str: Matching label from label_list, or None if no match
    """
    if label is None:
        return None

    # Convert to string and clean: remove whitespace, tabs, newlines
    cleaned = str(label).strip().replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
    cleaned = ' '.join(cleaned.split())
    if not cleaned:
        return None

    # Handle numeric labels (0, 1, 2) - map to positive, negative, neutral
    if cleaned.isdigit():
        label_map = {'0': 'negative', '1': 'positive', '2': 'neutral'}  # Standard mapping: 0=neg, 1=pos, 2=neu
        if cleaned in label_map:
            mapped_label = label_map[cleaned]
            if mapped_label in label_list:
                return mapped_label
    
    # Handle LABEL_X format (e.g., LABEL_0, LABEL_1, LABEL_2)
    if cleaned.upper().startswith('LABEL_'):
        label_num = cleaned.upper().replace('LABEL_', '').strip()
        if label_num.isdigit():
            label_map = {'0': 'negative', '1': 'positive', '2': 'neutral'}
            if label_num in label_map:
                mapped_label = label_map[label_num]
                if mapped_label in label_list:
                    return mapped_label
    
    # Handle common label variations
    label_variations = {
        'positive': ['positive', 'pos', '1', 'positive sentiment', 'bullish', 'bull'],
        'negative': ['negative', 'neg', '-1', 'negative sentiment', 'bearish', 'bear'],
        'neutral': ['neutral', 'neu', '0', 'neutral sentiment', 'hold']
    }
    
    normalized = cleaned.lower()
    
    # Explicit mappings for common financial sentiment datasets
    explicit_mappings = {
        'bearish': 'negative',
        'bear': 'negative',
        'bullish': 'positive',
        'bull': 'positive',
        'neutral': 'neutral'
    }
    
    # Check explicit mappings first
    if normalized in explicit_mappings:
        mapped = explicit_mappings[normalized]
        if mapped in label_list:
            return mapped
    
    # Check direct match (case-insensitive)
    matching = next((label for label in label_list if label.lower() == normalized), None)
    if matching:
        return matching
    
    # Check variations
    for target_label in label_list:
        if target_label.lower() in label_variations:
            if normalized in [v.lower() for v in label_variations[target_label.lower()]]:
                return target_label
    
    # Try partial matching (e.g., "POSITIVE" matches "positive")
    for target_label in label_list:
        if normalized == target_label.lower() or normalized.startswith(target_label.lower()[:3]):
            return target_label
    
    return None


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, mode='classification'):
    """
    Loads a data file into a list of InputBatch's. With this function, the InputExample's are converted to features
    that can be used for the model. Text is tokenized, converted to ids and zero-padded. Labels are mapped to integers.

    Parameters
    ----------
    examples: list
        A list of InputExample's.
    label_list: list
        The list of labels.
    max_seq_length: int
        The maximum sequence length.
    tokenizer: BertTokenizer
        The tokenizer to be used.
    mode: str, optional
        The task type: 'classification' or 'regression'. Default is 'classification'

    Returns
    -------
    features: list
        A list of InputFeature's, which is an InputBatch.
    """
    # Check if max_seq_length exceeds model's maximum position embeddings
    if hasattr(tokenizer, 'model_max_length'):
        model_max_length = tokenizer.model_max_length
        if max_seq_length > model_max_length:
            logger.warning(
                f"max_seq_length ({max_seq_length}) exceeds model's maximum position embeddings "
                f"({model_max_length}). Truncating to {model_max_length}."
            )
            max_seq_length = model_max_length
    elif hasattr(tokenizer, 'max_len'):
        model_max_length = tokenizer.max_len
        if max_seq_length > model_max_length:
            logger.warning(
                f"max_seq_length ({max_seq_length}) exceeds tokenizer's max_len "
                f"({model_max_length}). Truncating to {model_max_length}."
            )
            max_seq_length = model_max_length

    if mode == 'classification':
        label_map = {label: i for i, label in enumerate(label_list)}
        label_map[None] = 9090
        if not hasattr(convert_examples_to_features, '_logged_label_map'):
            logger.info(f"Label map: {label_map}")
            convert_examples_to_features._logged_label_map = True

    features = []
    for (ex_index, example) in enumerate(examples):
        # Try using the modern tokenizer API with truncation (preferred method)
        use_modern_api = False
        if hasattr(tokenizer, '__call__'):
            try:
                # Use the modern API with truncation and padding
                encoded = tokenizer(
                    example.text,
                    max_length=max_seq_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors=None,  # Return lists, not tensors
                    add_special_tokens=True
                )
                input_ids = encoded['input_ids']
                attention_mask = encoded['attention_mask']
                # Get token_type_ids if available, otherwise create zeros
                if 'token_type_ids' in encoded:
                    token_type_ids = encoded['token_type_ids']
                else:
                    token_type_ids = [0] * len(input_ids)
                
                # Ensure we have exactly max_seq_length tokens (should already be handled by padding='max_length')
                if len(input_ids) != max_seq_length:
                    if len(input_ids) > max_seq_length:
                        # Truncate if somehow longer (shouldn't happen with truncation=True)
                        input_ids = input_ids[:max_seq_length]
                        attention_mask = attention_mask[:max_seq_length]
                        token_type_ids = token_type_ids[:max_seq_length]
                    else:
                        # Pad if shorter (shouldn't happen with padding='max_length', but handle it)
                        padding_length = max_seq_length - len(input_ids)
                        pad_token_id = tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None else 0
                        input_ids = input_ids + [pad_token_id] * padding_length
                        attention_mask = attention_mask + [0] * padding_length
                        token_type_ids = token_type_ids + [0] * padding_length
                
                use_modern_api = True
                # Convert to tokens for logging (optional, for debugging)
                tokens = tokenizer.convert_ids_to_tokens(input_ids) if hasattr(tokenizer, 'convert_ids_to_tokens') else []
            except (TypeError, AttributeError, ValueError) as e:
                # Fallback to manual tokenization if modern API doesn't work
                logger.debug(f"Modern tokenizer API failed, using fallback: {e}")
                use_modern_api = False
        
        if not use_modern_api:
            # Fallback: manual tokenization for older tokenizer APIs or if modern API failed
            tokens = tokenizer.tokenize(example.text)
            
            # Proper truncation: keep max_seq_length - 2 tokens (for [CLS] and [SEP])
            max_tokens = max_seq_length - 2
            if len(tokens) > max_tokens:
                # Truncate from the end (keep the beginning)
                tokens = tokens[:max_tokens]
            
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            token_type_ids = [0] * len(tokens)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            attention_mask = [1] * len(input_ids)
            
            # Apply padding
            padding = [0] * (max_seq_length - len(input_ids))
            pad_token_id = tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None else 0
            input_ids += [pad_token_id] * len(padding)
            attention_mask += padding
            token_type_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(token_type_ids) == max_seq_length

        if mode == 'classification':
            matched_label = normalize_label(example.label, label_list)
            if matched_label is None:
                if example.label is not None and ex_index < 5:
                    logger.warning(f"Label '{example.label}' not found in {label_list}, treating as None")
                label_id = label_map[None]
            else:
                label_id = label_map[matched_label]
        elif mode == 'regression':
            label_id = float(example.label) if example.label is not None else 0.0
        else:
            raise ValueError(f"Mode must be 'classification' or 'regression', got: {mode}")

        agree = example.agree
        mapagree = {'0.5': 1, '0.66': 2, '0.75': 3, '1.0': 4}
        try:
            agree = mapagree[agree]
        except:
            agree = 0

        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info(
                "token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          label_id=label_id,
                          agree=agree))
    return features


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=1)[:, None])
    return e_x / np.sum(e_x, axis=1)[:, None]


def get_metrics(df):
    "Computes accuracy and precision-recall for different sentiments."

    df.loc[:, 'guess'] = df.predictions.apply(np.argmax)
    df.loc[:, 'accurate'] = df.apply(lambda x: x['guess'] == x['labels'], axis=1)
    accuracy = df.accurate.sum() / df.shape[0]

    pos_recall = df[df['labels'] == 0].accurate.sum() / df[df['labels'] == 0].shape[0]
    neg_recall = df[df['labels'] == 1].accurate.sum() / df[df['labels'] == 1].shape[0]
    net_recall = df[df['labels'] == 2].accurate.sum() / df[df['labels'] == 2].shape[0]

    pos_precision = df[df['guess'] == 0].accurate.sum() / df[df['guess'] == 0].shape[0]
    neg_precision = df[df['guess'] == 1].accurate.sum() / df[df['guess'] == 1].shape[0]
    net_precision = df[df['guess'] == 2].accurate.sum() / df[df['guess'] == 2].shape[0]

    pos_f1score = 2 * (pos_precision * pos_recall) / (pos_precision + pos_recall)
    neg_f1score = 2 * (neg_precision * neg_recall) / (neg_precision + neg_recall)
    net_f1score = 2 * (net_precision * net_recall) / (net_precision + net_recall)

    return {'Accuracy': accuracy,
            'Positive': {'precision': pos_precision, 'recall': pos_recall, 'f1-score': pos_f1score}, 'Negative': \
                {'precision': neg_precision, 'recall': neg_recall, 'f1-score': neg_f1score},
            'Neutral': {'precision': net_precision, 'recall': net_recall, 'f1-score': net_f1score}}


def get_prediction(text, model, tokenizer):
    """
    Get one prediction.

    Parameters
    ----------
    text: str
        The text to be analyzed.
    model: BertModel
        The model to be used.
    tokenizer: BertTokenizer
        The tokenizer to be used.

    Returns
    -------
    predition: np.array
        An array that includes probabilities for each class.
    """

    tokens = tokenizer.tokenize(text)
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    token_type_ids = [0] * len(tokens)
    attention_mask = [1] * len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    padding = [0] * (64 - len(input_ids))
    input_ids += padding
    attention_mask += padding
    token_type_ids += padding

    features = []
    features.append(
        InputFeatures(input_ids=input_ids,
                      token_type_ids=token_type_ids,
                      attention_mask=attention_mask,
                      label_id=None))

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

    model.eval()
    prediction = softmax(model(all_input_ids, all_attention_mask, all_token_type_ids).detach().numpy())
    return prediction


def chunks(l, n):
    """
    Simple utility function to split a list into fixed-length chunks.
    Parameters
    ----------
    l: list
        given list
    n: int
        length of the sequence
    """
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i + n]