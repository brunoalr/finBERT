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

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


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
        # Auto-detect delimiter by reading first line
        delimiter = ","
        with open(input_file, "r", encoding='utf-8') as f:
            first_line = f.readline()
            # Count tabs vs commas in first line
            tab_count = first_line.count('\t')
            comma_count = first_line.count(',')
            # Use tab if there are more tabs than commas, or if no commas but tabs exist
            if tab_count > comma_count or (comma_count == 0 and tab_count > 0):
                delimiter = "\t"
            f.seek(0)  # Reset to beginning
        
        # Always use quotechar for proper handling of quoted fields
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=delimiter, quotechar='"')
            lines = []
            for line in reader:
                # Ensure all cells are strings (Python 3 strings are already unicode)
                # For Python 2 compatibility, convert to unicode if needed
                if sys.version_info[0] == 2:
                    # Python 2: convert to unicode
                    try:
                        unicode_type = unicode  # noqa: F821
                    except NameError:
                        unicode_type = str
                    line = [unicode_type(cell, 'utf-8') for cell in line]
                else:
                    # Python 3: ensure all cells are strings
                    line = [str(cell) for cell in line]
                lines.append(line)
        return lines


class FinSentProcessor(DataProcessor):
    """
    Data processor for FinBERT.
    """

    def get_examples(self, data_dir, phase):
        """
        Get examples from the data directory.

        Parameters
        ----------
        data_dir: str
            Path for the data directory.
        phase: str
            Name of the .csv file to be loaded.
        """
        return self._create_examples(self._read_tsv(os.path.join(data_dir, (phase + ".csv"))), phase)

    def get_labels(self):
        return ["positive", "negative", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        # Detect format from header row
        has_header = len(lines) > 0
        is_id_text_format = False
        is_two_column_format = False
        
        if has_header:
            header = lines[0] if lines else []
            header_lower = [str(col).lower() for col in header]
            
            # Check if header is ID/index,text format (no label)
            if len(header) == 2:
                first_col = header_lower[0]
                second_col = header_lower[1]
                if (first_col in ['id', 'index'] and 
                    second_col == 'text'):
                    is_id_text_format = True
                # Check if header suggests 2-column format (sentence/text, sentiment/label)
                elif (any('sentence' in col or 'text' in col for col in header_lower) and
                      any('sentiment' in col or 'label' in col for col in header_lower)):
                    is_two_column_format = True
        
        for (i, line) in enumerate(lines):
            if i == 0:
                continue  # Skip header row
            # Skip empty lines
            if not line:
                logger.warning(f"Skipping line {i+1}: empty line")
                continue
            
            guid = "%s-%s" % (set_type, str(i))
            
            # Handle different column formats
            if is_id_text_format:
                # ID,text format: column 0 = ID, column 1 = text (no label)
                if len(line) < 2:
                    logger.warning(
                        f"Skipping line {i+1}: insufficient columns "
                        f"(expected at least 2, got {len(line)})")
                    continue
                text = line[1].strip() if len(line) > 1 else ""
                label = None  # No label in this format
                agree = None
            elif is_two_column_format:
                # 2-column format: column 0 = text, column 1 = label
                if len(line) < 2:
                    logger.warning(
                        f"Skipping line {i+1}: insufficient columns "
                        f"(expected at least 2, got {len(line)})")
                    continue
                text = line[0].strip() if len(line) > 0 else ""
                label = line[1].strip() if len(line) > 1 else ""
                agree = None
            elif len(line) == 2:
                # Fallback: if we have 2 columns but format is unknown,
                # check if first column looks like an ID (numeric or short)
                # and second looks like text (longer string)
                col0 = line[0].strip() if len(line) > 0 else ""
                col1 = line[1].strip() if len(line) > 1 else ""
                # If first column is numeric or very short, treat as ID,text
                if (col0.isdigit() or len(col0) < 10) and len(col1) > len(col0):
                    text = col1
                    label = None
                else:
                    # Otherwise assume text,label format
                    text = col0
                    label = col1
                agree = None
            else:
                # 3+ column format: column 1 = text, column 2 = label
                if len(line) < 3:
                    logger.warning(
                        f"Skipping line {i+1}: insufficient columns "
                        f"(expected at least 3, got {len(line)})")
                    continue
                text = line[1].strip() if len(line) > 1 else ""
                label = line[2].strip() if len(line) > 2 else ""
                try:
                    agree = line[3] if len(line) > 3 else None
                except Exception:
                    agree = None
            
            # Skip if text is empty
            if not text:
                logger.warning(f"Skipping line {i+1}: empty text")
                continue
            
            # Allow examples without labels (for test/prediction data)
            # Only skip if label is required but empty
            if label is not None and not label:
                logger.warning(f"Skipping line {i+1}: empty label")
                continue
            
            examples.append(
                InputExample(guid=guid, text=text, label=label, agree=agree))
        return examples


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

    if mode == 'classification':
        label_map = {label: i for i, label in enumerate(label_list)}
        label_map[None] = 9090
        # Log label_map for debugging (only first time)
        if not hasattr(convert_examples_to_features, '_logged_label_map'):
            logger.info(f"Label map: {label_map}")
            logger.info(f"Label list: {label_list}")
            convert_examples_to_features._logged_label_map = True

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens = tokenizer.tokenize(example.text)

        if len(tokens) > max_seq_length - 2:
            tokens = tokens[:(max_seq_length // 4) - 1] + tokens[
                                                          len(tokens) - (3 * max_seq_length // 4) + 1:]

        tokens = ["[CLS]"] + tokens + ["[SEP]"]

        token_type_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        attention_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        attention_mask += padding


        token_type_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(token_type_ids) == max_seq_length

        if mode == 'classification':
            # Handle None labels (for test/prediction data without labels)
            if example.label is None:
                label_id = label_map[None]
            else:
                # Clean label: remove tabs, newlines, extra whitespace
                label_cleaned = example.label.strip()
                # Remove any tab characters that might have been included
                label_cleaned = label_cleaned.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
                # Collapse multiple spaces
                label_cleaned = ' '.join(label_cleaned.split())
                # Normalize label to lowercase for case-insensitive matching
                label_normalized = label_cleaned.lower()
                # Find matching label in label_list (case-insensitive)
                matching_label = next((l for l in label_list if l.lower() == label_normalized), None)
                if matching_label is not None:
                    label_id = label_map[matching_label]
                else:
                    # Label not found, treat as None
                    # Only log warning for first few mismatches to avoid spam
                    if ex_index < 5:
                        logger.warning(
                            f"Label '{example.label}' (cleaned: '{label_cleaned}', normalized: '{label_normalized}') "
                            f"not found in label_list {label_list}, treating as None")
                    label_id = label_map[None]
        elif mode == 'regression':
            # Handle None labels (for test/prediction data without labels)
            if example.label is None:
                label_id = 0.0  # Default value for regression
            else:
                label_id = float(example.label)
        else:
            raise ValueError("The mode should either be classification or regression. You entered: " + mode)

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