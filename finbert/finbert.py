from __future__ import absolute_import, division, print_function

import random

import pandas as pd
import torch
from torch.nn import MSELoss, CrossEntropyLoss
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
    TensorDataset)
from tqdm import tqdm_notebook as tqdm
from tqdm import trange
from nltk.tokenize import sent_tokenize
import nltk

# Download punkt_tab if not already downloaded
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)
from finbert.utils import *
import numpy as np
import logging

from torch.optim import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')


def get_device(no_cuda=False):
    """
    Get the best available device for PyTorch operations.
    Priority: CUDA > MPS > CPU

    Parameters
    ----------
    no_cuda: bool
        If True, skip CUDA even if available (default: False)

    Returns
    -------
    torch.device
        The best available device
    """
    if not no_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class Config(object):
    """The configuration class for training."""

    def __init__(self,
                 data_dir,
                 bert_model,
                 model_dir,
                 max_seq_length=64,
                 train_batch_size=32,
                 eval_batch_size=32,
                 learning_rate=5e-5,
                 num_train_epochs=10.0,
                 warm_up_proportion=0.1,
                 no_cuda=False,
                 do_lower_case=True,
                 seed=42,
                 local_rank=-1,
                 gradient_accumulation_steps=1,
                 fp16=False,
                 output_mode='classification',
                 discriminate=True,
                 gradual_unfreeze=True,
                 encoder_no=12,
                 base_model='bert-base-uncased'):
        """
        Parameters
        ----------
        data_dir: str
            Path for the training and evaluation datasets.
        bert_model: BertModel
            The BERT model to be used. For example: BertForSequenceClassification.from_pretrained(...)
        model_dir: str
            The path where the resulting model will be saved.
        max_seq_length: int
            The maximum length of the sequence to be used. Default value is 64.
        train_batch_size: int
            The batch size for the training. Default value is 32.
        eval_batch_size: int
            The batch size for the evaluation. Default value is 32.
        learning_rate: float
            The learning rate. Default value is 5e5.
        num_train_epochs: int
            Number of epochs to train. Default value is 4.
        warm_up_proportion: float
            During the training, the learning rate is linearly increased. This value determines when the learning rate
            reaches the intended learning rate. Default value is 0.1.
        no_cuda: bool
            Determines whether to use gpu. Default is False.
        do_lower_case: bool
            Determines whether to make all training and evaluation examples lower case. Default is True.
        seed: int
            Random seed. Defaults to 42.
        local_rank: int
            Used for number of gpu's that will be utilized. If set -1, no distributed training will be done. Default
            value is -1.
        gradient_accumulation_steps: int
            Number of gradient accumulations steps. Defaults to 1.
        fp16: bool
            Determines whether to use 16 bits for floats, instead of 32.
        output_mode: 'classification' or 'regression'
            Determines whether the task is classification or regression.
        discriminate: bool
            Determines whether to apply discriminative fine-tuning.
        gradual_unfreeze: bool
            Determines whether to gradually unfreeze lower and lower layers as the training goes on.
        encoder_no: int
            Starting from which layer the model is going to be finetuned. If set 12, whole model is going to be
            fine-tuned. If set, for example, 6, only the last 6 layers will be fine-tuned.
        """
        self.data_dir = data_dir
        self.bert_model = bert_model
        self.model_dir = model_dir
        self.do_lower_case = do_lower_case
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.local_rank = local_rank
        self.eval_batch_size = eval_batch_size
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.warm_up_proportion = warm_up_proportion
        self.no_cuda = no_cuda
        self.seed = seed
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.output_mode = output_mode
        self.fp16 = fp16
        self.discriminate = discriminate
        self.gradual_unfreeze = gradual_unfreeze
        self.encoder_no = encoder_no
        self.base_model = base_model


class FinBert(object):
    """
    The main class for FinBERT.
    """

    def __init__(self,
                 config):
        self.config = config

    def prepare_model(self, label_list):
        """
        Sets some of the components of the model: Dataset processor, number of labels, usage of gpu and distributed
        training, gradient accumulation steps and tokenizer.
        Parameters
        ----------
        label_list: list
            The list of labels values in the dataset. For example: ['positive','negative','neutral']
        """

        self.processors = {
            "finsent": FinSentProcessor
        }

        self.num_labels_task = {
            'finsent': 2
        }

        if self.config.local_rank == -1 or self.config.no_cuda:
            self.device = get_device(self.config.no_cuda)
            # For CUDA, count devices; for MPS, there's only one device; for CPU, zero
            if self.device.type == "cuda":
                self.n_gpu = torch.cuda.device_count()
            elif self.device.type == "mps":
                self.n_gpu = 1
            else:
                self.n_gpu = 0
        else:
            torch.cuda.set_device(self.config.local_rank)
            self.device = torch.device("cuda", self.config.local_rank)
            self.n_gpu = 1
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.distributed.init_process_group(backend='nccl')
        logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            self.device, self.n_gpu, bool(self.config.local_rank != -1), self.config.fp16))

        if self.config.gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                self.config.gradient_accumulation_steps))

        self.config.train_batch_size = self.config.train_batch_size // self.config.gradient_accumulation_steps

        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        if self.device.type == "cuda" and self.n_gpu > 0:
            torch.cuda.manual_seed_all(self.config.seed)

        if os.path.exists(self.config.model_dir) and os.listdir(self.config.model_dir):
            raise ValueError("Output directory ({}) already exists and is not empty.".format(self.config.model_dir))
        if not os.path.exists(self.config.model_dir):
            os.makedirs(self.config.model_dir)

        self.processor = self.processors['finsent']()
        self.num_labels = len(label_list)
        self.label_list = label_list

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, do_lower_case=self.config.do_lower_case)

    def get_data(self, phase):
        """
        Gets the data for training or evaluation. It returns the data in the format that pytorch will process. In the
        data directory, there should be a .csv file with the name <phase>.csv
        Parameters
        ----------
        phase: str
            Name of the dataset that will be used in that phase. For example if there is a 'train.csv' in the data
            folder, it should be set to 'train'.
        Returns
        -------
        examples: list
            A list of InputExample's. Each InputExample is an object that includes the information for each example;
            text, id, label...
        """
        # Check if prepare_model has been called
        if not hasattr(self, 'processor') or self.processor is None:
            raise AttributeError(
                "The 'processor' attribute is not initialized. "
                "Please call 'finbert.prepare_model(label_list)' first before calling 'get_data()'."
            )

        self.num_train_optimization_steps = None
        examples = None
        examples = self.processor.get_examples(self.config.data_dir, phase)
        self.num_train_optimization_steps = int(
            len(
                examples) / self.config.train_batch_size / self.config.gradient_accumulation_steps) * self.config.num_train_epochs

        if phase == 'train':
            # Try to detect CSV vs TSV format
            train_path = os.path.join(self.config.data_dir, 'train.csv')
            # Read first line to detect delimiter
            with open(train_path, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                delimiter = ',' if ',' in first_line else '\t'
            
            train = pd.read_csv(train_path, sep=delimiter, index_col=False)
            weights = list()
            labels = self.label_list

            # Use bracket notation to access column (more robust)
            if 'label' not in train.columns:
                raise ValueError(f"Column 'label' not found in train.csv. Available columns: {train.columns.tolist()}")
            
            # Calculate class weights with safeguards
            class_weights = []
            for label in labels:
                label_count = train[train['label'] == label].shape[0]
                if label_count == 0:
                    logger.warning(f"Label '{label}' not found in training data! Using weight 1.0")
                    class_weights.append(1.0)
                else:
                    weight = train.shape[0] / label_count
                    if np.isnan(weight) or np.isinf(weight):
                        logger.warning(f"Invalid weight calculated for label '{label}': {weight}. Using weight 1.0")
                        class_weights.append(1.0)
                    else:
                        class_weights.append(weight)
            
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
            
            # Verify class weights are valid
            if torch.isnan(self.class_weights).any() or torch.isinf(self.class_weights).any():
                logger.error(f"Invalid class weights detected: {self.class_weights}. Using uniform weights.")
                self.class_weights = torch.ones(len(labels), dtype=torch.float32)
            
            logger.info(f"Class weights: {self.class_weights}")

        return examples

    def create_the_model(self):
        """
        Creates the model. Sets the model to be trained and the optimizer.
        """

        model = self.config.bert_model

        model.to(self.device)

        # Prepare optimizer
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        lr = self.config.learning_rate
        dft_rate = 1.2

        if self.config.discriminate:
            # apply the discriminative fine-tuning. discrimination rate is governed by dft_rate.

            encoder_params = []
            for i in range(12):
                encoder_decay = {
                    'params': [p for n, p in list(model.bert.encoder.layer[i].named_parameters()) if
                               not any(nd in n for nd in no_decay)],
                    'weight_decay': 0.01,
                    'lr': lr / (dft_rate ** (12 - i))}
                encoder_nodecay = {
                    'params': [p for n, p in list(model.bert.encoder.layer[i].named_parameters()) if
                               any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0,
                    'lr': lr / (dft_rate ** (12 - i))}
                encoder_params.append(encoder_decay)
                encoder_params.append(encoder_nodecay)

            optimizer_grouped_parameters = [
                {'params': [p for n, p in list(model.bert.embeddings.named_parameters()) if
                            not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01,
                 'lr': lr / (dft_rate ** 13)},
                {'params': [p for n, p in list(model.bert.embeddings.named_parameters()) if
                            any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0,
                 'lr': lr / (dft_rate ** 13)},
                {'params': [p for n, p in list(model.bert.pooler.named_parameters()) if
                            not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01,
                 'lr': lr},
                {'params': [p for n, p in list(model.bert.pooler.named_parameters()) if
                            any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0,
                 'lr': lr},
                {'params': [p for n, p in list(model.classifier.named_parameters()) if
                            not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01,
                 'lr': lr},
                {'params': [p for n, p in list(model.classifier.named_parameters()) if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0,
                 'lr': lr}]

            optimizer_grouped_parameters.extend(encoder_params)


        else:
            param_optimizer = list(model.named_parameters())

            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

        schedule = "warmup_linear"


        self.num_warmup_steps = int(float(self.num_train_optimization_steps) * self.config.warm_up_proportion)

        self.optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.config.learning_rate)

        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                    num_warmup_steps=self.num_warmup_steps,
                                                    num_training_steps=self.num_train_optimization_steps)

        return model

    def get_loader(self, examples, phase):
        """
        Creates a data loader object for a dataset.
        Parameters
        ----------
        examples: list
            The list of InputExample's.
        phase: 'train' or 'eval'
            Determines whether to use random sampling or sequential sampling depending on the phase.
        Returns
        -------
        dataloader: DataLoader
            The data loader object.
        """

        features = convert_examples_to_features(examples, self.label_list,
                                                self.config.max_seq_length,
                                                self.tokenizer,
                                                self.config.output_mode)

        # Log the necessasry information
        logger.info("***** Loading data *****")
        logger.info("  Num examples = %d", len(examples))
        logger.info("  Batch size = %d", self.config.train_batch_size)
        logger.info("  Num steps = %d", self.num_train_optimization_steps)

        # Load the data, make it into TensorDataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

        if self.config.output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        elif self.config.output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

        try:
            all_agree_ids = torch.tensor([f.agree for f in features], dtype=torch.long)
        except:
            all_agree_ids = torch.tensor([0.0 for f in features], dtype=torch.long)

        data = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids, all_agree_ids)

        # Distributed, if necessary
        if phase == 'train':
            my_sampler = RandomSampler(data)
        elif phase == 'eval':
            my_sampler = SequentialSampler(data)

        dataloader = DataLoader(data, sampler=my_sampler, batch_size=self.config.train_batch_size)
        return dataloader

    def train(self, train_examples, model):
        """
        Trains the model.
        Parameters
        ----------
        examples: list
            Contains the data as a list of InputExample's
        model: BertModel
            The Bert model to be trained.
        weights: list
            Contains class weights.
        Returns
        -------
        model: BertModel
            The trained model.
        """

        validation_examples = self.get_data('validation')

        global_step = 0

        self.validation_losses = []
        self.training_losses = []

        # Training
        train_dataloader = self.get_loader(train_examples, 'train')

        # Check model parameters for NaN/Inf before training
        has_nan_params = False
        for name, param in model.named_parameters():
            if param.requires_grad:
                if torch.isnan(param).any() or torch.isinf(param).any():
                    logger.error(f"Model parameter '{name}' contains NaN or Inf before training!")
                    has_nan_params = True
        if has_nan_params:
            logger.error("Model contains NaN/Inf parameters. This will cause training to fail. "
                        "Consider reinitializing the model or checking the pretrained weights.")

        model.train()

        step_number = len(train_dataloader)

        i = 0
        for _ in trange(int(self.config.num_train_epochs), desc="Epoch"):

            model.train()

            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            for step, batch in enumerate(tqdm(train_dataloader, desc='Iteration')):

                if (self.config.gradual_unfreeze and i == 0):
                    for param in model.bert.parameters():
                        param.requires_grad = False

                if (step % (step_number // 3)) == 0:
                    i += 1

                if (self.config.gradual_unfreeze and i > 1 and i < self.config.encoder_no):

                    for k in range(i - 1):

                        try:
                            for param in model.bert.encoder.layer[self.config.encoder_no - 1 - k].parameters():
                                param.requires_grad = True
                        except:
                            pass

                if (self.config.gradual_unfreeze and i > self.config.encoder_no + 1):
                    for param in model.bert.embeddings.parameters():
                        param.requires_grad = True

                batch = tuple(t.to(self.device) for t in batch)

                input_ids, attention_mask, token_type_ids, label_ids, agree_ids = batch

                logits = model(input_ids, attention_mask, token_type_ids)[0]
                
                # Check for NaN or Inf in logits during training
                skip_batch = False
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    nan_count = torch.isnan(logits).sum().item()
                    inf_count = torch.isinf(logits).sum().item()
                    logger.warning(f"NaN or Inf detected in training logits! NaN: {nan_count}, Inf: {inf_count}. Skipping batch.")
                    skip_batch = True
                
                # Check for extreme logit values that could cause numerical instability
                if not skip_batch:
                    logit_max = logits.max().item()
                    logit_min = logits.min().item()
                    if abs(logit_max) > 100 or abs(logit_min) > 100:
                        logger.warning(f"Extreme logit values detected: min={logit_min:.2f}, max={logit_max:.2f}. "
                                     f"This may cause numerical instability. Clipping logits.")
                        logits = torch.clamp(logits, min=-100, max=100)
                
                weights = self.class_weights.to(self.device)
                
                # Check for invalid label_ids (9090 is the ignore index, shouldn't be in training)
                if not skip_batch and (label_ids == 9090).any():
                    invalid_count = (label_ids == 9090).sum().item()
                    total_count = label_ids.size(0)
                    if invalid_count == total_count:
                        logger.warning(f"All {total_count} labels in batch are invalid (9090). Skipping batch.")
                        skip_batch = True
                    else:
                        logger.warning(f"Found {invalid_count}/{total_count} invalid label_ids (9090) in training batch. "
                                     f"These will be ignored in loss calculation.")

                if not skip_batch:
                    if self.config.output_mode == "classification":
                        # Verify weights are valid
                        if torch.isnan(weights).any() or torch.isinf(weights).any():
                            logger.error(f"Invalid class weights detected: {weights}. Using uniform weights.")
                            weights = torch.ones(self.num_labels, dtype=torch.float32).to(self.device)
                        
                        # Use ignore_index=9090 to ignore invalid labels
                        loss_fct = CrossEntropyLoss(weight=weights, ignore_index=9090)
                        loss = loss_fct(logits.view(-1, self.num_labels), label_ids.view(-1))
                        
                        # Check if loss is NaN or Inf
                        if torch.isnan(loss) or torch.isinf(loss):
                            logger.warning(f"NaN or Inf detected in training loss! Loss value: {loss.item()}. "
                                         f"Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}], "
                                         f"Label IDs: {label_ids.unique().tolist()}, "
                                         f"Weights: {weights.tolist()}")
                            skip_batch = True
                    elif self.config.output_mode == "regression":
                        loss_fct = MSELoss()
                        loss = loss_fct(logits.view(-1), label_ids.view(-1))
                        if torch.isnan(loss) or torch.isinf(loss):
                            logger.warning(f"NaN or Inf detected in training loss! Loss value: {loss.item()}. Skipping batch.")
                            skip_batch = True

                if skip_batch:
                    # Skip this batch but still update counters for gradient accumulation
                    if (step + 1) % self.config.gradient_accumulation_steps == 0:
                        # Zero gradients to maintain consistency
                        self.optimizer.zero_grad()
                    continue

                if self.config.gradient_accumulation_steps > 1:
                    loss = loss / self.config.gradient_accumulation_steps
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.config.fp16:
                        lr_this_step = self.config.learning_rate * warmup_linear(
                            global_step / self.num_train_optimization_steps, self.config.warm_up_proportion)
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1

            # Calculate average training loss for this epoch
            avg_tr_loss = tr_loss / nb_tr_steps if nb_tr_steps > 0 else 0
            self.training_losses.append(avg_tr_loss)

            # Validation

            validation_loader = self.get_loader(validation_examples, phase='eval')
            model.eval()

            valid_loss, valid_accuracy = 0, 0
            nb_valid_steps, nb_valid_examples = 0, 0

            # Define weights for validation (same as training)
            weights = self.class_weights.to(self.device)

            for input_ids, attention_mask, token_type_ids, label_ids, agree_ids in tqdm(validation_loader, desc="Validating"):
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                token_type_ids = token_type_ids.to(self.device)
                label_ids = label_ids.to(self.device)
                agree_ids = agree_ids.to(self.device)

                with torch.no_grad():
                    logits = model(input_ids, attention_mask, token_type_ids)[0]
                    
                    # Check for NaN or Inf in logits
                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        logger.warning("NaN or Inf detected in validation logits!")
                        logger.warning(f"NaN count: {torch.isnan(logits).sum().item()}, Inf count: {torch.isinf(logits).sum().item()}")
                        # Skip this batch or use a fallback
                        continue

                    if self.config.output_mode == "classification":
                        loss_fct = CrossEntropyLoss(weight=weights)
                        tmp_valid_loss = loss_fct(logits.view(-1, self.num_labels), label_ids.view(-1))
                        
                        # Check if loss is NaN or Inf
                        if torch.isnan(tmp_valid_loss) or torch.isinf(tmp_valid_loss):
                            logger.warning(f"NaN or Inf detected in validation loss! Loss value: {tmp_valid_loss.item()}")
                            continue
                    elif self.config.output_mode == "regression":
                        loss_fct = MSELoss()
                        tmp_valid_loss = loss_fct(logits.view(-1), label_ids.view(-1))
                        if torch.isnan(tmp_valid_loss) or torch.isinf(tmp_valid_loss):
                            logger.warning(f"NaN or Inf detected in validation loss! Loss value: {tmp_valid_loss.item()}")
                            continue

                    # tmp_valid_loss is already a scalar, so we can use .item() directly
                    valid_loss += tmp_valid_loss.item()
                    nb_valid_steps += 1

            valid_loss = valid_loss / nb_valid_steps

            self.validation_losses.append(valid_loss)
            print("Validation losses: {}".format(self.validation_losses))

            if valid_loss == min(self.validation_losses):

                try:
                    os.remove(self.config.model_dir / ('temporary' + str(best_model)))
                except:
                    print('No best model found')
                torch.save({'epoch': str(i), 'state_dict': model.state_dict()},
                           self.config.model_dir / ('temporary' + str(i)))
                best_model = i

        # Save a trained model and the associated configuration
        checkpoint = torch.load(self.config.model_dir / ('temporary' + str(best_model)))
        model.load_state_dict(checkpoint['state_dict'])
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(self.config.model_dir, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(self.config.model_dir, CONFIG_NAME)
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())
        os.remove(self.config.model_dir / ('temporary' + str(best_model)))
        return model

    def evaluate(self, model, examples):
        """
        Evaluate the model.
        Parameters
        ----------
        model: BertModel
            The model to be evaluated.
        examples: list
            Evaluation data as a list of InputExample's/
        Returns
        -------
        evaluation_df: pd.DataFrame
            A dataframe that includes for each example predicted probability and labels.
        """

        eval_loader = self.get_loader(examples, phase='eval')

        logger.info("***** Running evaluation ***** ")
        logger.info("  Num examples = %d", len(examples))
        logger.info("  Batch size = %d", self.config.eval_batch_size)

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        predictions = []
        labels = []
        agree_levels = []
        text_ids = []

        for input_ids, attention_mask, token_type_ids, label_ids, agree_ids in tqdm(eval_loader, desc="Testing"):
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
            label_ids = label_ids.to(self.device)
            agree_ids = agree_ids.to(self.device)

            with torch.no_grad():
                logits = model(input_ids, attention_mask, token_type_ids)[0]

                if self.config.output_mode == "classification":
                    # Use ignore_index=9090 to ignore None labels (placeholder for missing labels)
                    loss_fct = CrossEntropyLoss(ignore_index=9090)
                    tmp_eval_loss = loss_fct(logits.view(-1, self.num_labels), label_ids.view(-1))
                elif self.config.output_mode == "regression":
                    loss_fct = MSELoss()
                    tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

                np_logits = logits.cpu().numpy()

                if self.config.output_mode == 'classification':
                    prediction = np.array(np_logits)
                elif self.config.output_mode == "regression":
                    prediction = np.array(np_logits)

                for agree_id in agree_ids:
                    agree_levels.append(agree_id.item())

                for label_id in label_ids:
                    labels.append(label_id.item())

                for pred in prediction:
                    predictions.append(pred)

                text_ids.append(input_ids)

                # tmp_eval_loss = loss_fct(logits.view(-1, self.num_labels), label_ids.view(-1))
                # tmp_eval_loss = model(input_ids, token_type_ids, attention_mask, label_ids)

                eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1

            # logits = logits.detach().cpu().numpy()
            # label_ids = label_ids.to('cpu').numpy()
            # tmp_eval_accuracy = accuracy(logits, label_ids)

            # eval_loss += tmp_eval_loss.mean().item()
            # eval_accuracy += tmp_eval_accuracy

        evaluation_df = pd.DataFrame({'predictions': predictions, 'labels': labels, "agree_levels": agree_levels})

        return evaluation_df


def predict(text, model, write_to_csv=False, path=None, use_gpu=False, gpu_name='cuda:0', batch_size=5):
    """
    Predict sentiments of sentences in a given text. The function first tokenizes sentences, make predictions and write
    results.
    Parameters
    ----------
    text: string
        text to be analyzed
    model: BertForSequenceClassification
        path to the classifier model
    write_to_csv (optional): bool
    path (optional): string
        path to write the string
    use_gpu: (optional): bool 
        enables inference on GPU
    gpu_name: (optional): string
        multi-gpu support: allows specifying which gpu to use
    batch_size: (optional): int
        size of batching chunks
    """
    model.eval()

    # Ensure punkt_tab is available
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        logger.info("Downloading NLTK punkt_tab tokenizer...")
        nltk.download('punkt_tab', quiet=True)
    
    sentences = sent_tokenize(text)

    if use_gpu:
        # Use the helper function to get the best available device
        device_obj = get_device(no_cuda=False)
        # If user specified a specific GPU name and we're using CUDA, use that
        if device_obj.type == "cuda" and gpu_name.startswith("cuda"):
            device = gpu_name
        else:
            device = str(device_obj)
    else:
        device = "cpu"
    logging.info("Using device: %s " % device)
    label_list = ['positive', 'negative', 'neutral']
    label_dict = {0: 'positive', 1: 'negative', 2: 'neutral'}
    result = pd.DataFrame(columns=['sentence', 'logit', 'prediction', 'sentiment_score'])
    for batch in chunks(sentences, batch_size):
        examples = [InputExample(str(i), sentence) for i, sentence in enumerate(batch)]

        features = convert_examples_to_features(examples, label_list, 64, tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(device)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long).to(device)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long).to(device)

        with torch.no_grad():
            model     = model.to(device)

            logits = model(all_input_ids, all_attention_mask, all_token_type_ids)[0]
            logging.info(logits)
            logits = softmax(np.array(logits.cpu()))
            sentiment_score = pd.Series(logits[:, 0] - logits[:, 1])
            predictions = np.squeeze(np.argmax(logits, axis=1))

            batch_result = {'sentence': batch,
                            'logit': list(logits),
                            'prediction': predictions,
                            'sentiment_score': sentiment_score}

            batch_result = pd.DataFrame(batch_result)
            result = pd.concat([result, batch_result], ignore_index=True)

    result['prediction'] = result.prediction.apply(lambda x: label_dict[x])
    if write_to_csv:
        result.to_csv(path, sep=',', index=False)

    return result
