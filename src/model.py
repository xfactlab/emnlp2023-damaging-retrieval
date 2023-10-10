from pprint import pprint
import numpy as np
import torch
from torch import nn
import transformers
from transformers import AutoConfig
from transformers import T5PreTrainedModel
import copy
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.d_model, config.d_model)
        classifier_dropout = (
            config.dropout_rate
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.d_model, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class AdaptivePooler(nn.Module):
    """ Calcualte weighted average of the inputs with learnable weights """

    def __init__(self, config):
        super().__init__()
        self.input_size = config.d_model
        self.w = nn.Linear(self.input_size, 1, bias=True)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.classifier = nn.Linear(config.d_model, config.num_labels)

    def forward(self, inputs, mask=None):
        batch_size, seq_len, emb_dim = inputs.shape
        scores = torch.squeeze(self.w(inputs), dim=-1)
        weights = nn.functional.softmax(scores, dim=-1)
        if mask is not None:
            weights = weights * mask
            weights = weights / weights.sum(dim=-1, keepdims=True)
        outputs = (inputs.permute(2, 0, 1) * weights).sum(-1).T

        outputs = self.dropout(outputs)
        logits = self.classifier(outputs)

        return logits


class FiDEncoderForSequenceClassification(nn.Module):
    def __init__(self, config, model_encoder, pooler='adaptive'):
        super(FiDEncoderForSequenceClassification, self).__init__()
        self.num_labels = config.num_labels
        self.config = config

        self.encoder = model_encoder

        classification_class = AdaptivePooler if pooler == 'adaptive' else ClassificationHead
        self.classifier = classification_class(self.config)

        self.classifier.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_factor)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
        labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.classifier(outputs[0], mask=attention_mask)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class SentenceLSTM(nn.Module):
    def __init__(self, num_layers, embedding_size, num_labels, drop_out_rate):
        super(SentenceLSTM, self).__init__()
        self.num_layers = num_layers
        self.embedding_size = embedding_size
        self.num_labels = num_labels
        self.drop_out_rate = drop_out_rate

        self.lstm = nn.LSTM(input_size=self.embedding_size,
                            hidden_size=self.embedding_size,
                            num_layers=self.num_layers,
                            dropout=self.drop_out_rate,
                            batch_first=True)

        # Classifier Layers
        self.dropout = nn.Dropout(self.drop_out_rate)
        self.dense = nn.Linear(self.embedding_size, self.embedding_size)
        self.out_proj = nn.Linear(self.embedding_size, self.num_labels)

        # Initializing layers
        self.out_proj.apply(self._init_weights)
        self.dense.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=1)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, inputs, sequence_length_lst):
        packed_input = pack_padded_sequence(inputs, sequence_length_lst.cpu(), batch_first=True, enforce_sorted=False)

        output_packed, (h, c) = self.lstm(packed_input)

        padded_output, lengths = pad_packed_sequence(output_packed, batch_first=True)

        # classifier
        x = self.dropout(padded_output)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x



