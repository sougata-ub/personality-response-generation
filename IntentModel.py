import torch
import torch.nn as nn


class SequenceClassifier(torch.nn.Module):
    def __init__(self, transformer_model, config, n_layers=4, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.transformer = transformer_model
        self.out = torch.nn.Linear(config.hidden_size * 2, self.num_classes)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.n_layers = n_layers

    def forward(self, input_ids, attention_mask, classification_labels=None):

        # Batch max length
        max_length = (attention_mask != 0).max(0)[0].nonzero()[-1].item() + 1
        if max_length < input_ids.shape[1]:
            input_ids = input_ids[:, :max_length]
            attention_mask = attention_mask[:, :max_length]

        segment_id = torch.zeros_like(attention_mask)
        hidden = self.transformer(input_ids=input_ids, attention_mask=attention_mask,
                                  token_type_ids=segment_id)

        token_hidden = hidden[2][-self.n_layers:]
        token_hidden = torch.mean(torch.sum(torch.stack(token_hidden), dim=0),
                                  dim=1)

        classifier_hidden = hidden[1]
        hidden_cat = torch.cat([token_hidden, classifier_hidden], dim=1)

        classification_logits = self.out(self.dropout(hidden_cat))
        outputs = [classification_logits]
        if classification_labels is not None:
            loss_fct_classification = torch.nn.CrossEntropyLoss()

            loss_classification = loss_fct_classification(classification_logits.view(-1, self.num_classes),
                                                          classification_labels.view(-1))

            outputs += [loss_classification]
        return outputs
