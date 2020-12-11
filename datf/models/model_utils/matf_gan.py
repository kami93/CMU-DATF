""" Code for all the model submodules part
    of various model architecures. """

import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, device, embed_dim_agent, classifier_hidden=512, dropout=0.5):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
                nn.Linear(embed_dim_agent, classifier_hidden),
                nn.LeakyReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(classifier_hidden, 1),
                nn.Sigmoid()
                ).to(device)

    def forward(self, x):
        return self.classifier(x)

