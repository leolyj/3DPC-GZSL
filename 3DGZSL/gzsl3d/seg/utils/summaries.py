import os
from tensorboardX import SummaryWriter

class TensorboardSummary:
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(os.path.join(self.directory))
        return writer