import random
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.utils import Sequence

class VariableLengthSeriesGenerator(Sequence):
    """
        Takes in a sequence of data-points at equal intervals, to produce batches for
        training/validation of (random) variable length (zero-padded).
    """

    def __init__(self, x_set, y_set, batch_size, max_length, min_length=None, **kargs):
        self.x_set = x_set
        self.y_set = y_set
        self.batch_size = batch_size
        self.max_length = max_length
        if not min_length:
            self.min_length = max_length//2 + 1
        self.max_delete = max_length - min_length
        self.generator = TimeseriesGenerator(
                            x_set, y_set, length=max_length,
                            batch_size=batch_size, **kargs
                        )

    def __len__(self):
        return len(self.generator)

    def __getitem__(self, idx):
        delete_idx = random.randint(0, self.max_delete)
        batch_x, batch_y = self.generator[idx]
        batch_x[:,:delete_idx,:] = 0

        return batch_x, batch_y
