import numpy as np
from torch.utils.data import Dataset, Subset


def train_validation_test_split(
    dataset: Dataset,
    train_split: float = 0.80349,
    test_split: float = 0.0985296,
    val_split: float = 0.0980607,
    shuffle: bool = False,
    display: bool = False,
) -> tuple([Dataset, Dataset, Dataset]):
    """
    Split a dataset into a train, validation and test dataset.
    """
    # first assert that the splits ratios are correct
    assert (
        int(100 * (train_split + test_split + val_split)) == 100
    ), "train_split + test_split + val_split must be equals to 1, not {}".format(
        train_split + test_split + val_split
    )

    # build the indices
    idx = list(range(len(dataset)))

    # compute the number of examples in each dataset
    nb_val = int(np.round(len(dataset) * val_split))
    nb_test = int(np.round(len(dataset) * test_split))
    nb_train = len(dataset) - (nb_val + nb_test)

    if display:
        print("nb_train:", nb_train)
        print("nb_val  :", nb_val)
        print("nb_test :", nb_test)

    if shuffle:
        np.random.shuffle(idx)

    train_dataset = Subset(dataset, idx[:nb_train])
    val_dataset = Subset(dataset, idx[nb_train : nb_train + nb_val])
    test_dataset = Subset(dataset, idx[nb_train + nb_val :])

    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    print("TESTING train_validation_test_split")

    dataset = list(range(100))

    train_dataset, val_dataset, test_dataset = train_validation_test_split(
        dataset,
        train_split=0.6,
        test_split=0.2,
        val_split=0.2,
        shuffle=True,
        display=True,
    )

    print("train_dataset:", train_dataset.indices)
    print("val_dataset  :", val_dataset.indices)
    print("test_dataset :", test_dataset.indices)
