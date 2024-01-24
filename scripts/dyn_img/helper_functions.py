from tqdm import tqdm


def train_epoch(model, data, optimizer, loss_func) -> float:
    """Perform the training of one epoch.

    Parameters
    ----------
    model
        Model to be trained
    data
        Dataloader with training data
    optimizer
        Pytorch optimizer, e.g. Adam
    loss_func
        Loss function to be calculated, e.g. MSE

    Returns
    -------
        training loss

    Raises
    ------
    ValueError
        loss is NaN
    """
    running_loss = 0.

    for sample in tqdm(data):
        optimizer.zero_grad(set_to_none=True)  # Zero your gradients for every batch!
        
        sample, label = sample
        output = model(sample)
        loss = loss_func(label, output)
        loss.backward()
        
        if loss.item() != loss.item():
            raise ValueError("NaN returned by loss function...")

        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(data.dataset)


def validate_epoch(model, data, loss_func) -> float:
    """Perform the validation of one epoch.

    Parameters
    ----------
    model
        Model to be trained
    data
        Dataloader with validation data
    loss_func
        Loss function to be calculated, e.g. MSE

    Returns
    -------
        validation loss
    """
    running_loss = 0.
    
    for sample in tqdm(data):
        inputs, labels = sample
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        running_loss += loss.item()

    return running_loss / len(data.dataset)
