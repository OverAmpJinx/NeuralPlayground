def training(model, loss_fn, optimizer, train_loader, X_val_split=None, y_val_split=None, epochs=100):
    """General training loop for a PyTorch model, with validation loss tracking which can be plotted later

    Args:
        model (): PyTorch model to be trained
        loss_fn (): Loss function for the model (e.g.: nn.MSELoss())
        optimizer (): Optimizer for the model
        train_loader (): loader containing training data in batches, setup with torch.utils.data.Tensordataset() and DataLoader()
        X_val_split (): X values for validation set if given
        y_val_split (): y values for validation set if given
        epochs (int, optional): Number of epochs to train. Defaults to 100.

    Returns:
        train_losses: training losses, appended step by step for each epoch
        val_losses: valdation losses, appended step by step for each epoch (if none provided, will be empty)
    """
    val_needed = X_val_split is not None and y_val_split is not None
    if val_needed:
        X_val_split = X_val_split.to(device)
        y_val_split = y_val_split.to(device).reshape(-1, 1) 
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        batch_loss = 0
    
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).reshape(-1, 1) 
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss += loss.item() * X_batch.size(0)
        
        
        model.eval()
        with torch.no_grad():
            if val_needed:
                y_val_pred = model(X_val_split)
                val_loss = loss_fn(y_val_pred, y_val_split.reshape(-1, 1)) 
                val_losses.append(val_loss.item()) 
        train_losses.append(batch_loss / len(train_loader.dataset))
            
    return train_losses, val_losses