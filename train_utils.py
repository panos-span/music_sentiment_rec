import torch
import numpy as np

class EarlyStopper:
    def __init__(self, model, save_path, patience=1, min_delta=0):
        self.model = model
        self.save_path = save_path
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            torch.save(self.model.state_dict(), self.save_path)
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def train_one_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for x, y, lengths in train_loader:        
        loss, logits = model(x.float().to(device), y.to(device), lengths.to(device))
        # prepare
        optimizer.zero_grad()
        # backward
        loss.backward()
        # optimizer step
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)    
    return avg_loss


def validate_one_epoch(model, val_loader, device):    
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y, lengths in val_loader:
            loss, logits = model(x.float().to(device), y.to(device), lengths.to(device))
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)    
    return avg_loss



def overfit_with_a_couple_of_batches(model, train_loader, optimizer, device, epochs:int):
    print('Training in overfitting mode...')
    
    # get only the 1st batch
    x_b1, y_b1, lengths_b1 = next(iter(train_loader))    
    model.train()
    for epoch in range(epochs):        
        loss, logits = model(x_b1.float().to(device), y_b1.to(device), lengths_b1.to(device))
        # prepare
        optimizer.zero_grad()
        # backward
        loss.backward()
        # optimizer step
        optimizer.step()

        if epoch == 0 or (epoch+1)%20 == 0:
            print(f'Epoch {epoch+1}, Loss at training set: {loss.item()}')


def train(model, train_loader, val_loader, optimizer, epochs, save_path, device, overfit_batch):
    if overfit_batch:
        overfit_with_a_couple_of_batches(model, train_loader, optimizer, device, epochs)
    else:
        print(f'Training started for model {save_path.replace(".pth", "")}...')
        early_stopper = EarlyStopper(model, save_path, patience=5)
        for epoch in range(epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            validation_loss = validate_one_epoch(model, val_loader, device)
            if epoch== 0 or (epoch+1)%5==0:
                print(f'Epoch {epoch+1}/{epochs}, Loss at training set: {train_loss}\n\tLoss at validation set: {validation_loss}')          
            
            if early_stopper.early_stop(validation_loss):
                print('Early Stopping was activated.')
                print(f'Epoch {epoch+1}/{epochs}, Loss at training set: {train_loss}\n\tLoss at validation set: {validation_loss}')
                print('Training has been completed.\n')
                break