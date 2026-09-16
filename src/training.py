import torch
from tqdm import tqdm

def eval(model,
         valid_loader,
         metric):

    with torch.no_grad():
        for X_batch, y_batch in tqdm(valid_loader,
                                     desc='VAL'):
            logits = model(X_batch)

            metric.update(logits, y_batch)
        return metric.compute()

def train(model,
          epochs,
          data_loader,
          criterion,
          optimizer,
          valid_loader=None,
          metric=None):

    
    for epoch in range(epochs):

        model.train()
        total_loss = 0

        progress = tqdm(data_loader, desc=f'EPOCH: {epoch + 1}/{epochs}')

        for X_batch, y_batch in progress:

            logits = model(X_batch)

            loss = criterion(logits, y_batch)
            with torch.no_grad():
                total_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            progress.set_postfix(
                loss=f'{loss.item():.2f}'
            )

        model.eval()
        eval_value = eval(
            model=model,
            valid_loader=valid_loader,
            metric=metric
        )
        metric.reset()

        print(f'EVAL: {eval_value}')