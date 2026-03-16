import torch
import torch.nn as nn
import torch.nn.functional as F

def train_model(model, train_loader, val_loader,
                optimizer, criterion, vocab,
                train_losses, train_accuracies, val_losses, val_accuracies,
                epochs, device):

    model.to(device)
    wait = 0  # early stopping patience counter

    for epoch in range(epochs):

        # TRAINING ------------

        model.train()
        running_loss = 0
        running_acc = 0
        total_train = 0
        total_train_tokens = 0

        for images, latex in train_loader:
            images = images.to(device)
            latex = latex.to(device).long()   # (batch_size, seq_len)

            optimizer.zero_grad()

            # Teacher forcing input + target shift
            decoder_in = latex[:, :-1]   # Exclude the last token (to pass into the decoder)
            target = latex[:, 1:]        # Exclude the first token (to check if the decoder's predict was true)

            output = model(images, decoder_in)   # (batch_size, seq_len -1 , vocab_size)

            batch_size, seq_len1, vocab_size = output.shape
                            
            loss = criterion(
                output.reshape(batch_size * seq_len1, vocab_size),
                target.reshape(batch_size * seq_len1)
                )

            loss.backward()
            optimizer.step()

            # Accuray
            preds = output.argmax(dim=-1)   # (batch_size, seq_len -1)
            mask = target != vocab["<pad>"]

            correct = ((preds == target) * mask).sum().item()
            tokens = mask.sum().item()

            running_acc += correct
            total_train_tokens += tokens
            running_loss += loss.item() * images.size(0)
            total_train += images.size(0)

        train_loss = running_loss / total_train
        train_acc = running_acc / total_train_tokens  # token-level accuracy

    # VALIDATION ----------------------

        model.eval()
        val_loss_total = 0
        val_correct = 0
        val_total_tokens = 0

        with torch.no_grad():
            for images, latex in val_loader:
                images = images.to(device)
                latex = latex.to(device).long()

                decoder_in = latex[:, :-1]
                target = latex[:, 1:]

            output = model(images, decoder_in)

            batch_size, seq_len1, vocab_size = output.shape
            loss = criterion(
                output.reshape(batch_size * seq_len1, vocab_size),
                target.reshape(batch_size * seq_len1)
                )

            val_loss_total += loss.item() * images.size(0)

            preds = output.argmax(dim=-1)
            mask = target != vocab["<pad>"]

            val_correct += ((preds == target) * mask).sum().item()
            val_total_tokens += mask.sum().item()

        val_loss = val_loss_total / len(val_loader.dataset)
        val_acc = val_correct / val_total_tokens

        # SAVE METRICS -------------------

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # EARLY STOPPING -------------------
        if len(val_losses) > 3:
            if val_losses[-1] > min(val_losses[:-1]) - 1e-3:
                wait += 1
            else:
                wait = 0
        if wait >= 4:
            print("Early Stopping")
            break