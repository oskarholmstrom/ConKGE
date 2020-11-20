
import torch
from model import ConKGE
import time
import torch.nn.functional as F
from evaluate import predict, hits


def train_model(model, train_dataloader, valid_dataloader, optimizer, scheduler, device, dataset, epochs=4):
    no_batches = len(train_dataloader)
    print('Training...')
    for epoch in range(0, epochs):

        print("")
        print("-"*70)
        print("")

        t0_epoch = time.time()
        t0_batch = time.time()

        total_loss = 0
        batch_loss = 0

        model.train()

        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)

            orig_inputs = batch[0]
            inputs = batch[1]
            positions = batch[2]
            masks = batch[3]
            labels = batch[4]

            model.zero_grad()


            outputs = model(input_ids = inputs,
                            attention_mask=masks,
                            position_ids = positions,
                            labels = labels
                            )

            loss = outputs[0]
            total_loss += loss.item()
            batch_loss += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Total norm over all gradients together (viewed as a single vector)
            optimizer.step()

            scheduler.step() # Update the learning rate

            if (step % 1000 == 0 and not step == 0) or (step == no_batches-1):
                time_elapsed = time.time() - t0_batch

                print("Epoch: {}/{}, \t Batch: {}/{}, \t Train loss: {:.6f}, \t Time: {:.2f},"
                      .format(epoch+1, epochs, step, no_batches, batch_loss / no_batches, time_elapsed))

                batch_loss = 0
                t0_batch = time.time()


        avg_train_loss = total_loss / no_batches
        time_elapsed = time.time() - t0_epoch

        print("\nSummary epoch {}/{}:".format(epoch+1, epochs))
        print("Avg. train loss: {:.6f}, \t Time: {:.2f}".format(avg_train_loss, time_elapsed))

    
        # For every ten epoch, evaluate on the validation set
        if (epoch % 10 == 0 and not epoch == 0):
            print("Evaluate on validation set: ")
            preds, true_inputs, true_labels = predict(model, valid_dataloader, device)
            hits_1, hits_3, hits_10, total, ratio_h1, ratio_h3, ratio_h10 = hits(preds, true_inputs, true_labels, dataset)
            print("TOTAL: ", total)
            print("HITS@1: ", hits_1, ratio_h1)
            print("HITS@3: ", hits_3, ratio_h3)
            print("HITS@10", hits_10, ratio_h10)

            # Delete preds, true_inputs and true_labels to save space
            del preds
            del true_inputs
            del true_labels
        


    print("\nTraining complete!")

