
import torch
import os
from model import ConKGE
import time
import torch.nn.functional as F
from evaluate import evaluate
from predict import predict


def train_model(args, model, train_dataloader, valid_dataloader, optimizer, scheduler, device, dataset, start_epoch, epochs=4):
    no_batches = len(train_dataloader)
    print('Training...')
    max_mrr = 0
    for epoch in range(start_epoch, epochs):

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


            outputs = model(args = args,
                            input_ids = inputs,
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

            scheduler.step() # Update learning rate

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

    
        # For every fifty epoch: evaluate on the validation set
        if ((epoch % 50 == 0 or epoch == epochs-1)):
            print("Evaluate on validation set: ")
            preds, true_inputs, true_labels = predict(args, model, valid_dataloader, device)
            mrr, hits_1, hits_3, hits_10, ratio_h1, ratio_h3, ratio_h10 = evaluate(preds, true_inputs, true_labels, dataset)
            
            print('{{"metric": "MRR", "value": {}}}'.format(mrr))
            print('{{"metric": "hits_1 (Total)", "value": {}}}'.format(hits_1))
            print('{{"metric": "hits_1 (Ratio)", "value": {}}}'.format(ratio_h1))
            print('{{"metric": "hits_3 (Total)", "value": {}}}'.format(hits_3))
            print('{{"metric": "hits_3 (Ratio)", "value": {}}}'.format(ratio_h3))
            print('{{"metric": "hits_10 (Total)", "value": {}}}'.format(hits_10))
            print('{{"metric": "hits_10 (Ratio)", "value": {}}}'.format(ratio_h10))
            # Delete preds, true_inputs and true_labels to save space
            del preds
            del true_inputs
            del true_labels

            # Save checkpoint
            print("max mrr: ", max_mrr)
            print("mrr: ", mrr)
            if mrr > max_mrr:
                print("Saving checkpoint...", end=" ")
                state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
                checkpoint_path = os.path.join(args.experiment_dir, 'checkpoint.pt')
                torch.save(state, checkpoint_path)
                max_mrr = mrr
                print("Done")

    print("\nTraining complete!")

