import numpy as np
np.random.seed(1234)
import torch
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
from torch import nn
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import time
import datetime
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from model import BertClass
from parallel import DataParallelModel, DataParallelCriterion
from utils import create_dataset, get_free_gpu, format_time, flat_accuracy


def train(model, train_loader, test_loader, epochs):

    training_stats = []
    total_t0 = time.time()

    for epoch_i in range(epochs):
        
        print(f"\n======== Epoch {epoch_i+1} / {epochs} ========")
        t0 = time.time()
        total_train_loss = 0
        model.train()

        for step, batch in enumerate(train_loader):
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_loader), elapsed))

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            model.zero_grad()
            logits = model(b_input_ids, 
                         token_type_ids=None, 
                         attention_mask=b_input_mask, 
                         labels=b_labels)
            # loss_fct = DataParallelCriterion(nn.CrossEntropyLoss())
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, b_labels.view(-1))
            
            if num_gpu > 1:
                loss = loss.mean()
            total_train_loss += loss.item()
            loss.backward()
            # Clip the norm of the gradients to 1.0.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_loader)     
        training_time = format_time(time.time() - t0)
        print(f"\n  Average training loss: {avg_train_loss:.2f}")
        print(f"  Training took: {training_time}")
        
        # validation
        print("\nRunning Validation...")

        t0 = time.time()
        model.eval()

        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        for batch in test_loader:
            
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            with torch.no_grad():        
                logits = model(b_input_ids, 
                               token_type_ids=None, 
                               attention_mask=b_input_mask,
                               labels=b_labels)
            if num_gpu > 1:
                loss = loss.mean()
            total_eval_loss += loss.item()

            # logits = np.concatenate([logit[0].detach().cpu().numpy() for logit in logits], axis=0)
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            total_eval_accuracy += flat_accuracy(logits, label_ids)
            
        avg_val_accuracy = total_eval_accuracy / len(test_loader)
        print(f"  Accuracy: {avg_val_accuracy:.4f}")

        avg_val_loss = total_eval_loss / len(test_loader)
        
        validation_time = format_time(time.time() - t0)
        
        print(f"  Validation Loss: {avg_val_loss:.2f}")
        print(f"  Validation took: {validation_time}")

        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )
    print("\nTraining complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

    return


def save_model(output_dir, model):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Saving model to {output_dir}")

    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)

    # torch.save(args, os.path.join(output_dir, 'training_args.bin'))

    # model = model_class.from_pretrained(output_dir)
    # tokenizer = tokenizer_class.from_pretrained(output_dir)
    # model.to(device)
    return


if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    valid_gpu_idx = get_free_gpu(5012)
    num_gpu = len(valid_gpu_idx)
    if num_gpu == 0:
        print("No GPU available!")
        exit(-1)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(gpu_id) for gpu_id in valid_gpu_idx])
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

