from collections import defaultdict
import torch
from torch import nn
import time
from utils import format_time, ids2string
from nltk.corpus import stopwords
from parallel import DataParallelCriterion
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


def category_keywords(model, train_loader, seed_word_list, device, top_num=20):
    
    t0 = time.time()
    cat_emb = defaultdict(list)
    cat_words = [defaultdict(float) for i in range(len(seed_word_list))]
    model.eval()

    for step, batch in enumerate(train_loader):
        if step % 40 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_loader), elapsed))
        with torch.no_grad():
            input_ids = batch[0].to(device)
            input_mask = batch[1].to(device)
            labels = batch[2]
            
            outputs = model(input_ids,
                            pred_mode="token",
                            token_type_ids=None, 
                            attention_mask=input_mask)
            
            predictions = outputs[0]

            for i, cat_word_id in enumerate(seed_word_list):
                match_idx = torch.zeros(input_ids.shape).bool().to(device)
                for word_id in cat_word_id:
                    match_idx = (input_ids == word_id) | match_idx

                if torch.sum(match_idx).item() > 0:
                    _, sorted_res = torch.topk(predictions[match_idx], top_num, dim=-1)
                    for word_list in sorted_res:
                        for j, word in enumerate(word_list):
                            cat_words[i][word.item()] += 1
    
    return cat_words


def filter_keywords(cat_words, inv_vocab, orig_seeds, top_num=50):
    
    all_words = defaultdict(list)
    sorted_dicts = []

    for i, cat_dict in enumerate(cat_words):
        sorted_dict = {k:v for k, v in sorted(cat_dict.items(), key=lambda item: item[1], reverse=True)[:top_num]}
        sorted_dicts.append(sorted_dict)
        for w in sorted_dict:
            all_words[w].append(i)

    repeat_words = []
    for w in all_words:
        if len(all_words[w]) > 1:
            repeat_words.append(w)

#     print(f"repeat: {[inv_vocab[w] for w in repeat_words]}")
    
    seed_word_list = []
    for sorted_dict in sorted_dicts:
        seed_word_list.append(np.array(list(sorted_dict.keys())))
    
    stopwords_vocab = stopwords.words('english')

    for i, word_list in enumerate(seed_word_list):
        delete_idx = []
        for j, w in enumerate(word_list):
            word = inv_vocab[w]
#             if w in orig_seeds[i]:
#                 continue
            if not word.isalpha() or len(word) == 1 or word in stopwords_vocab or w in repeat_words:
                delete_idx.append(j)
        seed_word_list[i] = np.delete(seed_word_list[i], delete_idx)
        print("\n" + " ".join([inv_vocab[w] for w in seed_word_list[i]]))
    
    return seed_word_list


def prepare_pretrain(model, train_loader, seed_word_list, device, top_num=20, match_threshold=10):
    
    t0 = time.time()
    cat_emb = defaultdict(list)
    threshold = [1, 5, 10, 15, 20, 25, 30]
    all_true_labels = defaultdict(list)
    all_input_ids = []
    all_mask_pos = []
    all_input_mask = []
    all_labels = []
    all_embs = []
    all_pred_labels = defaultdict(list)
    model.eval()

    for step, batch in enumerate(train_loader):
        if step % 40 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_loader), elapsed))
        with torch.no_grad():
            input_ids = batch[0].to(device)
            input_mask = batch[1].to(device)
            labels = batch[2]
            outputs = model(input_ids,
                            pred_mode="token",
                            token_type_ids=None, 
                            attention_mask=input_mask,
                            with_states=True)

            predictions = outputs[0]
            hidden_states = outputs[1]
            _, sorted_res = torch.topk(predictions, top_num, dim=-1)
            
            for i, cat_word_id in enumerate(seed_word_list):
                match_idx = sorted_res.clone().fill_(0).bool()
                for word_id in cat_word_id:
                    match_idx = (sorted_res == word_id) | match_idx
                
                for k_id, k in enumerate(threshold):
                    valid_idx = torch.sum(match_idx, dim=-1) > k
                    valid_doc = torch.sum(valid_idx, dim=-1) > 0
                    all_pred_labels[k_id] += [i] * torch.sum(valid_doc).item()
                    all_true_labels[k_id].append(labels[valid_doc].numpy())
                    
                    if k == match_threshold:
                        cur_inputs_ids = input_ids.clone().detach().cpu()
                        for valid_doc_idx in torch.where(valid_doc)[0]:
                            target_pos = torch.where(valid_idx[valid_doc_idx])[0][0]
                            all_embs.append(hidden_states[valid_doc_idx, target_pos].unsqueeze(0).detach().cpu())
                        all_input_ids.append(cur_inputs_ids[valid_doc])
#                         cur_inputs_ids[valid_idx] = mask_id
#                         all_input_ids_masked.append(cur_inputs_ids[valid_doc])
#                         cur_label = -100 * torch.ones(cur_inputs_ids.shape, dtype=torch.long)
#                         cur_label[valid_idx] = i
                        all_mask_pos.append(valid_idx[valid_doc].detach().cpu())
                        all_input_mask.append(input_mask[valid_doc].detach().cpu())
                        all_labels += [i] * torch.sum(valid_doc).item()

#             for i, cat_word_id in enumerate(seed_word_list):
#                 match_idx = torch.zeros(input_ids.shape).bool().to(device)
#                 for word_id in cat_word_id:
#                     match_idx = (input_ids == word_id) | match_idx

#                 if torch.sum(match_idx).item() > 0:
#                     sorted_val, sorted_res = torch.sort(predictions[match_idx], dim=-1, descending=True)

#                     for j, word_list in enumerate(sorted_res):
#                         hit_keywords = set(word_list[:top_num].detach().cpu().numpy()) & set(cat_word_id)
#                         for k_id, k in enumerate(threshold):
#                             if len(hit_keywords) > k:
#                                 if k == 10:
#                                     cur_inputs_ids = input_ids[torch.where(match_idx)[0][j]].clone().unsqueeze(0)
#                                     cur_label = -100 * torch.ones(cur_inputs_ids.shape, dtype=torch.long)
#                                     cur_label[0, torch.where(match_idx)[1][j]] = i
#                                     all_input_ids.append(cur_inputs_ids.detach().cpu())
#                                     all_labels.append(cur_label)
#                                     all_input_mask.append(input_mask[torch.where(match_idx)[0][j]].detach().cpu().unsqueeze(0))
#                                 all_pred_labels[k_id].append(i)
#                                 all_true_labels[k_id].append(labels[torch.where(match_idx)[0][j]].item())
    
    all_input_ids = torch.cat(all_input_ids, dim=0)
    all_embs = torch.cat(all_embs, dim=0)
    all_mask_pos = torch.cat(all_mask_pos, dim=0)
    all_input_mask = torch.cat(all_input_mask, dim=0)
#     all_labels = torch.cat(all_labels, dim=0)
    all_labels = torch.tensor(all_labels)
    
    return all_input_ids, all_mask_pos, all_input_mask, all_labels, all_pred_labels, all_true_labels, all_embs


def prepare_pretrain_weighted(model, train_loader, seed_word_list, device, top_num=20, match_threshold=10):
    
    t0 = time.time()
    cat_emb = defaultdict(list)
    threshold = [1, 5, 10, 15, 20, 25, 30]
    all_true_labels = defaultdict(list)
    all_input_ids = []
    all_mask_pos = []
    all_input_mask = []
    all_labels = []
    all_weights = []
    all_pred_labels = defaultdict(list)
    model.eval()

    for step, batch in enumerate(train_loader):
        if step % 40 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_loader), elapsed))
        with torch.no_grad():
            input_ids = batch[0].to(device)
            input_mask = batch[1].to(device)
            labels = batch[2]
            outputs = model(input_ids,
                            pred_mode="token",
                            token_type_ids=None, 
                            attention_mask=input_mask,
                            with_states=True)

            predictions = outputs[0]
            hidden_states = outputs[1]
            _, sorted_res = torch.topk(predictions, top_num, dim=-1)
            
            for i, cat_word_id in enumerate(seed_word_list):
                match_idx = sorted_res.clone().fill_(0).bool()
                for word_id in cat_word_id:
                    match_idx = (sorted_res == word_id) | match_idx
                
                for k_id, k in enumerate(threshold):
                    valid_idx = torch.sum(match_idx, dim=-1) > k
                    valid_doc = torch.sum(valid_idx, dim=-1) > 0
                    all_pred_labels[k_id] += [i] * torch.sum(valid_doc).item()
                    all_true_labels[k_id].append(labels[valid_doc].numpy())
                    
                    if k == match_threshold:
                        cur_inputs_ids = input_ids.clone().detach().cpu()
                        all_weights.append(torch.sum(match_idx, dim=-1).max(-1)[0][valid_doc].float().detach().cpu() / top_num)
                        all_input_ids.append(cur_inputs_ids[valid_doc])
#                         cur_inputs_ids[valid_idx] = mask_id
#                         all_input_ids_masked.append(cur_inputs_ids[valid_doc])
#                         cur_label = -100 * torch.ones(cur_inputs_ids.shape, dtype=torch.long)
#                         cur_label[valid_idx] = i
                        all_mask_pos.append(valid_idx[valid_doc].detach().cpu())
                        all_input_mask.append(input_mask[valid_doc].detach().cpu())
                        all_labels += [i] * torch.sum(valid_doc).item()

#             for i, cat_word_id in enumerate(seed_word_list):
#                 match_idx = torch.zeros(input_ids.shape).bool().to(device)
#                 for word_id in cat_word_id:
#                     match_idx = (input_ids == word_id) | match_idx

#                 if torch.sum(match_idx).item() > 0:
#                     sorted_val, sorted_res = torch.sort(predictions[match_idx], dim=-1, descending=True)

#                     for j, word_list in enumerate(sorted_res):
#                         hit_keywords = set(word_list[:top_num].detach().cpu().numpy()) & set(cat_word_id)
#                         for k_id, k in enumerate(threshold):
#                             if len(hit_keywords) > k:
#                                 if k == 10:
#                                     cur_inputs_ids = input_ids[torch.where(match_idx)[0][j]].clone().unsqueeze(0)
#                                     cur_label = -100 * torch.ones(cur_inputs_ids.shape, dtype=torch.long)
#                                     cur_label[0, torch.where(match_idx)[1][j]] = i
#                                     all_input_ids.append(cur_inputs_ids.detach().cpu())
#                                     all_labels.append(cur_label)
#                                     all_input_mask.append(input_mask[torch.where(match_idx)[0][j]].detach().cpu().unsqueeze(0))
#                                 all_pred_labels[k_id].append(i)
#                                 all_true_labels[k_id].append(labels[torch.where(match_idx)[0][j]].item())
    
    all_input_ids = torch.cat(all_input_ids, dim=0)
    all_weights = torch.cat(all_weights, dim=0)
    all_mask_pos = torch.cat(all_mask_pos, dim=0)
    all_input_mask = torch.cat(all_input_mask, dim=0)
#     all_labels = torch.cat(all_labels, dim=0)
    all_labels = torch.tensor(all_labels)
    
    return all_input_ids, all_mask_pos, all_input_mask, all_labels, all_pred_labels, all_true_labels, all_weights


def pretrain(model, train_loader, epochs, optimizer, scheduler, device, mask_id, margin=0.1, rank_loss_weight=1.0):
    
    model.train()

    for i in range(epochs):
        t0 = time.time()
        total_train_loss = 0
        
        print(f"\nEpoch {i+1}:")
        
        for step, batch in enumerate(train_loader):

            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print(f"lr: {optimizer.param_groups[0]['lr']:.4g}\tBatch {step:>5,}  of  {len(train_loader):>5,}.\tElapsed: {elapsed}.")

            input_ids = batch[0].to(device)
            mask_pos = batch[1].to(device)
            input_mask = batch[2].to(device)
            label_scalars = batch[3].to(device)
            
            labels = input_ids.clone().fill_(-100)
            for j, doc_pos in enumerate(mask_pos):
                labels[j, doc_pos] = label_scalars[j]

            model.zero_grad()

            outputs = model(input_ids, 
                            pred_mode="topic",
                            token_type_ids=None, 
                            attention_mask=input_mask)
            
            logits = outputs[0]
#             masked_topic_loss = nn.CrossEntropyLoss()(logits.view(-1, model.module.config.num_labels), labels.view(-1))
            
            cls_pred = nn.Softmax(dim=-1)(logits[:, 0, :])
            cls_pred = cls_pred[torch.arange(cls_pred.size(0)), label_scalars]
            
#             input_ids[mask_pos] = mask_id
            sampled_id = torch.randint(30522, (torch.sum(mask_pos).item(),)).to(device)
#             sampled_id = torch.multinomial(torch.ones(30522), num_samples=)
            input_ids[mask_pos] = sampled_id
            
            outputs_masked = model(input_ids, 
                                   pred_mode="topic",
                                   token_type_ids=None, 
                                   attention_mask=input_mask)
            
            logits = outputs_masked[0]
            
            ## masked topic loss
            masked_topic_loss = nn.CrossEntropyLoss()(logits.view(-1, model.module.config.num_labels), labels.view(-1))
            
            cls_pred_masked = nn.Softmax(dim=-1)(logits[:, 0, :])
            cls_pred_masked = cls_pred_masked[torch.arange(cls_pred_masked.size(0)), label_scalars]
            rank_topic_loss = nn.MarginRankingLoss(margin=margin)(cls_pred, cls_pred_masked, cls_pred.clone().fill_(-1))
            
            
            loss = masked_topic_loss + rank_loss_weight * rank_topic_loss
#             loss = topic_loss

            if torch.cuda.device_count() > 1:
                loss = loss.mean()
            total_train_loss += loss.item()
            loss.backward()
            # Clip the norm of the gradients to 1.0.
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_loader)   
        training_time = format_time(time.time() - t0)
        print(f"\n  Average training loss: {avg_train_loss:.2g}")
        print(f"  Training took: {training_time}")
    
    return


def train_bootstrap(model, train_loader, epochs, optimizer, scheduler, device):
    
    model.train()

    for i in range(epochs):
        t0 = time.time()
        total_train_loss = 0
        
        print(f"\nEpoch {i+1}:")
        
        for step, batch in enumerate(train_loader):

            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print(f"lr: {optimizer.param_groups[0]['lr']:.4g}\tBatch {step:>5,}  of  {len(train_loader):>5,}.\tElapsed: {elapsed}.")

            input_ids = batch[0].to(device)
            input_mask = batch[1].to(device)
            labels = batch[2].to(device)

            model.zero_grad()

            outputs = model(input_ids, 
                            pred_mode="topic",
                            token_type_ids=None, 
                            attention_mask=input_mask)
            
            logits = outputs[0][:, 0, :]
#             topic_loss = nn.CrossEntropyLoss()(logits.view(-1, model.module.config.num_labels), labels.view(-1))
            
            masked_topic_loss = nn.CrossEntropyLoss()(logits.view(-1, model.module.config.num_labels), labels.view(-1))
            
         
            loss = masked_topic_loss
#             loss = topic_loss

            if torch.cuda.device_count() > 1:
                loss = loss.mean()
            total_train_loss += loss.item()
            loss.backward()
            # Clip the norm of the gradients to 1.0.
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_loader)   
        training_time = format_time(time.time() - t0)
        print(f"\n  Average training loss: {avg_train_loss:.2g}")
        print(f"  Training took: {training_time}")
        
    return 


def pretrain_weighted(model, train_loader, epochs, optimizer, scheduler, device, mask_id, margin=0.1, rank_loss_weight=1.0):
    
    model.train()

    for i in range(epochs):
        t0 = time.time()
        total_train_loss = 0
        
        print(f"\nEpoch {i+1}:")
        
        for step, batch in enumerate(train_loader):

            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print(f"lr: {optimizer.param_groups[0]['lr']:.4g}\tBatch {step:>5,}  of  {len(train_loader):>5,}.\tElapsed: {elapsed}.")

            input_ids = batch[0].to(device)
            mask_pos = batch[1].to(device)
            input_mask = batch[2].to(device)
            label_scalars = batch[3].to(device)
            weights = batch[4].to(device)

            model.zero_grad()

            outputs = model(input_ids, 
                            pred_mode="topic",
                            token_type_ids=None, 
                            attention_mask=input_mask)
            
            logits = outputs[0]
#             topic_loss = nn.CrossEntropyLoss()(logits.view(-1, model.module.config.num_labels), labels.view(-1))
            
            cls_pred = nn.Softmax(dim=-1)(logits[:, 0, :])
            cls_pred = cls_pred[torch.arange(cls_pred.size(0)), label_scalars]
            
            input_ids[mask_pos] = mask_id
            
            outputs_masked = model(input_ids, 
                                   pred_mode="topic",
                                   token_type_ids=None, 
                                   attention_mask=input_mask)
            
            logits = outputs[0]
            
            weights_extend = input_ids.clone().float().fill_(0)
            
            labels = input_ids.clone().fill_(-100)
            for j, doc_pos in enumerate(mask_pos):
                labels[j, doc_pos] = label_scalars[j]
                weights_extend[j, doc_pos] = weights[j]
            
            ## masked topic loss
            masked_topic_loss = nn.CrossEntropyLoss(reduction='none')(logits.view(-1, model.module.config.num_labels), labels.view(-1))
            masked_topic_loss = torch.mean(masked_topic_loss * weights_extend.view(-1))
            
            cls_pred_masked = nn.Softmax(dim=-1)(logits[:, 0, :])
            cls_pred_masked = cls_pred_masked[torch.arange(cls_pred_masked.size(0)), label_scalars]
            
            rank_topic_loss = nn.MarginRankingLoss(margin=margin, reduction='none')(cls_pred, cls_pred_masked, cls_pred.clone().fill_(1))
            rank_topic_loss = torch.mean(rank_topic_loss * weights)
            
#             print(f"rank_loss: {rank_topic_loss}; mask_loss: {masked_topic_loss}")
            
            loss = masked_topic_loss + rank_loss_weight * rank_topic_loss
#             loss = topic_loss

            if torch.cuda.device_count() > 1:
                loss = loss.mean()
            total_train_loss += loss.item()
            loss.backward()
            # Clip the norm of the gradients to 1.0.
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_loader)   
        training_time = format_time(time.time() - t0)
        print(f"\n  Average training loss: {avg_train_loss:.2g}")
        print(f"  Training took: {training_time}")
    
    return


def self_train(model, test_loader, update_interval, batch_size, epochs, device, margin=0.1, rank_loss_weight=0.0, topic_loss_weight=0.0, self_conf_threshold=0.0):
    
    output = inference(model, test_loader, device, return_data=True, return_logits=True)
    target_scores = output[0]
    last_pred_labels = output[1]
    all_input_ids = output[2]
    all_input_mask = output[3]
    truth_labels = output[4]
    logits = output[5]
    
    total_loss = []
    total_acc = []
    
#     sorted_idx = sort_idx(target_scores)
#     target_scores = target_scores[sorted_idx]
#     last_pred_labels = last_pred_labels[sorted_idx]
#     all_input_ids = all_input_ids[sorted_idx]
#     all_input_mask = all_input_mask[sorted_idx]
#     truth_labels = truth_labels[sorted_idx]
#     logits = logits[sorted_idx]
    

    threshold = [0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    predictions = nn.Softmax(dim=-1)(logits)
    conf_pred = torch.max(predictions, dim=-1)[0]
    conf, future_predictions = torch.max(target_scores, dim=-1)

    for i in range(len(threshold)):
        cur_threshold = threshold[i]
        match = (conf > cur_threshold)
        if torch.sum(match).item() == 0:
            continue
        print("Target scores:")
        print(f"threshold: {cur_threshold} num_doc: {torch.sum(match).item()} acc: {torch.sum(future_predictions[match] == truth_labels[match]).item() / len(truth_labels[match])}")
        match = (conf_pred > cur_threshold)
        if torch.sum(match).item() == 0:
            continue
        print("Prediction scores:")
        print(f"threshold: {cur_threshold} num_doc: {torch.sum(match).item()} acc: {torch.sum(last_pred_labels[match] == truth_labels[match]).item() / len(truth_labels[match])}")
        print()

    total_steps = int(len(all_input_ids) * epochs / batch_size)
    optimizer = AdamW(model.parameters(), lr=2e-6, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    high_conf_idx = torch.where(conf > self_conf_threshold)[0]
    cur_idx = 0
    t0 = time.time()
    batch_loader_size = update_interval * batch_size
    for i in range(int(total_steps / update_interval)):
        select_idx = []
        while len(select_idx) < batch_loader_size:
            if cur_idx in high_conf_idx:
                select_idx.append(cur_idx)
            cur_idx = (cur_idx + 1) % len(all_input_ids)
        
        batch_dataset = TensorDataset(all_input_ids[select_idx], all_input_mask[select_idx], target_scores[select_idx])
        print(f"{len(batch_dataset)} samples")
        batch_loader = DataLoader(batch_dataset, sampler=RandomSampler(batch_dataset), batch_size=batch_size)
        
        batch_loss, batch_acc = self_train_batch(model, batch_loader, test_loader, optimizer, scheduler, device)
        
        print(batch_loss)
        print(batch_acc)
        total_loss.append(batch_loss)
        total_acc.append(batch_acc)
        
        training_time = format_time(time.time() - t0)
        print(f"Step: {i}\tTraining took: {training_time}\tlr: {optimizer.param_groups[0]['lr']:.4g}")
        output = inference(model, test_loader, device)
        target_scores = output[0]
#         target_scores = target_scores[sorted_idx]
        pred_labels = output[1]
#         pred_labels = pred_labels[sorted_idx]
        conf, _ = torch.max(target_scores, dim=-1)
        high_conf_idx = torch.where(conf > self_conf_threshold)[0]
        print(f"Delta label: {(1 - torch.sum(pred_labels == last_pred_labels).item() / len(pred_labels))*100:.2f}%")
        last_pred_labels = pred_labels
            
#     reg_iterator = iter(pretrain_loader)
#     for i in range(total_steps):
#         if i % update_interval == 0 and not i == 0:
#             training_time = format_time(time.time() - t0)
#             print(f"\nStep: {i}\tTraining took: {training_time}\tlr: {optimizer.param_groups[0]['lr']:.4g}")
#             output = inference(model, test_loader, device)
#             target_scores = output[0]
#             pred_labels = output[1]
#             conf, _ = torch.max(target_scores, dim=-1)
#             high_conf_idx = torch.where(conf > self_conf_threshold)[0]
#             print(f"Delta label: {(1 - torch.sum(pred_labels == last_pred_labels).item() / len(pred_labels))*100:.2f}%")
#             last_pred_labels = pred_labels

#         if cur_idx + batch_size < len(target_scores):
#             select_idx = list(range(cur_idx, cur_idx + batch_size))
#         else:
#             select_idx = list(range(cur_idx, len(target_scores))) + list(range((cur_idx + batch_size) % len(target_scores)))
#         cur_idx = (cur_idx + batch_size) % len(target_scores)
#         select_idx = [idx for idx in select_idx if idx in high_conf_idx]

#         batch = (all_input_ids[select_idx], all_input_mask[select_idx], target_scores[select_idx])
#         try:
#             reg_batch = next(reg_iterator)
#         except StopIteration:
#             reg_iterator = iter(pretrain_loader)
#             reg_batch = next(reg_iterator)

#         train(model, batch, reg_batch, optimizer, scheduler, device, mask_id, rank_loss_weight=0.0, topic_loss_weight=0.0)
    
    return total_loss, total_acc


def self_train_batch(model, batch_loader, test_loader, optimizer, scheduler, device):
    
    model.train()
    total_train_loss = 0
    losses = []
    accs = []
    
    for i, batch in enumerate(batch_loader):
        input_ids = batch[0].to(device)
        input_mask = batch[1].to(device)
        target_scores = batch[2].to(device)

        model.zero_grad()

        outputs = model(input_ids, 
                        pred_mode="topic",
                        token_type_ids=None, 
                        attention_mask=input_mask)

        logits = outputs[0][:, 0, :]
        preds = nn.Softmax(dim=-1)(logits)
        self_train_loss = nn.KLDivLoss(reduction='batchmean')(preds.view(-1, model.module.config.num_labels).log(), target_scores.view(-1, model.module.config.num_labels))

        loss = self_train_loss

        if torch.cuda.device_count() > 1:
            loss = loss.mean()
        total_train_loss += loss.item()
#         print(f"total_train_loss: {loss}")
        loss.backward()
        # Clip the norm of the gradients to 1.0.
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
#         if i % 10 == 0:
#             losses.append(loss.item())
#             output = inference(model, test_loader, device)
#             accs.append(output[-1])
    
    avg_train_loss = total_train_loss / len(batch_loader)
    print(f"\nAverage training loss: {avg_train_loss:.2g}")

    return losses, accs


def sort_idx(target_scores):
    
    max_score, _ = torch.max(target_scores, dim=-1)
    _, sorted_idx = torch.sort(max_score, descending=True)

    return sorted_idx


def train(model, batch, reg_batch, optimizer, scheduler, device, mask_id, margin=0.1, topic_loss_weight=1.0, rank_loss_weight=1.0):
    
    model.train()
    
    input_ids = batch[0].to(device)
    input_mask = batch[1].to(device)
    target_scores = batch[2].to(device)

    model.zero_grad()

    
    outputs = model(input_ids, 
                    pred_mode="topic",
                    token_type_ids=None, 
                    attention_mask=input_mask)

    logits = outputs[0][:, 0, :]
    preds = nn.Softmax(dim=-1)(logits)
#     self_train_loss = DataParallelCriterion(nn.KLDivLoss(reduction='batchmean'))(logits.view(-1, model.module.config.num_labels).log(), target_scores.view(-1, model.module.config.num_labels))
    self_train_loss = nn.KLDivLoss(reduction='batchmean')(preds.view(-1, model.module.config.num_labels).log(), target_scores.view(-1, model.module.config.num_labels))
    
    input_ids = reg_batch[0].to(device)
    mask_pos = reg_batch[1].to(device)
    input_mask = reg_batch[2].to(device)
    label_scalars = reg_batch[3].to(device)

    outputs = model(input_ids, 
                    pred_mode="topic",
                    token_type_ids=None, 
                    attention_mask=input_mask)

    logits = outputs[0]
#     logits = [output[:, 0, :] for output in outputs]
#             topic_loss = nn.CrossEntropyLoss()(logits.view(-1, model.module.config.num_labels), labels.view(-1))

    cls_pred = nn.Softmax(dim=-1)(logits[:, 0, :])
    cls_pred = cls_pred[torch.arange(cls_pred.size(0)), label_scalars]

    input_ids[mask_pos] = mask_id

    outputs_masked = model(input_ids, 
                           pred_mode="topic",
                           token_type_ids=None, 
                           attention_mask=input_mask)

    logits = outputs_masked[0]

    labels = input_ids.clone().fill_(-100)
    for j, doc_pos in enumerate(mask_pos):
        labels[j, doc_pos] = label_scalars[j]

    ## masked topic loss
    masked_topic_loss = nn.CrossEntropyLoss()(logits.view(-1, model.module.config.num_labels), labels.view(-1))

    cls_pred_masked = nn.Softmax(dim=-1)(logits[:, 0, :])
    cls_pred_masked = cls_pred_masked[torch.arange(cls_pred_masked.size(0)), label_scalars]

    rank_topic_loss = nn.MarginRankingLoss(margin=margin)(cls_pred, cls_pred_masked, cls_pred.clone().fill_(1))

    loss = self_train_loss + topic_loss_weight * masked_topic_loss + rank_loss_weight * rank_topic_loss
    
    if torch.cuda.device_count() > 1:
        loss = loss.mean()
    loss.backward()
    # Clip the norm of the gradients to 1.0.
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    
#     print(f"\n  Average training loss: {avg_train_loss:.2g}")
#     print(f"  Training took: {training_time}")

    return 


def target_score(logits):
    
    preds = nn.Softmax(dim=-1)(logits)
    weight = preds**2 / torch.sum(preds, dim=0)

    return (weight.t() / torch.sum(weight, dim=1)).t()


def inference(model, test_loader, device, return_data=False, return_logits=False):
    
    sents = []
    words = []
#     target_scores = []
    pred_labels = []
    truth_labels = []
    all_logits = []
    if return_data:
        all_input_ids = []
        all_input_mask = []
    model.eval()
    
    for step, batch in enumerate(test_loader):
#         if step % 40 == 0 and not step == 0:
#             elapsed = format_time(time.time() - t0)
#             print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_loader), elapsed))
        with torch.no_grad():
            input_ids = batch[0].to(device)
            input_mask = batch[1].to(device)
            labels = batch[2]
            outputs = model(input_ids, 
                            pred_mode="topic",
                            token_type_ids=None, 
                            attention_mask=input_mask)
            logits = outputs[0][:, 0, :]
#             target_scores.append(target_score(logits).detach().cpu())
            pred_labels.append(torch.argmax(logits, dim=-1).detach().cpu())
            truth_labels.append(labels)
            all_logits.append(logits.detach().cpu())
            if return_data:
                all_input_ids.append(input_ids.detach().cpu())
                all_input_mask.append(input_mask.detach().cpu())
    
    pred_labels = torch.cat(pred_labels, dim=0)
#     target_scores = torch.cat(target_scores, dim=0)
    logits = torch.cat(all_logits, dim=0)
    target_scores = target_score(logits)
    output = (target_scores, pred_labels)
    truth_labels = torch.cat(truth_labels, dim=0)
    
    if return_data:
        all_input_ids = torch.cat(all_input_ids, dim=0)
        all_input_mask = torch.cat(all_input_mask, dim=0)
        output = output + (all_input_ids, all_input_mask, truth_labels)
    if return_logits:
        output = output + (logits,)
    
    acc = torch.sum(pred_labels == truth_labels).item() / len(truth_labels)
    wrong_ids = all_input_ids[pred_labels != truth_labels]
    output = output + (acc,)
    output = output + (wrong_ids,)
    print(f"Acc: {acc:.4f}")
    
    return output