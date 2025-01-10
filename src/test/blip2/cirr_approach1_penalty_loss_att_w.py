import datetime
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import json

from tqdm import tqdm

from src.tools.files import json_dump
from src.tools.utils import concat_all_gather


def attention_rollout(As):
    """Computes attention rollout from the given list of attention matrices.
    https://arxiv.org/abs/2005.00928
    """
    num_layers = As.shape[0]
    rollout = [None] * num_layers

    for i in range(num_layers):
        if i == 0:  # Base case
            rollout[i] = As[i]
        else:
            # General case
            rollout[i] = torch.matmul(As[i], rollout[i - 1])
    rollout = torch.stack(rollout)
    return rollout

def attention_rollout_per_sample(attentions, BATCH_INDEX, BATCH_SAMPLE):
    # attentions of dim [num_batches, num_layers, batch_size, num_heads, seq_len, seq_len]
    # Select the attention for the specified batch and the first layer
    attention = attentions[BATCH_INDEX, :, BATCH_SAMPLE]  # Shape: [num_layers, num_heads, seq_len, seq_len]
    rollout = attention_rollout(attention)
    return rollout

def compute_weights_based_on_attention_rollout(attentions, BATCH_INDEX, BATCH_SAMPLE, temp=1, threshold=0.01, return_logs = False):
  rollout = attention_rollout_per_sample(attentions, BATCH_INDEX, BATCH_SAMPLE) #Dim: [num_layers, num_heads, seq_len, seq_len] 
  # Average over last layer
  rollout_avg_last_layer = torch.mean(rollout[-1], dim=0) # Dim: [seq_len, seq_len]

  # Get the average for each column
  rollout_avg_last_layer_per_column = torch.mean(rollout_avg_last_layer, dim=0) # Dim: [seq_len]

  #Split into values for queries and text
  queries_rollout = rollout_avg_last_layer_per_column[:32]
  text_rollout    = rollout_avg_last_layer_per_column[32:]

  # Set to low value to all the values that are under a threshold
  threshold = threshold
  queries_rollout[queries_rollout < threshold] = -1e7
  text_rollout[text_rollout < threshold] = -1e7

  #How many non-zero values are
  if return_logs:
    print(f"Queries rollouts lower than 0: {torch.sum(queries_rollout < 0)} out of {len(queries_rollout)}")
    print(f"Text rollouts lower than 0: {torch.sum(text_rollout < 0)} out of {len(text_rollout)}")

  queries_dist = F.softmax(queries_rollout/temp, dim=0)
  text_dist = F.softmax(text_rollout/temp, dim=0)

  return queries_dist, text_dist


def get_interesting_text_tokens(text_dist, tokenizer, BATCH_SAMPLE):
  _, tokens = get_input_ids_and_tokens(tokenizer, captions[BATCH_SAMPLE])

  #Get the ids of the dist > 0
  ids = torch.where(text_dist > 0)[0]
  for id in ids:
    try:
      print(f"Token: {tokens[id]} \t - Prob: {text_dist[id]:.4f}")
    except IndexError:
      print(f"Token: [PAD] \t - Prob: {text_dist[id]:.4f}")

class TestCirr:
    def __init__(self):
        pass

    @staticmethod
    @torch.no_grad()
    def __call__(model, data_loader, fabric):
        model.eval()

        fabric.print("Computing features for test...")
        start_time = time.time()

        vl_feats = []
        pair_ids = []
        for i,batch in tqdm(enumerate(data_loader)):
            ref_img = batch["ref_img"]
            caption = batch["edit"]
            pair_id = batch["pair_id"]

            pair_ids.extend(pair_id.cpu().numpy().tolist())

            device = ref_img.device

            ref_img_embs = model.ln_vision(model.visual_encoder(ref_img))
            ref_img_atts = torch.ones(ref_img_embs.size()[:-1], dtype=torch.long).to(
                device
            )

            text_tokens = model.tokenizer(
                caption,
                padding="longest",
                truncation=True,
                max_length=64,
                return_tensors="pt",
            ).to(device)

            # Shift encoder
            query_tokens = model.query_tokens.expand(ref_img_embs.shape[0], -1, -1)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                device
            )
            attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)

            output = model.Qformer.bert(
                text_tokens.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=ref_img_embs,
                encoder_attention_mask=ref_img_atts,
                return_dict=True,
                output_attentions=True
            )
            if i == 0:
              # Assuming `output` is the object returned from the model.
              attentions = output.attentions
              cross_attentions = output.cross_attentions

              # Move tensors to CPU before saving
              attentions = [att.cpu() for att in attentions]
              torch.save(attentions, f"attentions_{i}.pt")
              
              # Move cross-attentions to CPU and save
              # print(len(cross_attentions))
              # # for j in range(len(cross_attentions)):
              # #   print(len(cross_attentions[j]))
              if cross_attentions is not None:  # Check if cross-attentions are present
                  cross_attentions = [cross_attentions[j].cpu() for j in range(0,len(cross_attentions),2)]  # Move each tensor to CPU
                  torch.save(cross_attentions, f"cross_attentions_{i}.pt")
              
              print(caption)
              captions = [str(caption[i]) for i in range(len(caption))]
              pair_id = [int(pair_id[i]) for i in range(len(pair_id))]
              print(pair_id)
              # Save as JSON
              data = {
                  "caption": caption,
                  "pair_id": pair_id,
              }

              with open("captions_and_ids.json", "w") as f:
                  json.dump(data, f)
            
            # get attentions
            self_attentions = None
            # cross_attentions = None
            if output.attentions is not None:
                self_attentions = output.attentions
            # if output.cross_attentions is not None:
            #     cross_attentions = output.cross_attentions
            
            # get the weights based on the self attention rollout
            BATCH_INDEX = 0
            weights_queries_per_sample = []
            weights_text_per_sample = []

            self_attentions = [att.detach().clone().cpu() for att in self_attentions]
            self_attentions = torch.stack(self_attentions, dim=0)
            self_attentions = torch.unsqueeze(self_attentions, dim=0)
            for BATCH_SAMPLE in range(len(caption)):
                # copy_attention = self_attentions.copy().detach().cpu()
                # copy_attention  = torch.unsqueeze(copy_attention, dim=0)
                weights_queries, weights_text = compute_weights_based_on_attention_rollout(self_attentions, BATCH_INDEX, BATCH_SAMPLE, temp=0.7, threshold=0.01,  return_logs = False)
                weights_queries_per_sample.append(weights_queries)
                weights_text_per_sample.append(weights_text)
        

            # visual embeddings
            # visual_embs = model.visual_emb_proj(ref_img_embs.mean(dim=1))
            
            # query embeddings
            vl_embs = output.last_hidden_state[:, : query_tokens.size(1), :]
            vl_feat = F.normalize(model.text_proj(vl_embs), dim=-1)
            # query_si_feat = vl_feat.mean(dim=1)
            
            # text embeddings
            txt_emb = output.last_hidden_state[:, query_tokens.size(1) :, :]
            text_si_feat = F.normalize(model.text_proj(txt_emb), dim=-1)
            # text_embs = text_si_feat.mean(dim=1)

            # compute the weighted sum based on the weights from attention (with matrix multiplication)
            weights_queries_per_sample = torch.stack(weights_queries_per_sample, dim=0).to(device)
            weights_text_per_sample = torch.stack(weights_text_per_sample, dim=0).to(device)
            if i == 0:
              print(weights_queries_per_sample[0])
              print(weights_text_per_sample[0])
            query_si_feat = torch.einsum("ij,ijk->ik", weights_queries_per_sample, vl_feat)
            text_embs = torch.einsum("ij,ijk->ik", weights_text_per_sample, text_si_feat)

            # average the query and text embeddings
            vl_feat = (query_si_feat + text_embs) / 2
            if i ==0:
              print(vl_feat.shape)

            # # Weighted embedding combination (multimodal, visual, text)
            # # weights = F.softmax(model.embedding_combination, dim=0)
            # weights = model.embedding_combination
            # vl_feat = weights[0] * query_si_feat + weights[1] * visual_embs + weights[2] * text_embs

            vl_feats.append(vl_feat.cpu())

        pair_ids = torch.tensor(pair_ids, dtype=torch.long)
        vl_feats = torch.cat(vl_feats, dim=0)

        vl_feats = concat_all_gather(vl_feats, fabric)
        pair_ids = concat_all_gather(pair_ids, fabric)

        if fabric.global_rank == 0:
            pair_ids = pair_ids.cpu().numpy().tolist()

            assert len(vl_feats) == len(pair_ids)
            img_ids = [data_loader.dataset.pairid2ref[pair_id] for pair_id in pair_ids]
            assert len(img_ids) == len(pair_ids)

            id2emb = OrderedDict()
            for img_id, target_emb_pth in tqdm(data_loader.dataset.id2embpth.items()):
                if img_id not in id2emb:
                    tar_emb = F.normalize(
                        torch.load(target_emb_pth, weights_only=True).cpu(), dim=-1
                    )
                    # mean over all visual tokens
                    tar_emb = tar_emb.mean(dim=0)
                    id2emb[img_id] = tar_emb

            tar_feats = torch.stack(list(id2emb.values()), dim=0).to("cpu")
            vl_feats = vl_feats.to("cpu")

            # sims_q2t = torch.einsum("ie,je->ij", vl_feats, tar_feats)
            # Process in batches to avoid memory issues
            batch_size = 64
            sims_q2t = []
            for i in tqdm(range(0, vl_feats.size(0), batch_size)):
                vl_feats_batch = vl_feats[i : i + batch_size]
                sim_batch = torch.einsum("ie,je->ij", vl_feats_batch, tar_feats)
                sims_q2t.append(sim_batch)
            
            sims_q2t = torch.cat(sims_q2t, dim=0)

            # sims_q2t = sims_q2t.max(dim=-1)[0]
            # sims_q2t = sims_q2t.max(dim=-1)[0]

            # Create a mapping from pair_id to row index for faster lookup
            pairid2index = {pair_id: i for i, pair_id in enumerate(pair_ids)}

            # Create a mapping from target_id to column index for faster lookup
            tarid2index = {tar_id: j for j, tar_id in enumerate(id2emb.keys())}

            # Update the similarity matrix based on the condition
            for pair_id in tqdm(pair_ids):
                que_id = data_loader.dataset.pairid2ref[pair_id]
                if que_id in tarid2index:
                    sims_q2t[pairid2index[pair_id], tarid2index[que_id]] = -100
            sims_q2t = sims_q2t.cpu().numpy()

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print("Evaluation time {}".format(total_time_str))

            recalls = {}
            recalls["version"] = "rc2"
            recalls["metric"] = "recall"

            recalls_subset = {}
            recalls_subset["version"] = "rc2"
            recalls_subset["metric"] = "recall_subset"

            target_imgs = np.array(list(id2emb.keys()))

            assert len(sims_q2t) == len(pair_ids)
            for pair_id, query_sims in tqdm(zip(pair_ids, sims_q2t)):
                sorted_indices = np.argsort(query_sims)[::-1]

                query_id_recalls = list(target_imgs[sorted_indices][:50])
                recalls[str(pair_id)] = query_id_recalls

                members = data_loader.dataset.pairid2members[pair_id]
                query_id_recalls_subset = [
                    target
                    for target in target_imgs[sorted_indices]
                    if target in members
                ][:3]
                recalls_subset[str(pair_id)] = query_id_recalls_subset

            json_dump(recalls, "recalls_cirr.json")
            json_dump(recalls_subset, "recalls_cirr_subset.json")

            print(f"Recalls saved in {Path.cwd()}/recalls_cirr.json")

        fabric.barrier()
