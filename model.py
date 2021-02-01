
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertConfig, BertForMaskedLM

class ConKGE(nn.Module):
    """
    Class for creating a contextual knowledge graph embedding.
    Passes a graph sequence of entity and relationembeddings into a BERT-model.
    """

    #def __init__(self, config, num_ent, num_rel):
    def __init__(self, config, num_elem):
        super().__init__()
        self.element_embedding = nn.Embedding(num_elem, config.hidden_size, padding_idx = 0)

        #self.ent_embedding = nn.Embedding(num_ent, config.hidden_size, padding_idx=0)
        #self.rel_embedding = nn.Embedding(num_rel, config.hidden_size, padding_idx=0)
        #config.vocab_size = num_ent + num_rel
        config.vocab_size = num_elem
        self.bert_mlm = BertForMaskedLM(config=config)


    def forward(
                self,
                args,
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                labels=None
                ):

        """
        if args.triples_head_tail:
            no_entity_nodes = 3
            no_rel_nodes = 1 
        if args.path_length > 0:
            no_entity_nodes = 3 + args.path_length
            no_rel_nodes = 1 + args.path_length

        # Input_ids are structured as follow: [[CLS], Entity tokens, .., Relation tokens, ..]
        ent_embeddings = self.ent_embedding(input_ids[: ,:no_entity_nodes])
        rel_embeddings = self.rel_embedding(input_ids[:, no_entity_nodes:no_entity_nodes + no_rel_nodes])

        inputs_embeds = torch.cat([ent_embeddings, rel_embeddings], dim=1)
        """
        inputs_embeds = self.element_embedding(input_ids)
        
        # Output is a list of [loss, logits]
        outputs = self.bert_mlm(
                                input_ids=None,
                                attention_mask=attention_mask,
                                token_type_ids=None,
                                position_ids=position_ids,
                                head_mask=None,
                                inputs_embeds=inputs_embeds,
                                labels = labels # lm_labels is deprecated but needed to run on cloud server. For current versions of HugginFace Trnsfm. lm_labels => labels
                                )

        return outputs


