from transformers import BertTokenizer, BertForMaskedLM
import torch
import numpy as np


def test():

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased', return_dict=True)

    inputs = tokenizer("The capital of France is [MASK].", return_tensors="pt")
    labels = tokenizer("The capital of France is Paris.", return_tensors="pt")["input_ids"]

    print(inputs)
    print(labels)
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    logits = outputs.logits
    print(loss)
    print(len(logits[0]))
    argmax = torch.argmax(logits[0][4])
    
    logits_np = logits[0][4].detach().numpy()
    argsorted = np.argsort(logits_np)
    print(argmax)
    print(argsorted)
    if argmax.item() in argsorted[::-1][:10]:
        print("argmax is found")
    else:
        print("argmax is not found")


if __name__ == '__main__':
    test()