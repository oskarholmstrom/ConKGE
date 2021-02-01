from knowledge_graph import KnowledgeGraph

PAD = '[PAD]'
UNK = '[UNK]'
MASK_TOKEN = '[MASK]'
CLS_TOKEN = '[CLS]'

PAD_INDEX = 0

class GraphDataset(object):
    """
    Reads a text file where triples (head, relation, object) are present on each line.
    Converts the triples into a knowledge graph and creates a vocab for entities and relations.
    """

    def __init__(self, name, train_file, test_file=None, valid_file=None):

        self.name = name
        self.element_vocab = {PAD: 0, UNK: 1, MASK_TOKEN: 2, CLS_TOKEN: 3}

        self.head_rel = {} # Keys are "head, relation", values are a list of all possible tails for a head-relation pair.
        self.tail_rel = {} # Keys are "tail, relation", values are a list of all possible heads for a tail-relation pair.

        self.train_graph = self.read_data(train_file)

        if test_file != None:
            self.test_graph = self.read_data(test_file)
        if valid_file != None: 
            self.valid_graph = self.read_data(valid_file)


    def read_data(self, path):
        file = open(path)

        kg = KnowledgeGraph()

        with open(path) as fp:
            line = fp.readline()
            while line:
                triple = line.split()

                # Add triples to the knowledge graph
                kg.add_triple(triple)
                kg.num_triples += 1

                for element in triple:
                    if element not in self.element_vocab:
                        self.element_vocab[element] = len(self.element_vocab)

                head_rel = str(self.element_vocab[triple[0]]) + ', ' + str(self.element_vocab[triple[1]])
                if head_rel not in self.head_rel:
                    self.head_rel[head_rel] = []

                self.head_rel[head_rel].append(self.element_vocab[triple[2]])

                # Add tail and relation pairs to the vocab
                tail_rel = str(self.element_vocab[triple[2]]) + ', ' + str(self.element_vocab[triple[1]])
                if tail_rel not in self.tail_rel:
                    self.tail_rel[tail_rel] = []

                self.tail_rel[tail_rel].append(self.element_vocab[triple[0]])
                
                line = fp.readline()

        return kg