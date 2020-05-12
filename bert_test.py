'''
Created on 2019. 11. 18.

@author: chiho
'''
bert_base, vocabulary = nlp.model.get_model('bert_12_768_12',
                                             dataset_name='wiki_multilingual_cased',
                                             pretrained=True, ctx=ctx, use_pooler=True,
                                             use_decoder=False, use_classifier=False)

ds = gluon.data.SimpleDataset([['나 보기가 역겨워', '김소월']])
tok = nlp.data.BERTTokenizer(vocab=vocabulary, lower=False)
trans = nlp.data.BERTSentenceTransform(tok, max_seq_length=10)
list(ds.transform(trans)
[(array([    2,  8982,  9356, 47869,  9566,     3,  8935, 22333, 38851,
             3], dtype=int32),
  array(10, dtype=int32),
  array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=int32))]
  
class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)
        sent_dataset = gluon.data.SimpleDataset([[
            i[sent_idx],
        ] for i in dataset])
        self.sentences = sent_dataset.transform(transform)
        self.labels = gluon.data.SimpleDataset(
            [np.array(np.int32(i[label_idx])) for i in dataset])

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))
    
class BERTClassifier(nn.Block):
    def __init__(self,
                 bert,
                 num_classes=2,
                 dropout=None,
                 prefix=None,
                 params=None):
        super(BERTClassifier, self).__init__(prefix=prefix, params=params)
        self.bert = bert
        with self.name_scope():
            self.classifier = nn.HybridSequential(prefix=prefix)
            if dropout:
                self.classifier.add(nn.Dropout(rate=dropout))
            self.classifier.add(nn.Dense(units=num_classes))

    def forward(self, inputs, token_types, valid_length=None):
        _, pooler_out = self.bert(inputs, token_types, valid_length)
        return self.classifier(pooler_out)      