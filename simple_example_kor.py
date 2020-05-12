#coding: utf-8
'''
Created on 2019. 10. 30.

@author: tobew
conda create -n bert_ve37 python=3.7
pip install --pre --upgrade mxnet-cu100
pip install gluonnlp

https://gist.github.com/haven-jeon/3d7c538398e93dab2ed4899159a5d943
Vocab file is not found. Downloading.
https://github.com/apache/incubator-mxnet/issues/4431
https://gluon-nlp.mxnet.io/
'''
import pandas as pd
import numpy as np
from mxnet.gluon import nn, rnn
from mxnet import gluon, autograd
import gluonnlp as nlp
from mxnet import nd 
import mxnet as mx
import time
import itertools
import random


class BERTDataset(mx.gluon.data.Dataset):
    # BERT embedding�쓣 �씠�슜�븳 �뜲�씠�꽣�뀑 留뚮뱾湲� 
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
    # BERT embedding�쓣 �씠�슜�븳 遺꾨쪟湲� �겢�옒�뒪 
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
        _, pooler = self.bert(inputs, token_types, valid_length)
        return self.classifier(pooler)
    
def main():
    # gpu �궗�슜�븯�룄濡� �꽕�젙 
    # pip install --pre --upgrade mxnet-cu100
    # �쐞�쓽 option�뿉�꽌 cuda library 踰꾩쟾�쓣 留욎떠�빞 �븿...  cuda library 踰꾩쟾10.0 �씤 寃쎌슦,  9.0 �씠硫�  cu90 �씠�윴 �떇�쑝濡� 諛붽퓭�빞 �븿 
    ctx = mx.gpu() 

    #mxnet.base.MXNetError: [07:43:15] C:\Jenkins\workspace\mxnet\mxnet\src\ndarray\ndarray.cc:1295: GPU is not enabled
    bert_base, vocabulary = nlp.model.get_model('bert_12_768_12',
                                                 dataset_name='wiki_multilingual_cased',
                                                 pretrained=True, ctx=ctx, use_pooler=True,
                                                 use_decoder=False, use_classifier=False)
    # bert 援ъ“ �솗�씤�슜
    print("bert model check")
    print(bert_base)
    print('---------')
    # �떒�닚�븳 �뜲�씠�꽣瑜� bert瑜� �씠�슜�븯�뿬 蹂��솚 諛� 異쒕젰�븳 �삁, �솗�씤�슜
#     ds = gluon.data.SimpleDataset([['�굹 蹂닿린媛� �뿭寃⑥썙', '源��냼�썡']])    
#     tok = nlp.data.BERTTokenizer(vocab=vocabulary, lower=False)    
#     trans = nlp.data.BERTSentenceTransform(tok, max_seq_length=10)
#     
#     print(list(ds.transform(trans)))  # bert瑜� �씠�슜�븯�뿬 蹂��솚...   
#         
        
    #https://github.com/e9t/nsmc  
    #https://github.com/e9t/nsmc  
    # tsv �뙆�씪�뿉�꽌  0,1,2 而щ읆�뿉�꽌  1�씠 臾몄옣, 2sms label�엫
    # �뜲�씠�꽣�뀑 �깮�꽦  
    dataset_train = nlp.data.TSVDataset("ratings_train.txt", field_indices=[1,2], num_discard_samples=1)
    dataset_test = nlp.data.TSVDataset("ratings_test.txt", field_indices=[1,2], num_discard_samples=1)
    
    # bert tokenizer �깮�꽦 
    bert_tokenizer = nlp.data.BERTTokenizer(vocabulary, lower=False)
    # 湲몄씠�뿉 �뵲�씪 �궗�슜�븯�뒗 硫붾え由ъ쓽 �겕湲곌� �떖�씪吏� 
    # 硫붾え由ш� 遺�議깊븳 寃쎌슦 max_len 湲몄씠瑜� �뒛�뿬�빞 �븿.. 二쇱쓽!!! 
#     max_len = 64
    max_len = 32
    data_train = BERTDataset(dataset_train, 0, 1, bert_tokenizer, max_len, True, False)
    data_test = BERTDataset(dataset_test, 0, 1, bert_tokenizer, max_len, True, False)
    model = BERTClassifier(bert_base, num_classes=2, dropout=0.3)
    # 遺꾨쪟湲� 珥덇린�솕, gpu 吏��젙 
    model.classifier.initialize(ctx=ctx)
    model.hybridize()
    
    # softmax cross entropy loss for classification
    loss_function = gluon.loss.SoftmaxCELoss()
    # 泥숇룄瑜� �젙�쓽    
    metric = mx.metric.Accuracy()
    
    # 硫붾え由ш� 遺�議깊븳 寃쎌슦 batch_size瑜� 以꾩뿬�빞 �븳�떎. 
#     batch_size = 64
    batch_size = 32
    lr = 5e-5
    # �뜲�씠�꽣�뀑�쑝濡� 遺��꽣 �뜲�씠�꽣瑜� load, 蹂묐젹泥섎━瑜� �쐞�빐�꽌 num_worker瑜� 5濡� 吏��젙 
    train_dataloader = mx.gluon.data.DataLoader(data_train, batch_size=batch_size, num_workers=5)
    test_dataloader = mx.gluon.data.DataLoader(data_test, batch_size=batch_size, num_workers=5)
    # �븰�뒿�쓣 �쐞�븳 hyper parameter �꽕�젙 
    trainer = gluon.Trainer(model.collect_params(), 'bertadam',
                            {'learning_rate': lr, 'epsilon': 1e-9, 'wd':0.01})
    
    log_interval = 4
    num_epochs = 4
    # LayerNorm Bias Weight Decay
    for _, v in model.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0
    params = [
        p for p in model.collect_params().values() if p.grad_req != 'null'
    ]
    
    # �젙�솗�룄 �룊媛�瑜� �쐞�븳 �궡遺� �븿�닔 
    def evaluate_accuracy(model, data_iter, ctx=ctx):
        acc = mx.metric.Accuracy()
        i = 0
        for i, (t,v,s, label) in enumerate(data_iter):
            token_ids = t.as_in_context(ctx)
            valid_length = v.as_in_context(ctx)
            segment_ids = s.as_in_context(ctx)
            label = label.as_in_context(ctx)
            output = model(token_ids, segment_ids, valid_length.astype('float32'))
            acc.update(preds=output, labels=label)
            if i > 1000:
                break
            i += 1
        return(acc.get()[1])
    
    # �븰�뒿�쓣 �쐞�븳 �뙣�윭誘명꽣 �꽕�젙 遺�遺�
    #learning rate warmup 
    step_size = batch_size 
    num_train_examples = len(data_train)
    num_train_steps = int(num_train_examples / step_size * num_epochs)
    warmup_ratio = 0.1
    num_warmup_steps = int(num_train_steps * warmup_ratio)
    step_num = 0
    
    # �븰�뒿 �닔�뻾 遺�遺� 
    for epoch_id in range(num_epochs):
        metric.reset()
        step_loss = 0
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(train_dataloader):
            step_num += 1
            if step_num < num_warmup_steps:
                new_lr = lr * step_num / num_warmup_steps
            else:
                offset = (step_num - num_warmup_steps) * lr / (
                    num_train_steps - num_warmup_steps)
                new_lr = lr - offset
            trainer.set_learning_rate(new_lr)
            with mx.autograd.record():
                # load data to GPU
                token_ids = token_ids.as_in_context(ctx)
                valid_length = valid_length.as_in_context(ctx)
                segment_ids = segment_ids.as_in_context(ctx)
                label = label.as_in_context(ctx)
    
                # forward computation
                out = model(token_ids, segment_ids, valid_length.astype('float32'))
                ls = loss_function(out, label).mean()
    
            # backward computation
            ls.backward()
            trainer.allreduce_grads()
            nlp.utils.clip_grad_global_norm(params, 1)
            trainer.update(token_ids.shape[0])
    
            step_loss += ls.asscalar()
            metric.update([label], [out])
            if (batch_id + 1) % (50) == 0:
                print('[Epoch {} Batch {}/{}] loss={:.4f}, lr={:.10f}, acc={:.3f}'
                             .format(epoch_id + 1, batch_id + 1, len(train_dataloader),
                                     step_loss / log_interval,
                                     trainer.learning_rate, metric.get()[1]))
                step_loss = 0
        test_acc = evaluate_accuracy(model, test_dataloader, ctx)
        # epoch 留덈떎 �뀒�뒪�듃 �젙�솗�룄瑜� 異쒕젰 
        print('Test Acc : {}'.format(test_acc))
        
        
if __name__=='__main__':
    main()      
    
    
    
