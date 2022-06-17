# -*- coding: utf-8 -*-

import argparse
import os
import pickle
import tensorflow as tf

from models.bert_slot_model import BertSlotModel
from to_array.tokenizationK import FullTokenizer
from to_array.bert_to_array import BERTToArray


# read command-line parameters
parser = argparse.ArgumentParser('Evaluating the BERT / ALBERT NLU model')
parser.add_argument('--model', '-m', help = 'Path to BERT / ALBERT NLU model', type = str, required = True)
parser.add_argument('--type', '-tp', help = 'bert or albert', type = str, default = 'bert', required = False)


VALID_TYPES = ['bert', 'albert']

args = parser.parse_args()
load_folder_path = args.model
type_ = args.type

# this line is to disable gpu
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

config = tf.ConfigProto(intra_op_parallelism_threads=1,
                        inter_op_parallelism_threads=1,
                        allow_soft_placement=True,
                        device_count = {'CPU': 1})
sess = tf.compat.v1.Session(config=config)

if type_ == 'bert':
    bert_model_hub_path = '/content/drive/MyDrive/bert-module'
    is_bert = True
    
elif type_ == 'albert':
    bert_model_hub_path = 'https://tfhub.dev/google/albert_base/1'
    is_bert = False
    
else:
    raise ValueError('type must be one of these values: %s' % str(VALID_TYPES))


bert_vocab_path = os.path.join(bert_model_hub_path, 'assets/vocab.korean.rawtext.list')
bert_vectorizer = BERTToArray(is_bert, bert_vocab_path)

with open(os.path.join(load_folder_path, 'tags_to_array.pkl'), 'rb') as handle:
  tags_vectorizer = pickle.load(handle)
  slots_num = len(tags_vectorizer.label_encoder.classes_)

# 모델 불러오기
model = BertSlotModel.load(load_folder_path, sess)

# 토크나이저 불러오기
tokenizer = FullTokenizer(vocab_file=bert_vocab_path)

while True:
    print('\nEnter your sentence: ')
    try:
        input_text = input().strip()

        #input_text_arr = input_text.splitlines()
        
    except:
        continue
        
    if input_text in ['quit', '종료', '그만', '멈춰', 'stop']:
      break
    else : 
      #input_text_arr = BERTToArray.__to_array(input_text)
      #text_arr = bert_to_array.tokenizer.tokenize(input_text)
      text_arr = tokenizer.tokenize(input_text)
      text_arr = [' '.join(text_arr)]
      input_ids, input_mask, segment_ids = bert_vectorizer.transform(text_arr)
      inferred_tags, slot_score = model.predict_slots([input_ids, input_mask, segment_ids], tags_vectorizer)

      print(text_arr)
      print(inferred_tags)
      print(slot_score)


tf.compat.v1.reset_default_graph()

