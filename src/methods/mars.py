import re

import numpy as np
import torch
import torch.nn.functional as F
from flair.data import Sentence
from flair.models import SequenceTagger
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from scipy.special import softmax


tagger = SequenceTagger.load("flair/chunk-english")

VOCAB_PATH = 'gs://cloud-tpu-checkpoints/bert/keras_bert/uncased_L-12_H-768_A-12/vocab.txt'
vocab_table = tf.lookup.StaticVocabularyTable(
    tf.lookup.TextFileInitializer(
        filename=VOCAB_PATH,
        key_dtype=tf.string,
        key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
        value_dtype=tf.int64,
        value_index=tf.lookup.TextFileIndex.LINE_NUMBER
    ), 
    num_oov_buckets=1
)
cls_id, sep_id = vocab_table.lookup(tf.convert_to_tensor(['[CLS]', '[SEP]']))
bert_tokenizer = text.BertTokenizer(
    vocab_lookup_table=vocab_table,
    token_out_type=tf.int64, 
    preserve_unused_token=True, 
    lower_case=True
)

bem = hub.load('https://tfhub.dev/google/answer_equivalence/bem/1')


def check_exist(words, tagged_sentence):
    for i, chunks in enumerate(tagged_sentence):
        if words.replace(" ", "").lower() == chunks.text.replace(" ", "").lower():
            return True, i
    return False, -1


def phrase_tokenizer(sentence):
    tokenized_sentence = []
    tagged_sentence = Sentence(sentence)
    tagger.predict(tagged_sentence)

    words = re.findall(r'\w+|[^\w\s]', sentence)

    i = 0
    positions = []
    while(i < len(words)):
        found = False
        for j in range(i+1, len(words)):
            combined_word = " ".join(words[i:j])
            exist, index = check_exist(combined_word, tagged_sentence.get_spans('np'))
            if exist:
                span = tagged_sentence.get_spans('np')[index]
                tokenized_sentence.append(span.text)
                positions.append((span.start_position, span.end_position))
                found = True
                i = j
                break
        if not found:
            word = words[i]
            start_pos = sentence.find(word, positions[-1][1] if positions else 0)
            end_pos = start_pos + len(word)
            tokenized_sentence.append(word)
            positions.append((start_pos, end_pos))
            i += 1
    return tokenized_sentence, positions


def bertify_example(example):
  question = bert_tokenizer.tokenize(example['question']).merge_dims(1, 2)
  reference = bert_tokenizer.tokenize(example['reference']).merge_dims(1, 2)
  candidate = bert_tokenizer.tokenize(example['candidate']).merge_dims(1, 2)

  input_ids, segment_ids = text.combine_segments(
      (candidate, reference, question), cls_id, sep_id)

  return {'input_ids': input_ids.numpy(), 'segment_ids': segment_ids.numpy()}


def pad(a, length=512):
  return np.append(a, np.zeros(length - a.shape[-1], np.int32))


def bertify_examples(examples):
  input_ids = []
  segment_ids = []
  for example in examples:
    example_inputs = bertify_example(example)
    input_ids.append(pad(example_inputs['input_ids']))
    segment_ids.append(pad(example_inputs['segment_ids']))

  return {'input_ids': np.stack(input_ids), 'segment_ids': np.stack(segment_ids)}


def get_importance_scores(answer_text, question_text):
    importance_scores = []

    print(f"Question: {question_text}")
    print(f"Answer: {answer_text}")

    phrases, positions = phrase_tokenizer(answer_text)
    print(f"Phrases: {phrases}")

    for i in range(len(phrases)):
        removed_answer_text = phrases[:i] + phrases[i+1:]

        removed_answer_text = ' '.join(removed_answer_text)

        bem_input = [{
            'question': question_text,
            'reference': answer_text,
            'candidate': removed_answer_text
        }]

        inputs = bertify_examples(bem_input)
        raw_outputs = bem(inputs)
        bem_score = float(softmax(np.squeeze(raw_outputs))[1])
        score = 1 - bem_score
        importance_scores.append(score)

    importance_scores = np.array(importance_scores)

    return importance_scores, phrases, positions
        

def mars(sample_paths, normalized_length, tokenizer, config):

    method_records = []

    for path in sample_paths:
        question_text = path["prompt"]
        final_answer = path["final_answer"]
        answer_ids = path["answer_ids"]
        answer_text = path["answer_text"]
        output_scores = path["output_scores"]

        probs = F.softmax(output_scores, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)

        importance_scores, phrases, positions = get_importance_scores(answer_text, question_text)
        if importance_scores.size == 0 or len(phrases) != len(positions):
            method_records.append((answer_text, 0.0, final_answer))
            continue
            
        answer_text_tokenized = tokenizer(answer_text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = answer_text_tokenized['offset_mapping']
        phrase_token_spans = []

        for phrase, (phrase_start_idx, phrase_end_idx) in zip(phrases, positions):
            phrase_token_start_idx = None
            phrase_token_end_idx = None

            for i, (s, e) in enumerate(offsets):
                if s <= phrase_start_idx < e:
                    answer_token_start_idx = i
                if s < phrase_end_idx + len(phrase) <= e:
                    answer_token_end_idx = i + 1
                if answer_token_start_idx is not None and answer_token_end_idx is not None:
                    break
            phrase_token_spans.append((answer_token_start_idx, answer_token_end_idx))
        
        if not phrase_token_spans or len(phrase_token_spans) != len(phrase):
            method_records.append((answer_text, 0.0, final_answer))
            continue






        
