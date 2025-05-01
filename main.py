import re

def getWordTagCount(split_sentences):
  word_count = {}
  tag_count = {}

  for i in range(len(split_sentences)):
    split_sentence = split_sentences[i]
    for word in split_sentence:
      word_tag = word.split('_')
      word = word_tag[0]
      tag = word_tag[1]

      if word not in word_count:
        word_count[word] = 1
      else:
        word_count[word] += 1

      if tag not in tag_count:
        tag_count[tag] = 1
      else:
        tag_count[tag] += 1
  
  return word_count, tag_count

def getTransitionProbabilities(tag_count, split_sentences):
  trnsmn_prob = []

  for tag in tag_count:
    if (tag == 'END'):
      continue
    else: 
      row = {}
      trnsn = {}
      for i in range(len(split_sentences)):
        for j in range(len(split_sentences[i])):
          word_pos = split_sentences[i][j].split('_')
          word_tag = word_pos[1]
          if (re.match(tag, word_tag)):
            if (j != len(split_sentences[i]) - 1):
              next_word_pos = split_sentences[i][j+1].split('_')
              next_word_tag = next_word_pos[1]
              trnsn[next_word_tag] = trnsn.get(next_word_tag, 0) + 1
              row[word_tag] = trnsn
              print(f"{word_tag} => {next_word_tag}")
      for tag in row:
        # print(f"{tag} => {row[tag]}")
        for next_tag in row[tag]:
          trnsmn_prob.append((tag, next_tag, row[tag][next_tag] / tag_count[tag]))
  
  return trnsmn_prob

def getEmissionProbabilities(tag_count, split_sentences):
  emsn_prob = []

  for tag in tag_count:
    if (tag == 'START' or tag == 'END'):
      continue
    else:
      row = {}
      for i in range(len(split_sentences)):
        for j in range(len(split_sentences[i])):
          word_pos = split_sentences[i][j].split('_')
          word_tag = word_pos[1]
          if (re.match(tag, word_tag)):
            word = word_pos[0]
            if (word not in row):
              row[word] = 1
            else:
              row[word] += 1
      for word in row:
        emsn_prob.append((tag, word, row[word] / tag_count[tag]))

  return emsn_prob

def main():
  x_train = [
    'The_DET cat_NOUN sleeps_VERB',
    'A_DET dog_NOUN barks_VERB',
    'The_DET dog_NOUN sleeps_VERB',
    'My_DET dog_NOUN runs_VERB fast_ADV',
    'A_DET cat_NOUN meows_VERB loudly_ADV',
    'Your_DET cat_NOUN runs_VERB',
    'The_DET bird_NOUN sings_VERB sweetly_ADV',
    'A_DET bird_NOUN chirps_VERB'
  ]

  x_train_split = [['<s>_START'] + sentence.split(' ') + ['<e>_END'] for sentence in x_train]

  word_count, tag_count = getWordTagCount(x_train_split)
  trnsmn_prob = getTransitionProbabilities(tag_count, x_train_split)
  emsn_prob = getEmissionProbabilities(tag_count, x_train_split)

  print(tag_count, '\n')
  print(trnsmn_prob, '\n')
  print(emsn_prob)

if __name__ == '__main__':
  main()