import re
from collections import defaultdict

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

class HiddenMarkovModel:
  def __init__(self, tag_count, split_sentences):
    self.tag_count = tag_count
    self.split_sentences = split_sentences
    self.trnsmn_probs = []
    self.emsn_probs = []

  def build(self):
    self.__transmission_probabilities()
    self.__emission_probabilities()
  
  def viterbi(self, observations):
    states = [tag for tag in self.tag_count if tag != 'START' and tag != 'END']

    # Convert transmission and emission probability lists to lookup dictionaries
    trans_probs = defaultdict(lambda: defaultdict(lambda: 1e-6))
    for from_tag, to_tag, prob in self.trnsmn_probs:
        trans_probs[from_tag][to_tag] = prob

    emit_probs = defaultdict(lambda: defaultdict(lambda: 1e-6))
    for tag, word, prob in self.emsn_probs:
        emit_probs[tag][word] = prob

    # Initialize Viterbi table
    V = [{}]
    path = {}

    for state in states:
        trans_prob = trans_probs['START'].get(state, 1e-6)
        emit_prob = emit_probs[state].get(observations[0], 1e-6)
        V[0][state] = trans_prob * emit_prob
        path[state] = [state]

    # Run Viterbi for t > 0
    for t in range(1, len(observations)):
        V.append({})
        new_path = {}

        for curr_state in states:
            max_prob, best_prev_state = max(
                (V[t-1][prev_state] * trans_probs[prev_state].get(curr_state, 1e-6) *
                 emit_probs[curr_state].get(observations[t], 1e-6), prev_state)
                for prev_state in states
            )
            V[t][curr_state] = max_prob
            new_path[curr_state] = path[best_prev_state] + [curr_state]

        path = new_path

    # Final transition to END
    max_prob, best_last_state = max(
        (V[len(observations)-1][state] * trans_probs[state].get('END', 1e-6), state)
        for state in states
    )

    return path[best_last_state]
  
  def __transmission_probabilities(self):
    for tag in self.tag_count:
      if (tag == 'END'): # To prevent END tag from being used as a transmission
        continue
      else: 
        transmissions = {}
        next_tag_count = {}
        for i in range(len(self.split_sentences)): # For each sentence
          for j in range(len(self.split_sentences[i])): # For each word in the sentence
            word_pos = self.split_sentences[i][j].split('_') # Split the word and tag
            word_tag = word_pos[1]
            if (re.match(tag, word_tag)): # If the tag matches
              if (j != len(self.split_sentences[i]) - 1): # If not the last word
                # Get the next word and its tag
                next_word_pos = self.split_sentences[i][j+1].split('_') 
                next_word_tag = next_word_pos[1]
                next_tag_count[next_word_tag] = next_tag_count.get(next_word_tag, 0) + 1 
                transmissions[word_tag] = next_tag_count # Add the next tag count to the transmissions dictionary
                # print(f"{word_tag} => {next_word_tag}")

        for tag in transmissions:
          # print(f"{tag} => {transmissions[tag]}")
          for next_tag in transmissions[tag]:
            # Calculate the transmission probability
            self.trnsmn_probs.append((tag, next_tag, transmissions[tag][next_tag] / self.tag_count[tag])) 
    return

  def __emission_probabilities(self):
    for tag in self.tag_count:
      if (tag == 'START' or tag == 'END'): # To prevent START and END tags from being used as emissions
        continue
      else:
        emissions = {}
        for i in range(len(self.split_sentences)): # For each sentence
          for j in range(len(self.split_sentences[i])): # For each word in the sentence
            word_pos = self.split_sentences[i][j].split('_') # Split the word and tag
            word_tag = word_pos[1]
            if (re.match(tag, word_tag)): 
              word = word_pos[0]
              if (word not in emissions): # If the word is not in the emissions dictionary
                emissions[word] = 1 # Add the word to the emissions dictionary
              else:
                emissions[word] += 1 # Increment the count of the word in the emissions dictionary

        for word in emissions:
          # print(f"{tag} => {word}: {emissions[word] / self.tag_count[tag]}")
          # Calculate the emission probability
          self.emsn_probs.append((tag, word, emissions[word] / self.tag_count[tag]))
    return
  

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

  x_test = [
    'The cat meows',
    'My dog barks loudly',
  ]

  x_train_split = [['<s>_START'] + sentence.split(' ') + ['<e>_END'] for sentence in x_train]
  word_count, tag_count = getWordTagCount(x_train_split)

  hmm = HiddenMarkovModel(tag_count, x_train_split)
  hmm.build()

  for sentence in x_test:
    words = sentence.split(' ')
    predicted_tags = hmm.viterbi(words)
    print(f"Sentence: {sentence}")
    print(f"Predicted tags: {predicted_tags}\n")

if __name__ == '__main__':
  main()