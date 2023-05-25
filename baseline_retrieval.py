from numpy import dot
from numpy.linalg import norm
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer

# TODO: make sure that the tokens used for questions, answers and special are accounted for in max token

def random_sentence_cut(article, tokenizer, MAX_TOKENS=512, extra_length = 0, *args, **kwargs):

  MAX_TOKENS = MAX_TOKENS - extra_length
 
  sentences = article.split(". ")

  # get the permutation of the sentences
  num_sentences = len(sentences)
  sentence_list = list(range(num_sentences))
  random.shuffle(sentence_list)

  selected_sentences = []
  total_tokens = 0

  # get a cut of senteces that is MAX_TOKENS long or less
  for sentence_idx in sentence_list:
    tokens = tokenizer.tokenize(sentences[sentence_idx])
    num_tokens = len(tokens)
    if total_tokens == MAX_TOKENS:
      break
    elif (total_tokens + num_tokens) <= MAX_TOKENS:
      selected_sentences.append(sentence_idx)
      total_tokens += num_tokens

  # use the senteces in the original order
  selected_sentences.sort()
  selected_sentences = [sentences[i] for i in selected_sentences]

  return " ".join(selected_sentences)


def start_ending_biased_sentece_cut(article, tokenizer, MAX_TOKENS=512, extra_length = 0, *args, **kwargs):

  MAX_TOKENS = MAX_TOKENS - extra_length

  sentences = article.split(". ")
  num_sentences = len(sentences)
  sentence_list = list(range(num_sentences))

  # get probability distribution for the senteces which is biased towards the start and end of the article
  quadratic = lambda x : 0.1*(x - (num_sentences/2))**2 + 1 # strictly positive quadratic function with minimum at the middle of the article
  sentence_probs = np.array([quadratic(i) for i in range(num_sentences)])
  sentence_probs = sentence_probs/np.sum(sentence_probs)

  sentence_list = np.random.choice(sentence_list, size=num_sentences, replace=False, p=sentence_probs)

  selected_sentences = []
  total_tokens = 0

  for sentence_idx in sentence_list:
    tokens = tokenizer.tokenize(sentences[sentence_idx])
    num_tokens = len(tokens)
    if total_tokens == MAX_TOKENS:
      break
    elif (total_tokens + num_tokens) <= MAX_TOKENS:
      selected_sentences.append(sentence_idx)
      total_tokens += num_tokens

  # use the senteces in the original order
  selected_sentences.sort()
  selected_sentences = [sentences[i] for i in selected_sentences]

  return " ".join(selected_sentences)


def tf_idf_sentece_cut(article, tokenizer, query, MAX_TOKENS = 512, extra_length = 0, *args, **kwargs):

  MAX_TOKENS = MAX_TOKENS - extra_length

  sentences = article.split(". ")
  num_sentences = len(sentences)

  # tf_idf
  vectorizer = TfidfVectorizer()
  tf_idf = vectorizer.fit_transform(sentences)
  query_vector = vectorizer.transform([query])
  cos_sim = lambda a,b : dot(a, b)/(norm(a)*norm(b))
  cosine_similarities = np.array([cos_sim(tf_idf[i].toarray(), query_vector) for i in range(num_sentences)])
  sentence_list = np.argsort(cosine_similarities)[::-1]

  assert len(sentence_list) == num_sentences

  selected_sentences = []
  total_tokens = 0

  # get the closest sentences to tf_idf
  for sentence_idx in sentence_list:
    tokens = tokenizer.tokenize(sentences[sentence_idx])
    num_tokens = len(tokens)
    if total_tokens == MAX_TOKENS:
      break
    elif (total_tokens + num_tokens) <= MAX_TOKENS:
      selected_sentences.append(sentence_idx)
      total_tokens += num_tokens
    else:
      break

  # use the senteces in the original order
  selected_sentences.sort()
  selected_sentences = [sentences[i] for i in selected_sentences]

  return " ".join(selected_sentences)