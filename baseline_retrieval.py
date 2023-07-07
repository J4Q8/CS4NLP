from numpy import dot
from numpy.linalg import norm
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util

def split_text_into_sentences(text):
  text = text.replace("\n", "") # replace linebreaks
  sentences = text.split(". ") # split sentences
  sentences = [string for string in sentences if string] # remove empty strings ""
  sentences = sentences[:-1] if not sentences[-1].strip() else sentences # make sure last sentece is not empty
  sentences = [sentence if sentence.endswith(".") else sentence + ". " for sentence in sentences] # last sentence usually just ends with "." instead of ". ", do not add delimiter for them
  return sentences

def split_text_into_chunks(text, max_token_size):
    
    # support spliting into sentences as well
    if max_token_size == "sentence":
       return split_text_into_sentences(text)

    text = text.replace("\n", "") # replace linebreaks
    sentences = text.split(". ") # split sentences
    sentences = [string for string in sentences if string] # remove empty strings ""
    sentences = [sentence if sentence.endswith(".") else sentence + ". " for sentence in sentences] # last sentence usually just ends with "." instead of ". ", do not add delimiter for them

    chunks = []
    current_chunk = ""
    current_chunk_size = 0

    for sentence in sentences:
        sentence_size =  len(sentence.split())
        if current_chunk_size +sentence_size < max_token_size:
            current_chunk += sentence
            current_chunk_size += sentence_size
        else:
            chunks.append(current_chunk)
            current_chunk = sentence
            current_chunk_size = sentence_size

    if current_chunk: # add last element
        chunks.append(current_chunk)

    return chunks

def random_sentence_cut(article, tokenizer, MAX_TOKENS=512, extra_length = 0, chunk_size = 256, *args, **kwargs):

  MAX_TOKENS = MAX_TOKENS - extra_length
  
  sentences = split_text_into_chunks(article, chunk_size)

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
  final_sentences = [sentences[i] for i in selected_sentences]

  return " ".join(final_sentences), selected_sentences


def start_ending_biased_sentece_cut(article, tokenizer, MAX_TOKENS=512, extra_length = 0, chunk_size = 256, *args, **kwargs):

  MAX_TOKENS = MAX_TOKENS - extra_length

  sentences = split_text_into_chunks(article, chunk_size)
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
  final_sentences = [sentences[i] for i in selected_sentences]

  return " ".join(final_sentences), selected_sentences


def tf_idf_sentece_cut(article, tokenizer, query, MAX_TOKENS = 512, extra_length = 0, chunk_size = 256, *args, **kwargs):

  MAX_TOKENS = MAX_TOKENS - extra_length

  sentences = split_text_into_chunks(article, chunk_size)
  num_sentences = len(sentences)

  # tf_idf
  vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(4,6))
  tf_idf = vectorizer.fit_transform(sentences)
  query_vector = vectorizer.transform([query])
  cos_sim = lambda a,b : dot(a, b)/(norm(a)*norm(b))
  cosine_similarities = np.array([cos_sim(tf_idf[i].toarray().flatten(), query_vector.toarray().flatten()) for i in range(num_sentences)])
  sentence_list = np.argsort(cosine_similarities)[::-1]

  assert len(sentence_list) == num_sentences

  selected_sentences = []
  total_tokens = 0

  # get the closest sentences to tf_idf
  for sentence_idx in sentence_list:
    tokens = tokenizer.tokenize(sentences[sentence_idx] + ".")
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
  final_sentences = [sentences[i] for i in selected_sentences]

  return " ".join(final_sentences), selected_sentences


def sentence_embedding_cut(article, tokenizer, query, MAX_TOKENS = 512, extra_length = 0, chunk_size = 256, sentembb_model = None, *args, **kwargs):
    MAX_TOKENS = MAX_TOKENS - extra_length

    sentences = split_text_into_chunks(article, chunk_size)

    query_embedding = sentembb_model.encode(query)

    batch_size = 500
    num_sentences = len(sentences)
    num_batches = int(np.ceil(num_sentences / batch_size))

    similarity_scores = []

    # get all passage embeddings in batches
    for i in range(num_batches):
      start_index = i * batch_size
      end_index = min((i + 1) * batch_size, num_sentences)

      # Get the batch of sentences
      batch_sentences = sentences[start_index:end_index]

      # Encode the batch of sentences into embeddings
      batch_embeddings = sentembb_model.encode(batch_sentences)

      # Append the batch embeddings to the list
      #passage_embedding.extend(batch_embeddings)

      similarity = util.cos_sim(query_embedding, batch_embeddings).numpy()[0]
      similarity_scores.extend(similarity)
    #passage_embedding = sentembb_model.encode(sentences)

    #similarity = util.cos_sim(query_embedding, passage_embedding).numpy()[0]
    #print("Similarity:", similarity_scores)

    result = list(zip(range(0, len(sentences)), similarity_scores))

    # sort them by similarity score
    sentences_sortby_similarity = sorted(result, key=lambda x: x[1], reverse=True)
    #print(sentences_sortby_similarity)


    selected_sentences = []
    total_tokens = 0

    for (sentence_idx, similarity) in sentences_sortby_similarity:
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
    final_sentences = [sentences[i] for i in selected_sentences]

    return " ".join(final_sentences), selected_sentences