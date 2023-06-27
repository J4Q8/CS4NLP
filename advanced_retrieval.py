from sentence_transformers import SentenceTransformer, util # pip install -U sentence-transformers

def sentence_embedding_cut(article, tokenizer, query, MAX_TOKENS = 512, extra_length = 0, *args, **kwargs):
    model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
    MAX_TOKENS = MAX_TOKENS - extra_length

    sentences = article.split(". ")
    sentences = sentences[:-1] if not sentences[-1].strip() else sentences
    sentences = [sentence + "." for sentence in sentences]


    query_embedding = model.encode(query)
    passage_embedding = model.encode(sentences)

    similarity = util.dot_score(query_embedding, passage_embedding).numpy()[0]
    #print("Similarity:", similarity)

    result = list(zip(range(0, len(sentences)), similarity))

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
    selected_sentences = [sentences[i] for i in selected_sentences]

    return " ".join(selected_sentences)