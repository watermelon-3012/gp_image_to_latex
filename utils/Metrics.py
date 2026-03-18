import collections
import math

'''
This is the Levenshtein  distance. Levenshtein  distance is used to measure the difference between two sequences. The Levenshtein distance between two words
is the minimum number of single-character edits (insertions, deletions or substitutions) required to change one word into the other
'''
def lev_distance(sequence_one, sequence_two):
    rows = len(sequence_one)
    cols = len(sequence_two)
    ##dist_tab is a table. We will do some operations on this later
    dist_tab = np.zeros((rows + 1, cols + 1), dtype=int)
    ##here we are initializing the elements inside the table first. We will soon change the numbers in some entries after this.
    for i in range(1, rows + 1):
      dist_tab[i][0] = i
    for i in range(1, cols + 1):
      dist_tab[0][i] = i
    for r in range(1, rows + 1):
      for c in range(1, cols + 1):

        #if tokens match
        if sequence_one[r - 1] == sequence_two[c - 1]:
          #if tokens match, we keep the min-cost the same
          #same cost as min cost from prev tokens
          dist_tab[r][c] = dist_tab[r - 1][c - 1]
        else:

          #min of deletion, insertion, or substitution respectively
          dist_tab[r][c] = 1 + min(dist_tab[r - 1][c], dist_tab[r][c - 1], dist_tab[r - 1][c - 1])
    return dist_tab[rows][cols] # Return the raw Levenshtein distance

def compute_levenshtein_metrics(predicted, ground_truth):
    dist_list = []
    norm_list = []

    for pred, true in zip(predicted, ground_truth):
        dist = lev_distance(pred, true)   # token-level
        dist_list.append(dist)

        max_len = max(len(pred), len(true))
        norm = dist / max_len if max_len > 0 else 1
        norm_list.append(norm)

    avg_dist = np.mean(dist_list)
    avg_norm = np.mean(norm_list)
    return avg_dist, avg_norm

def bleu_n_score(generated_sequence, true_sequence, n):
    gen_len = len(generated_sequence)
    true_len = len(true_sequence)
    scores = []

    if gen_len == 0:
      return 0.0

    #calculate overlap for 1-grams to n-grams
    for gram_size in range(1,n+1):

      #generate grams
      gen_ngrams = [tuple(generated_sequence[i:i+gram_size]) for i in range(gen_len - gram_size + 1)]
      true_ngrams = [tuple(true_sequence[i:i+gram_size]) for i in range(true_len - gram_size + 1)]

      gen_grams_count = collections.Counter(gen_ngrams) #freq dictionaries of grams
      true_grams_count = collections.Counter(true_ngrams)

      #sum of how many grams appear in both the gen sequence and the true
      matching_grams_sum = sum(min(gen_grams_count[gram], true_grams_count[gram]) for gram in gen_grams_count)

      #divide sum of matching grams by total number of grams in the gen sequence (precision)

      #gram_score = 0
      #if len(gen_grams_count) > 0:
      #  gram_score = matching_grams_sum / sum(gen_grams_count.values())
      #scores.append(gram_score)

      total_gen = sum(gen_grams_count.values())
      if total_gen == 0:
        gram_score = 1e-9   # smoothing for missing n-grams
      else:
        gram_score = matching_grams_sum / total_gen
      scores.append(gram_score)

    #calculate geometric mean of scores for each gram 1-n
    geo_mean = 0.0
    for gram_score in scores:
      if gram_score == 0.0:
        #return 0 early: a score of 0 zeroes out mean and thus bleu score
        return 0.0
      geo_mean += math.log(gram_score)
    geo_mean = math.exp(geo_mean/n)

    #include brevity penalty in cases where gen sequence is longer than true sequence
    if gen_len < true_len:
      return math.exp(1 - true_len / gen_len) * geo_mean
    return geo_mean #no penalty otherwise

def compute_bleus(predicted, ground_truth):
  bleu_scores = []
  for pred, true in zip(predicted, ground_truth):
    bleu_score = bleu_n_score(pred, true, 4)
    bleu_scores.append(bleu_score)
  avg_bleu = np.mean(bleu_scores)
  return avg_bleu

# Compute metrics
def compute_metrics(predicted_sequences_tokens, true_sequences_tokens):
  lev_dist_val, lev_norm_val = compute_levenshtein_metrics(predicted_sequences_tokens, true_sequences_tokens)
  bleu = compute_bleus(predicted_sequences_tokens, true_sequences_tokens)
  # ---------------------------------------------------------
  # PRINT METRICS
  # ---------------------------------------------------------
  print("\n===================== METRICS ====================")
  print(f"Average Levenshtein Distance : {lev_dist_val:.4f}")
  print(f"Normalized Levenshtein       : {lev_norm_val:.4f}")
  print(f"Average BLEU Score                   : {bleu:.4f}")
  print("==================================================\n")

  return lev_dist_val, lev_norm_val, bleu

def save_test_result(file_name, true_sequences_tokens, true_sequences_latex, predicted_sequences_tokens, predicted_sequences_latex):
  '''
  File name's extension is .pkl
  Remember to include the path for file name
  (eg: '/content/drive/MyDrive/Group Project 2025-2026/result.pkl')
  '''
  test_result = {
      'true_sequences_tokens': true_sequences_tokens,
      'true_sequences_latex': true_sequences_latex,
      'predicted_sequences_tokens': predicted_sequences_tokens,
      'predicted_sequences_latex': predicted_sequences_latex
  }

  with open(file_name, 'wb') as f:
    pickle.dump(test_result, f)
  print(f"Saved test result in {file_name} successfully!")
  
def load_test_result(file_name):
  with open(file_name, 'rb') as f:
    test_result = pickle.load(f)
  print(f"Load test result from {file_name} successfully!")
  return test_result