# https://rosettacode.org/wiki/Knapsack_problem/0-1#Python


# For debugging purposes
def total_value(comb, limit):
    """ Total up a particular combination of items """
    totwt = totval = 0
    for dictionary in comb:
        totwt += dictionary['length']
        totval += dictionary['normalized_tf']
    return (totval, -totwt) if totwt <= limit else (0, 0)


def knapsack01_dp(sentences, limit):
    # Initialize
    table = [[0 for w in range(limit + 1)] for j in range(len(sentences) + 1)]

    for j in range(1, len(sentences) + 1):
        sentence_length = sentences[j - 1]['length']
        tf_score = sentences[j - 1]['normalized_tf']
        for w in range(1, limit + 1):
            if sentence_length > w:
                table[j][w] = table[j - 1][w]
            else:
                table[j][w] = max(table[j - 1][w],
                                  table[j - 1][w - sentence_length] + tf_score)

    result = []
    w = limit
    for j in range(len(sentences), 0, -1):
        was_added = table[j][w] != table[j - 1][w]

        if was_added:
            sentence_length = sentences[j - 1]['length']
            result.append(sentences[j - 1])
            w -= sentence_length

    return result
