def calculate(utterance, norm=None):
    utterance_words = utterance.split()[1:]

    utterance_words = list(filter(lambda word: not (word == '((' or word == '))' or word == '=' or word == '+'
                                                or word == '(' or word == ')' or word == '<noise>' or word == '</noise>'
                                                or word == '++' or word == '-' or word == '))('), utterance_words))

    num_switch = 0
    num_dialect = {'msa': 0, 'non_msa': 0}
    curr_dialect = 'msa'
    for i in range(len(utterance_words)):
        word = utterance_words[i]
        if (word == '<non-MSA>' and i != 0) or (word == '</non-MSA>' and i != len(utterance_words) - 1):
            num_switch += 1
        if word == '<non-MSA>':
            curr_dialect = 'non_msa'
            continue
        elif word == '</non-MSA>':
            curr_dialect = 'msa'
            continue
        num_dialect[curr_dialect] += 1

    num_words = float(num_dialect['msa'] + num_dialect['non_msa'])
    max_dialect = max(num_dialect['msa'], num_dialect['non_msa'])
    cmi = 100*((0.5*(num_words - max_dialect) + 0.5*num_switch) / num_words)

    if norm == None:
        return cmi
    else:
        cmi = (cmi - norm[1]) / norm[0]
        if cmi == 0:
            return '1'
        elif cmi < 0.15:
            return '2'
        elif cmi < 0.3:
            return '3'
        elif cmi < 0.45:
            return '4'
        elif cmi <= 1.0:
            return '5'
        return None