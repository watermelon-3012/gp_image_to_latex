import re
from torchtext.vocab import build_vocab_from_iterator

'''
(<sos>|<eos>) - Start and end of sequence
(\\[a-zA-Z]+) - LaTeX commands like \frac, \sum, \int
([{}_^$%&#]) - Symbols often used in LaTeX  like {, ^, _
([0-9]) - One digit like 1, 2, 3
([a-zA-Z]) - Single letters like x, y, a
(\S) - Any non-whitespace character like =, +, -
'''

def latex_tokenizer(text):
    # Capture LaTeX commands, brackets, symbols, numbers, letters, and any leftover characters
    pattern = r'(<sos>|<eos>)|(\\[a-zA-Z]+)|([{}_^$%&#])|([0-9])|([a-zA-Z])|(\S)'
    tokens = re.findall(pattern, text) #Returns all non-overlapping matches pattern as a list
    list_of_tokens = []
    for group in tokens:
        for token in group:
            if token:
                list_of_tokens.append(token)
    return list_of_tokens

def latex_iterator(texts):
    for text in texts:
        yield latex_tokenizer(text)

def build_vocab(list_of_text):
    specials = ["<unk>", "<pad>", "<sos>", "<eos>"]
    '''
    <unk>: unknown
    <pad>: padding
    <sos>: start of sequence
    <eos>: end of sequence
    '''
    vocab = build_vocab_from_iterator(
        latex_iterator(list_of_text),
        specials=specials
    )
    vocab.set_default_index(vocab["<unk>"])

    return vocab