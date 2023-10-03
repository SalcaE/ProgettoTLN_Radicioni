from nltk.wsd import lesk

sentence = "however , the jury said it believes `` these two offices should be combined to achieve greater efficiency and reduce the cost of administration '' ."

print(lesk(sentence, 'jury'))

