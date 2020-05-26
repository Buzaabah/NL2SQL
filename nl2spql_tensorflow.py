import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.preprocessing.text import Tokenizer # We focus on tokenizer here but there are different ways provided
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
			'I love my dog'
			'I love my cat'
			'You love my dog!'
			'Do you think my dog is amazing?'
			]

tokenizer = Tokenizer(num_words = 100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)


eng = open('data.en', 'r').readlines()
#sparql = open('data.sparql', 'r').readlines()

print("type_of_eng:" % type(eng)), print("len_eng:" % len(eng)) 
#print("type_of_sparql:" % type(sparql)) print("len_of_sparql:" % len(sparql))