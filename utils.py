from nltk.tokenize import TweetTokenizer
import gensim
import numpy as np
import re, random

class TweetMapper:
	MAX_NB_WORDS=20
	def __init__(self):
		self.model = gensim.models.KeyedVectors.load_word2vec_format('SBW-vectors-300-min5.bin', binary=True)
		self.tknzr = TweetTokenizer()

	def name(self):
		names = ['Juan','Pedro','Luis','Adrián','Carlos','Cristina','Marta','Sara','María','Lucía','Paula','Laura','Roberto','Miguel','Sandra','Ana']
		return random.choice(names)

	def vectorize(self, raw_tweets):
		tweets = []
		regex = re.compile('[%s]' % re.escape('1234567890!¡"$%&\'()*+,-./:;<=>?¿[\\]^_`{|}~'))
		#Cleaning tweets
		for tweet in raw_tweets:

			lower_tweet = tweet.lower()

			no_url = re.sub(r"\S*(\.com|\.ly|\.co|\.net|\.org|\.me|\.gl)\S*", "", lower_tweet)
			jaja = re.sub(r'\w*(jaja|kaka|jeje|jiji|juju|jojo|ajaj|jaaj)\w*','jaja',no_url)
			repeat = re.sub(r'([a-z])\1{2,}',r'\1', jaja)
			no_pun = regex.sub('', repeat)
			#tokenizing
			tokenized = self.tknzr.tokenize(no_pun)
			#regenerate tweet
			important_words=[]
			for word in tokenized:
				if not word.startswith('#'):
					important_words.append(word)
					
			tweets.append(important_words)

		shape = (len(tweets), self.MAX_NB_WORDS, 300)
		tweets_tensor = np.zeros(shape, dtype=np.float32)

		for i in range(len(tweets)):
			#vectorizing each word in the tweet with a vector shape = (300,)
			length = len(tweets[i])
			for f in range(length):
				word = tweets[i][f]
				if f >= self.MAX_NB_WORDS:
					continue
				#if is not in the vocabulary
				if word in self.model.wv.vocab:
					tweets_tensor[i][f] = self.model.wv[word]
				else:
					#if it is a mention vectorize a name, for example @michael123 -> would be Carlos
					if word.startswith('@'):
						tweets_tensor[i][f] = self.model.wv[self.name()]
					#if not append the unknown token
					else:
						tweets_tensor[i][f] = self.model.wv['unk']
			#End of sentence token
			if length - 1 < self.MAX_NB_WORDS:
				tweets_tensor[i][length - 1] = self.model.wv['eos']

		return tweets_tensor
