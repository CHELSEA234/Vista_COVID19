'''
the demo to different embedding methods.
1. senc2vec
2. sciBERT
3. sentence_transformer
'''
import torch
import sent2vec

from transformers import *
from scipy import spatial
from sentence_transformers import SentenceTransformer

sentence_1 = ['the respiratory disease is caused by the unhealty life habit.']
sentence_2 = ['the disease case leads a disorder of structure or function in a human.']
sentence_3 = ['Viruses are microscopic parasites, generally much smaller than bacteria.']

def cosine_dis(embed_1, embed_2):
	'''compute the similarity.'''
	dis = spatial.distance.cosine(embed_1, embed_2)
	print(f"the cosine similarity is {1-dis:.3f}")

def sen_transformer():
	model = SentenceTransformer('roberta-base-nli-mean-tokens')
	sentence_embedding_1 = model.encode(sentence_1)[0]
	sentence_embedding_2 = model.encode(sentence_2)[0]
	sentence_embedding_3 = model.encode(sentence_3)[0]

	cosine_dis(sentence_embedding_1, sentence_embedding_2)
	cosine_dis(sentence_embedding_1, sentence_embedding_3)

def senc_vector_model():
	model = sent2vec.Sent2vecModel()
	model.load_model(model_path)

	emb_1 = model.embed_sentence(sentence_1[0])
	emb_2 = model.embed_sentence(sentence_2[0])
	emb_3 = model.embed_sentence(sentence_3[0]) 

	cosine_dis(emb_1, emb_2)
	cosine_dis(emb_1, emb_3)

def sciBERT_model():
	tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
	model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

	sentence_list = [sentence_1, sentence_2, sentence_3]
	last_hidden_states_list = []
	for sentence_cur in sentence_list:
		input_ids = torch.tensor([tokenizer.encode(sentence_cur[0], add_special_tokens=True)])
		with torch.no_grad():
			last_hidden_states = model(input_ids)[0]
			last_hidden_states = torch.mean(last_hidden_states, dim=1)[0]
			last_hidden_states_list.append(last_hidden_states)
	cosine_dis(last_hidden_states_list[0], last_hidden_states_list[1])
	cosine_dis(last_hidden_states_list[0], last_hidden_states_list[2])

def select_model(model_name):
	'''select model for demo'''
	print(f"{model_name} has been selected.")
	if model_name == "sen_transformer":
		sen_transformer()
	elif model_name == "senc_vec":
		senc_vector_model()
	elif model_name == "sciBERT":
		sciBERT_model()

if __name__ == '__main__':
	# select_model("sciBERT")
	model_path = '/nas/vista-ssd02/users/xiaoguo/COVID_research/sent2vec/pretrained_model/wiki_unigrams.bin'
	for name in ["sen_transformer", "senc_vec", "sciBERT"]:
		select_model(name)
