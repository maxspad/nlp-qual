from sklearn.base import TransformerMixin, BaseEstimator
import spacy
import numpy as np
from tqdm import tqdm

class SpacyTransformer(BaseEstimator, TransformerMixin):
    '''Transforms an array of texts strings to spacy Docs'''
    def __init__(self, spacy_model='en_core_web_sm', procs=1, prog=False):
        '''
        Transforms an array of text strings into spacy Docs

            spacy_model: the spacy model to use to generate the Docs. Default 'en_core_web_sm'
            procs: Number of processors to use, default 1, -1 is all available 
            prog: use a progress bar
        '''
        self.spacy_model = spacy_model
        self.procs = procs
        self.prog = prog     

    def fit(self, X, y=None):
        self._nlp = spacy.load(self.spacy_model)
        return self

    def transform(self, X, y=None):
        '''
        Transform an (n, 1) ndarray of text into a (n, 1) ndarray of spacy Doc objects
        '''
        texts = X[:,0].tolist()
        if not self.prog:
            docs = list(self._nlp.pipe(texts, n_process=self.procs))
        else:
            docs = [d for d in tqdm(self._nlp.pipe(texts, n_process=self.procs), total=len(texts))]
        docs = np.array(docs, dtype='object').reshape(len(docs),1)
        return docs

    def get_feature_names_out(self, input_features=None):
        return np.array(['docs'])

class SpacyTokenFilter(BaseEstimator, TransformerMixin):

    def __init__(self, punct=True, pron=True, stop=True, lemma=False, mrms=True, token_sep=' '):
        '''
        Given a (n, 1) ndarray of spacy Docs, filters the tokens within, optionally lemmatizing,
        for downstream processing by CountVectorizer or TfidfVectorizer

        punct: Whether to include punctuation, default True
        pron: Whether to include pronouns, default True
        stop: Whether to include stop words, default True
        lemma: Whether to lemmatize the tokens, default False
        mrms: Whether to filter mr/ms
        token_sep: After processing the allowed tokens will be returned as strings, with each
            token separated by `token_sep`
        '''

        self.punct = punct
        self.pron = pron
        self.lemma = lemma
        self.mrms = mrms
        self.stop = stop
        self.token_sep = token_sep
    
    def fit(self, X : np.ndarray, y=None):
        return self 

    def transform(self, X: np.ndarray, y=None):
        docs = X[:,0].tolist()

        processed_docs = [self._proc_doc(doc) for doc in docs]
        return processed_docs
        # return np.array(processed_docs).reshape(len(processed_docs), 1)
        
    def _proc_doc(self, doc):
        to_ret = []
        for token in doc:
            if self._check_token(token):
                if self.lemma:
                    to_ret.append(token.lemma_)
                else:
                    to_ret.append(token.text)
        to_ret = self.token_sep.join(to_ret)
        return to_ret

    def _check_token(self, token):
        if (token.pos_ == 'PUNCT') and (not self.punct):
            return False
        elif (token.pos_ == 'PRON') and (not self.pron):
            return False
        elif ((token.text.lower().find('mr') != -1) or (token.text.lower().find('ms') != -1)) and (not self.mrms):
            return False
        elif token.is_stop and not self.stop:
            return False
        else:
            return True

    def get_feature_names_out(self, input_features=None):
        return np.array(['proc_docs'])

class SpacyDocFeats(BaseEstimator, TransformerMixin):

    def __init__(self, token_count=True, pos_counts=True, ent_counts=True, vectors=True):
        '''
        Generate whole-document features from spacy Documents

        token_count: Whether to include a count of all (unfiltered) tokens in the document, default True
        pos_counts: Whether to include a count of each coarse Part-of-Speech type, default True
        ent_counts: Whether to include a count of each entity type, default True
        vectors: Whether to include an average word vector for the document, default True
        '''

        self.token_count = token_count
        self.pos_counts = pos_counts
        self.ent_counts = ent_counts
        self.vectors = vectors
        
    def fit(self, X, y=None):
        self._pos_list = ['ADJ','ADP','ADV','AUX','CCONJ','DET','INTJ','NOUN',
                          'NUM','PART','PRON','PROPN','PUNCT','SCONJ','SYM','VERB','X','SPACE']
        self._ent_list = ['CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW',
                          'LOC', 'MONEY', 'NORP', 'ORDINAL', 'ORG', 'PERCENT', 'PERSON',
                          'PRODUCT', 'QUANTITY', 'TIME', 'WORK_OF_ART','']
        self._pos_dict = {p : i for i, p in enumerate(self._pos_list)}
        self._ent_dict = {e : i for i, e in enumerate(self._ent_list)}

        self.vector_size_ = None

        return self

    def transform(self, X, y=None):
        def get_pos_counts(doc):
            pos_counts = np.zeros((1, len(self._pos_dict)))
            for tok in doc:
                pos_counts[0, self._pos_dict[tok.pos_]] += 1
            return pos_counts

        def get_ent_counts(doc):
            ent_counts = np.zeros((1, len(self._ent_dict)))
            for tok in doc:
                ent_counts[0, self._ent_dict[tok.ent_type_]] += 1
            return ent_counts
        
        docs = X[:,0].tolist()
        X_tf = []
        for doc in docs:
            feats = []
            if self.pos_counts:
                feats.append(get_pos_counts(doc))
            if self.ent_counts:
                feats.append(get_ent_counts(doc))
            if self.token_count:
                feats.append(np.array([len(doc)]).reshape(1,1))
            if self.vectors:
                if self.vector_size_ is None:
                    self.vector_size_ = len(doc.vector)

                feats.append(doc.vector.reshape(1,len(doc.vector)))

            feats = np.hstack(feats)
            X_tf.append(feats)
        return np.vstack(X_tf)

    def get_feature_names_out(self, input_features=None):
        to_ret = []
        if self.pos_counts:
            to_ret += self._pos_list
        if self.ent_counts:
            to_ret += self._ent_list
        if self.token_count:
            to_ret += ['token_count']
        if self.vectors:
            to_ret += [f'vec_{i}' for i in range(self.vector_size_)]
        return np.array(to_ret)

