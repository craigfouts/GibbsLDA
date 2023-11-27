import numpy as np
import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
import torch
from pyro.infer import SVI, TraceEnum_ELBO
from pyro.optim import Adam
from scipy.cluster.vq import kmeans, vq
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

class GibbsSLDA(BaseEstimator, TransformerMixin):
    def __init__(self, n_topics, n_docs=150, vocab_size=25, vocab_steps=10, sigma=1., alpha=None, beta=None):
        self.n_topics = n_topics
        self.n_docs = n_docs
        self.vocab_size = vocab_size
        self.vocab_steps = vocab_steps
        self.sigma = sigma
        self.alpha = 1/n_docs if alpha is None else alpha
        self.beta = 1/n_topics if beta is None else beta

        self.library_ = None
        self.doc_imgs_ = None
        self.doc_locs_ = None
        self.doc_topic_counts_ = np.zeros((n_docs, n_topics), dtype=np.int32)
        self.topic_word_counts_ = np.zeros((n_topics, vocab_size), dtype=np.int32)
        self.likelihood_log_ = []

    def _featurize(self, imgs, locs, markers):
        mask = (cdist(imgs, imgs) == 0).astype(np.int32)
        dists = cdist(locs, locs)
        weights = mask*np.exp(-(dists/self.sigma)**2)
        features = (markers.T@weights).T
        return features

    def _shuffle(self, words):
        docs = np.random.choice(self.n_docs, (words.shape[0], 1))
        topics = np.random.choice(self.n_topics, (words.shape[0], 1))
        for d in range(self.n_docs):
            idx, counts = np.unique(topics[docs == d], return_counts=True)
            self.doc_topic_counts_[d, idx.astype(np.int32)] = counts
        for k in range(self.n_topics):
            idx, counts = np.unique(words[topics == k], return_counts=True)
            self.topic_word_counts_[k, idx.astype(np.int32)] = counts
        return docs, topics

    def _build(self, X):
        imgs, locs, markers = X[:, :1], X[:, 1:3], X[:, 3:]
        n_imgs = np.unique(imgs).shape[0]
        # locs, markers = X[:, :2], X[:, 2:]
        doc_idx = np.random.permutation(X.shape[0])[:self.n_docs]
        # doc_idx = np.random.permutation(X.shape[0])[:self.n_docs]
        self.doc_imgs_ = imgs[doc_idx]
        self.doc_locs_ = locs[doc_idx]
        features = self._featurize(imgs, locs, markers)
        codebook, _ = kmeans(features, self.vocab_size, self.vocab_steps)
        words = vq(features, codebook)[0][None].T
        docs, topics = self._shuffle(words)
        self.library_ = np.concatenate([imgs, locs, words, docs, topics], -1)
        # self.library_ = np.concatenate([locs, words, docs, topics], -1)
        return self.library_
    
    def _sample_doc(self, img, loc, topic, eta=1e-100):
        mask = (self.doc_imgs_ == img).astype(np.int32).T[0]
        doc_probs = mask*self.sigma**2/(((loc - self.doc_locs_)**2).sum(-1) + eta)
        topic_probs = self.doc_topic_counts_[:, topic] + self.alpha
        topic_probs /= (self.doc_topic_counts_ + self.alpha).sum(-1)
        probs = doc_probs*topic_probs/(doc_probs*topic_probs).sum()
        doc = np.random.choice(self.n_docs, p=probs)
        return doc, probs[doc]

    def _sample_topic(self, word, doc):
        topic_probs = self.doc_topic_counts_[doc] + self.alpha
        topic_probs /= (self.doc_topic_counts_[doc] + self.alpha).sum()
        word_probs = self.topic_word_counts_[:, word] + self.beta
        word_probs /= (self.topic_word_counts_ + self.beta).sum(-1)
        probs = topic_probs*word_probs/(topic_probs*word_probs).sum()
        topic = np.random.choice(self.n_topics, p=probs)
        return topic, probs[topic]
    
    def _sample(self, img, loc, word, old_doc, old_topic):
        new_doc, doc_likelihood = self._sample_doc(img, loc, old_topic)
        new_topic, topic_likelihood = self._sample_topic(word, old_doc)
        likelihood = doc_likelihood + topic_likelihood
        return new_doc, new_topic, likelihood
    
    def _decrement(self, word, doc, topic):
        self.doc_topic_counts_[doc, topic] -= 1
        self.topic_word_counts_[topic, word] -= 1
        return self.doc_topic_counts_, self.topic_word_counts_
    
    def _increment(self, word, doc, topic):
        self.doc_topic_counts_[doc, topic] += 1
        self.topic_word_counts_[topic, word] += 1
        return self.doc_topic_counts_, self.topic_word_counts_
    
    def _step(self):
        self.likelihood_log_.append(0.)
        for i in range(self.library_.shape[0]):
            img, loc, (word, doc, topic) = self.library_[i, :1], self.library_[i, 1:3], self.library_[i, 3:].astype(np.int32)
            self._decrement(word, doc, topic)
            doc, topic, likelihood = self._sample(img, loc, word, doc, topic)
            self._increment(word, doc, topic)
            self.library_[i, -2:] = doc, topic
            self.likelihood_log_[-1] += likelihood
        return self.likelihood_log_[-1]
    
    def fit(self, X, n_steps=100, verbose=1):
        self._build(X)
        for i in tqdm(range(n_steps)) if verbose == 1 else range(n_steps):
            likelihood = self._step()
            if verbose == 2:
                print('step', i, 'likelihood:', likelihood)
        return self
    
    def transform(self, _=None):
        topics = self.library_[:, -1]
        return topics

class GibbsLDA():
    def __init__(self, n_topics, n_words=10, vocab_size=25, vocab_steps=100, alpha=None, beta=None):
        self.n_topics = n_topics
        self.n_words = n_words
        self.vocab_size = vocab_size
        self.vocab_steps = vocab_steps
        self.alpha = 1/vocab_size if alpha is None else alpha
        self.beta = 1/n_topics if beta is None else beta

        self.library_ = None
        self.word_labels_ = None       
        self.topic_word_counts_ = None
        self.doc_topic_counts_ = None
        self.topic_word_dists_ = None
        self.doc_topic_dists_ = None
        self.likelihood_log_ = []

    def _sample_topics(self):
        concentrations = self.topic_word_counts_ + self.alpha
        for k in range(self.n_topics):
            self.topic_word_dists_[k] = np.random.dirichlet(concentrations[k])
        return self.topic_word_dists_

    def _decrement(self, doc, label, value):
        self.topic_word_counts_[label, value] -= 1
        self.doc_topic_counts_[doc, label] -= 1
        return self.topic_word_counts_, self.doc_topic_counts_

    def _increment(self, doc, label, value):
        self.topic_word_counts_[label, value] += 1
        self.doc_topic_counts_[doc, label] += 1
        return self.topic_word_counts_, self.doc_topic_counts_
    
    def _sample_dists(self, doc, value, normalize=True):
        word_concentration = self.topic_word_counts_ + self.alpha
        topic_concentration = self.doc_topic_counts_ + self.beta
        topic_word_dist = word_concentration[:, value]/word_concentration.sum(-1)
        doc_topic_dist = topic_concentration[doc]/topic_concentration[doc].sum()
        probs = topic_word_dist*doc_topic_dist
        if normalize:
            return probs/probs.sum()
        return probs

    def _sample_word(self, doc, word, value):
        probs = self._sample_dists(doc, value)
        label = np.random.choice(probs.shape[0], p=probs)
        self.word_labels_[doc, word] = label
        self.likelihood_log_[-1] += probs[label]
        return label

    def _sample_words(self, doc):
        for n in range(self.n_words):
            old_label, word_value = self.word_labels_[doc, n], self.library_[doc, n]
            self._decrement(doc, old_label, word_value)
            new_label = self._sample_word(doc, n, word_value)
            self._increment(doc, new_label, word_value)
        return self.word_labels_

    def _sample_docs(self, n_samples, sample_words=True):
        concentrations = self.doc_topic_counts_ + self.beta
        for d in range(n_samples):
            self.doc_topic_dists_[d] = np.random.dirichlet(concentrations[d])
            if sample_words:
                self._sample_words(d)
        return self.doc_topic_dists_
    
    def _sample(self, n_samples, sample_words=True):
        topic_word_dists = self._sample_topics()
        doc_topic_dists = self._sample_docs(n_samples, sample_words)
        return topic_word_dists, doc_topic_dists

    def _shuffle(self, n_samples):
        labels = np.random.choice(self.n_topics, (n_samples, self.n_words))
        self.topic_word_counts_ = np.zeros((self.n_topics, self.vocab_size), dtype=np.int32)
        self.doc_topic_counts_ = np.zeros((n_samples, self.n_topics), dtype=np.int32)
        self.topic_word_dists_ = np.zeros_like(self.topic_word_counts_, dtype=np.float32)
        self.doc_topic_dists_ = np.zeros_like(self.doc_topic_counts_, dtype=np.float32)
        for k in range(self.n_topics):
            idx, counts = np.unique(self.library_[labels == k], return_counts=True)
            self.topic_word_counts_[k, idx] = counts
        for d in range(n_samples):
            idx, counts = np.unique(labels[d], return_counts=True)
            self.doc_topic_counts_[d, idx] = counts
        self._sample(n_samples, sample_words=False)
        return labels
    
    def _build(self, X):
        codebook, _ = kmeans(X[:, 2:], self.vocab_size, self.vocab_steps)
        neighbors = NearestNeighbors(n_neighbors=self.n_words).fit(X)
        _, neighbor_idx = neighbors.kneighbors(X)
        self.library_ = np.zeros((X.shape[0], self.n_words), dtype=np.int32)
        for i in range(X.shape[0]):
            self.library_[i], _ = vq(X[neighbor_idx, 2:][i], codebook)
        self.word_labels_ = self._shuffle(X.shape[0])
        return self.library_, self.word_labels_

    def fit(self, X, n_steps=100, verbose=1):
        self._build(X)
        for i in tqdm(range(n_steps)) if verbose == 1 else range(n_steps):
            self.likelihood_log_.append(0.)
            self._sample(X.shape[0])
            if verbose == 2:
                print('step', i, 'likelihood:', self.likelihood_log_[-1])
        return self
    
    def transform(self, _=None):
        doc_labels = self.doc_topic_dists_.argmax(-1)
        return doc_labels
    
class CollapsedGibbsLDA():
    def __init__(self, n_topics, topics_prior=None, docs_prior=None):
        self.n_topics = n_topics
        self.topics_prior = topics_prior
        self.docs_prior = docs_prior

        self.word_labels_ = None
        self.topic_word_counts_ = None
        self.doc_topic_counts_ = None
        self.topic_word_dists_ = None
        self.doc_topic_dists_ = None
        self.likelihood_log_ = []
    
    def _decrement_counts(self, doc, label, value):
        self.topic_word_counts_[label, value] -= 1
        self.doc_topic_counts_[doc, label] -= 1
        return self.topic_word_counts_, self.doc_topic_counts_
    
    def _increment_counts(self, doc, label, value):
        self.topic_word_counts_[label, value] += 1
        self.doc_topic_counts_[doc, label] += 1
        return self.topic_word_counts_, self.doc_topic_counts_
    
    def _sample_dists(self, doc, value, normalize=True):
        word_concentration = self.topic_word_counts_ + self.topics_prior
        self.topic_word_dists_[:, value] = word_concentration[:, value]/word_concentration.sum(-1)
        topic_concentration = self.doc_topic_counts_[doc] + self.docs_prior
        self.doc_topic_dists_[doc] = topic_concentration/topic_concentration.sum()
        probs = self.topic_word_dists_[:, value]*self.doc_topic_dists_[doc]
        if normalize:
            return probs/probs.sum()
        return probs

    def _sample_word(self, doc, word, value):
        probs = self._sample_dists(doc, value)
        label = np.random.choice(probs.shape[0], p=probs)
        self.word_labels_[doc, word] = label
        self.likelihood_log_[-1] += probs[label]
        return label
    
    def _sample_words(self, X, n_words, doc):
        for n in range(n_words):
            old_label, word_value = self.word_labels_[doc, n], X[doc, n]
            if self.topic_word_counts_[old_label, word_value] < 1 or self.doc_topic_counts_[doc, old_label] < 1:
                continue
            self._decrement_counts(doc, old_label, word_value)
            new_label = self._sample_word(doc, n, word_value)
            self._increment_counts(doc, new_label, word_value)
        return self.word_labels_
    
    def _sample_docs(self, X, n_docs, n_words):
        self.likelihood_log_.append(0.)
        for d in range(n_docs):
            self._sample_words(X, n_words, d)
        return self.doc_topic_dists_
    
    def _set_priors(self, vocab_size):
        if self.topics_prior is None:
            self.topics_prior = np.ones(vocab_size)/vocab_size
        if self.docs_prior is None:
            self.docs_prior = np.ones(self.n_topics)/self.n_topics
        return self.topics_prior, self.docs_prior
    
    def _init_labels(self, X, n_docs, n_words, vocab_size):
        probs = np.ones(self.n_topics)/self.n_topics
        self.word_labels_ = np.random.choice(self.n_topics, (n_docs, n_words), p=probs)
        self.topic_word_counts_ = np.zeros((self.n_topics, vocab_size), dtype=np.int32)
        for k in range(self.n_topics):
            idx, counts = np.unique(X[self.word_labels_ == k], return_counts=True)
            self.topic_word_counts_[k, idx] = counts
        self.doc_topic_counts_ = np.zeros((n_docs, self.n_topics), dtype=np.int32)
        for d in range(n_docs):
            idx, counts = np.unique(self.word_labels_[d], return_counts=True)
            self.doc_topic_counts_[d, idx] = counts
        return self.word_labels_, self.topic_word_counts_, self.doc_topic_counts_
    
    def _init_dists(self):
        self.topic_word_dists_ = np.zeros_like(self.topic_word_counts_, dtype=np.float32)
        self.doc_topic_dists_ = np.zeros_like(self.doc_topic_counts_, dtype=np.float32)
        return self.topic_word_dists_, self.doc_topic_dists_
    
    def fit(self, X, n_steps=100, verbose=1):
        n_docs, n_words, vocab_size = *X.shape, np.unique(X).shape[0]
        self._set_priors(vocab_size)
        self._init_labels(X, n_docs, n_words, vocab_size)
        self._init_dists()
        for _ in tqdm(range(n_steps)) if verbose == 1 else range(n_steps):
            self._sample_docs(X, n_docs, n_words)
        return self
    
    def transform(self, _=None):
        doc_labels = self.doc_topic_dists_.argmax(-1)
        return doc_labels
    
class PyroLDA():
    def __init__(self, n_topics, batch_size=100):
        self.n_topics = n_topics
        self.batch_size = batch_size
        
        self.topic_words_posterior_ = None
        self.doc_topics_posterior_ = None
        self.loss_log_ = []

    def _model(self, data=None):
        n_words, n_docs, vocab_size = *data.shape, data.unique().shape[-1]
        with pyro.plate('topics', self.n_topics):
            topic_words = pyro.sample('topic_words', dist.Dirichlet(torch.ones(vocab_size)/vocab_size))
        with pyro.plate('documents', n_docs, self.batch_size) as idx:
            if data is not None:
                data = data[:, idx]
            doc_topics = pyro.sample('doc_topics', dist.Dirichlet(torch.ones(self.n_topics)/self.n_topics))
            with pyro.plate('words', n_words):
                word_topics = pyro.sample('word_topics', dist.Categorical(doc_topics), infer={'enumerate': 'parallel'})
                data = pyro.sample('doc_words', dist.Categorical(topic_words[word_topics]), obs=data)
        
    def _guide(self, data):
        n_docs, vocab_size = data.shape[-1], data.unique().shape[-1]
        self.topic_words_posterior_ = pyro.param('topic_words_posterior', lambda: torch.ones(self.n_topics, vocab_size), constraint=constraints.greater_than(.5))
        self.doc_topics_posterior_ = pyro.param('doc_topics_posterior', lambda: torch.ones(n_docs, self.n_topics), constraint=constraints.greater_than(.5))
        with pyro.plate('topics', self.n_topics):
            pyro.sample('topic_words', dist.Dirichlet(self.topic_words_posterior_))
        with pyro.plate('documents', n_docs, self.batch_size) as idx:
            data = data[:, idx]
            pyro.sample('doc_topics', dist.Dirichlet(self.doc_topics_posterior_[idx]))

    def fit(self, X, n_steps=100, learning_rate=1e-1):
        optim = Adam({'lr': learning_rate})
        elbo = TraceEnum_ELBO(max_plate_nesting=2)
        svi = SVI(self._model, self._guide, optim, elbo)
        for _ in tqdm(range(n_steps)):
            loss = svi.step(X)
            self.loss_log_.append(loss)
        return self
    
    def transform(self, _=None):
        doc_topics = pyro.sample('doc_topics', dist.Dirichlet(self.doc_topics_posterior_)).argmax(-1)
        return doc_topics
