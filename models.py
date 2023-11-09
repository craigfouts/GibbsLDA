import numpy as np
import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
import torch
from pyro.infer import SVI, TraceEnum_ELBO
from pyro.optim import Adam
from tqdm import tqdm

class GibbsLDA():
    def __init__(self, n_topics, topics_prior=None, docs_prior=None):
        self.n_topics = n_topics
        self.topics_prior = topics_prior
        self.docs_prior = docs_prior

        self.word_labels_ = None
        self.topic_word_counts_ = None
        self.doc_topic_counts_ = None
        self.topic_word_dist_ = None
        self.doc_topic_dist_ = None
        self.likelihood_log_ = []

    def _sample_topics(self):
        concentrations = self.topics_prior + self.topic_word_counts_
        for k in range(self.n_topics):
            self.topic_word_dist_[k] = np.random.dirichlet(concentrations[k])
        return self.topic_word_dist_

    def _decrement_counts(self, doc, label, value):
        self.topic_word_counts_[label, value] -= 1
        self.doc_topic_counts_[doc, label] -= 1
        return self.topic_word_counts_, self.doc_topic_counts_

    def _increment_counts(self, doc, label, value):
        self.topic_word_counts_[label, value] += 1
        self.doc_topic_counts_[doc, label] += 1
        return self.topic_word_counts_, self.doc_topic_counts_
    
    def _sample_dists(self, doc, value):
        topic_word_dist = self.topic_word_counts_[:, value]/self.topic_word_counts_.sum(-1)
        doc_topic_dist = self.doc_topic_counts_[doc]/self.doc_topic_counts_[doc].sum()
        probs = topic_word_dist*doc_topic_dist
        return probs/probs.sum()

    def _sample_word(self, doc, word, value):
        probs = self._sample_dists(doc, value)
        label = np.random.choice(probs.shape[0], p=probs)
        self.word_labels_[doc, word] = label
        self.likelihood_log_[-1] += probs[label]
        return label

    def _sample_words(self, X, doc, n_words):
        for n in range(n_words):
            old_label, word_value = self.word_labels_[doc, n], X[doc, n]
            if self.topic_word_counts_[old_label, word_value] < 2 or self.doc_topic_counts_[doc, old_label] < 2:
                continue
            self._decrement_counts(doc, old_label, word_value)
            new_label = self._sample_word(doc, n, word_value)
            self._increment_counts(doc, new_label, word_value)
        return self.word_labels_

    def _sample_docs(self, X, n_docs, n_words, sample_words=True):
        self.likelihood_log_.append(0.)
        concentrations = self.docs_prior + self.doc_topic_counts_
        for d in range(n_docs):
            self.doc_topic_dist_[d] = np.random.dirichlet(concentrations[d])
            if sample_words:
                self._sample_words(X, d, n_words)
        return self.doc_topic_dist_

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

    def _init_dists(self, X, n_docs, n_words):
        self.topic_word_dist_ = np.zeros_like(self.topic_word_counts_, dtype=np.float32)
        self._sample_topics()
        self.doc_topic_dist_ = np.zeros_like(self.doc_topic_counts_, dtype=np.float32)
        self._sample_docs(X, n_docs, n_words, sample_words=False)
        return self.topic_word_dist_, self.doc_topic_dist_

    def fit(self, X, n_steps=100, verbose=1):
        n_docs, n_words, vocab_size = *X.shape, np.unique(X).shape[0]
        self._set_priors(vocab_size)
        self._init_labels(X, n_docs, n_words, vocab_size)
        self._init_dists(X, n_docs, n_words)
        for _ in tqdm(range(n_steps)) if verbose == 1 else range(n_steps):
            self._sample_topics()
            self._sample_docs(X, n_docs, n_words)
        return self
    
    def transform(self, _=None):
        doc_labels = self.doc_topic_dist_.argmax(-1)
        return doc_labels
    
class CollapsedGibbsLDA():
    def __init__(self, n_topics, topics_prior=None, docs_prior=None):
        self.n_topics = n_topics
        self.topics_prior = topics_prior
        self.docs_prior = docs_prior

        self.word_labels_ = None
        self.topic_word_counts_ = None
        self.doc_topic_counts_ = None
        self.topic_word_dist_ = None
        self.doc_topic_dist_ = None
        self.likelihood_log_ = []
    
    def _decrement_counts(self, doc, label, value):
        self.topic_word_counts_[label, value] -= 1
        self.doc_topic_counts_[doc, label] -= 1
        return self.topic_word_counts_, self.doc_topic_counts_
    
    def _increment_counts(self, doc, label, value):
        self.topic_word_counts_[label, value] += 1
        self.doc_topic_counts_[doc, label] += 1
        return self.topic_word_counts_, self.doc_topic_counts_
    
    def _sample_dists(self, doc, value):
        word_concentration = self.topic_word_counts_ + self.topics_prior
        self.topic_word_dist_[:, value] = word_concentration[:, value]/word_concentration.sum(-1)
        topic_concentration = self.doc_topic_counts_[doc] + self.docs_prior
        self.doc_topic_dist_[doc] = topic_concentration/topic_concentration.sum()
        probs = self.topic_word_dist_[:, value]*self.doc_topic_dist_[doc]
        return probs/probs.sum()

    def _sample_word(self, doc, word, value):
        probs = self._sample_dists(doc, value)
        label = np.random.choice(probs.shape[0], p=probs)
        self.word_labels_[doc, word] = label
        self.likelihood_log_[-1] += probs[label]
        return label
    
    def _sample_words(self, X, n_words, doc):
        for n in range(n_words):
            old_label, word_value = self.word_labels_[doc, n], X[doc, n]
            if self.topic_word_counts_[old_label, word_value] < 2 or self.doc_topic_counts_[doc, old_label] < 2:
                continue
            self._decrement_counts(doc, old_label, word_value)
            new_label = self._sample_word(doc, n, word_value)
            self._increment_counts(doc, new_label, word_value)
        return self.word_labels_
    
    def _sample_docs(self, X, n_docs, n_words):
        self.likelihood_log_.append(0.)
        for d in range(n_docs):
            self._sample_words(X, n_words, d)
        return self.doc_topic_dist_
    
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
        self.topic_word_dist_ = np.zeros_like(self.topic_word_counts_, dtype=np.float32)
        self.doc_topic_dist_ = np.zeros_like(self.doc_topic_counts_, dtype=np.float32)
        return self.topic_word_dist_, self.doc_topic_dist_
    
    def fit(self, X, n_steps=100, verbose=1):
        n_docs, n_words, vocab_size = *X.shape, np.unique(X).shape[0]
        self._set_priors(vocab_size)
        self._init_labels(X, n_docs, n_words, vocab_size)
        self._init_dists()
        for _ in tqdm(range(n_steps)) if verbose == 1 else range(n_steps):
            self._sample_docs(X, n_docs, n_words)
        return self
    
    def transform(self, _=None):
        doc_labels = self.doc_topic_dist_.argmax(-1)
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
