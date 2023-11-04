import numpy as np
import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
import torch
from pyro.infer import SVI, TraceEnum_ELBO
from pyro.optim import Adam
from tqdm import tqdm

def normalize(arr):
    return arr/arr.sum()

def sample_categorical(probs):
    n_bins = probs.shape[0]
    num = np.random.rand(1)
    sample = n_bins - 1
    for i in range(n_bins - 1):
        if num < probs[i]:
            sample = i
    return sample

class GibbsGMM():
    def __init__(self, n_topics, alpha=1., eta=1., sigma=1., init_random=True):
        self.n_topics = n_topics
        self.alpha = alpha
        self.eta = eta
        self.sigma = sigma
        self.init_random = init_random

        self.means_ = None
        self.labels_ = None
        self.counts_ = None
        self.likelihood_log_ = []

    def _build(self, n_samples, n_features):
        if self.init_random:
            self.means_ = np.random.normal(0., self.eta, size=(self.n_topics, n_features))
        else:
            self.means_ = np.zeros((self.n_topics, n_features))
        self.labels_ = np.random.randint(self.n_topics, size=(n_samples,))
        _, self.counts_ = np.unique(self.labels_, return_counts=True)
        return self.means_, self.labels_, self.counts_

    def _sample_topics(self, X):
        for k in range(self.n_topics):
            if self.counts_[k] == 0:
                continue
            topic_var = 1/(self.counts_[k]/self.sigma**2 + 1/self.eta**2)
            idx = self.labels_ == k
            topic_means = (self.counts_[k]/self.sigma**2)*topic_var*X[idx].mean(0)
            self.means_[k] = np.random.normal(topic_means, topic_var)
        return self.means_ 
    
    def _sample_words(self, X):
        n_samples = X.shape[0]
        self.likelihood_log_.append(0.)
        for i in range(n_samples):
            self.counts_[self.labels_[i]] -= 1
            probs = normalize(1/((X[i] - self.means_)**2).sum(-1))
            self.labels_[i] = sample_categorical(probs)
            # self.labels_[i] = np.random.choice(self.n_topics, p=probs)
            self.likelihood_log_[-1] += probs[self.labels_[i]].sum()
            self.counts_[self.labels_[i]] += 1
        return self.labels_

    def fit(self, X, n_steps=100, verbose=True):
        self._build(*X.shape)
        for _ in tqdm(range(n_steps)) if verbose else range(n_steps):
            self._sample_topics(X)
            self._sample_words(X)
        return self
    
class PyroLDA():
    def __init__(self, n_topics, vocab_size, batch_size=150):
        self.n_topics = n_topics
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        
        self.topic_words_posterior_ = None
        self.doc_topics_posterior_ = None
        self.loss_log_ = []

    def _model(self, data=None):
        n_words, n_docs = data.shape
        with pyro.plate('topics', self.n_topics):
            topic_words = pyro.sample('topic_words', dist.Dirichlet(torch.ones(self.vocab_size)/self.vocab_size))
        with pyro.plate('documents', n_docs, self.batch_size) as idx:
            if data is not None:
                data = data[:, idx]
            doc_topics = pyro.sample('doc_topics', dist.Dirichlet(torch.ones(self.n_topics)/self.n_topics))
            with pyro.plate('words', n_words):
                word_topics = pyro.sample('word_topics', dist.Categorical(doc_topics), infer={'enumerate': 'parallel'})
                data = pyro.sample('doc_words', dist.Categorical(topic_words[word_topics]), obs=data)
        
    def _guide(self, data):
        n_docs = data.shape[-1]
        self.topic_words_posterior_ = pyro.param('topic_words_posterior', lambda: torch.ones(self.n_topics, self.vocab_size), constraint=constraints.greater_than(.5))
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
    
    def transform(self, reduce=True):
        doc_topics = pyro.sample('doc_topics', dist.Dirichlet(self.doc_topics_posterior_))
        if reduce:
            return doc_topics.argmax(-1)
        return doc_topics
