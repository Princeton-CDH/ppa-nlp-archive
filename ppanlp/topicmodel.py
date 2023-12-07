from .imports import *
from .corpus import PPA
import tomotopy as tp

class BaseTopicModel:
    topicmodel_type='topicmodel_type'
    def __init__(
            self, 
            output_dir=None,
            path_docs=None,
            corpus=None,
            ntopic=50, 
            niter=100, 
            clean=None, 
            min_doc_len=25,
            frac=1,
            max_per_cluster=None,
            ):

        self.corpus=corpus if corpus!=None else PPA()
        self.ntopic = ntopic
        self.niter = niter
        self.clean = self.corpus.do_clean if clean == None else clean
        self.min_doc_len = min_doc_len
        self.frac=frac
        self.max_per_cluster=max_per_cluster
        self._mdl = None
        self.id2index = {}
        self.index2id = {}
        self._id_docs,self._ids,self._docs = None,None,None

        self.path_topicmodel = os.path.join(self.corpus.path_data,'topicmodels') 
        self.path = output_dir if output_dir and os.path.isabs(output_dir) else os.path.join(
            self.path_topicmodel,
            'models',
            (
                output_dir if output_dir else (
                    f'data.{self.topicmodel_type}.model.ntopic_{self.ntopic}.niter_{self.niter}.min_doc_len_{self.min_doc_len}.frac_{self.frac}.max_per_cluster_{self.max_per_cluster}'
                )
            )
        )
        self.path_model = os.path.join(self.path, 'model.bin')
        self.path_index = os.path.join(self.path, 'index.json')
        self.path_params = os.path.join(self.path, 'params.json')
        self.path_ldavis = os.path.join(self.path, 'ldavis')

    def iter_pages(self, lim=None):
        yield from self.corpus.iter_pages(
            lim=lim,
            min_doc_len=self.min_doc_len,
            frac=self.frac,
            max_per_cluster=self.max_per_cluster,
            as_dict=False
        )

    def iter_docs(self, lim=None):
        yield from (
            (page.id,page.txt)
            for page in self.iter_pages(lim=lim)
        )

    def iter_sents(self, lim=None, max_len=1024):
        import nltk
        iterr = (
            (page.id,sent[:max_len])
            for page in self.iter_pages()
            for sent in nltk.sent_tokenize(page.txt)
        )
        yield from iterlim(iterr, lim)


    def model(self, **kwargs):
        return # IMPLEMENT
    
    @cached_property
    def mdl(self):
        if self._mdl == None: self.model()
        return self._mdl

        


class TomotopyTopicModel(BaseTopicModel):
    topicmodel_type='tomotopy'
    def model(self, output_dir=None, force=False, lim=None):
        # get filename
        fdir=self.path if not output_dir else output_dir
        os.makedirs(fdir, exist_ok=True)
        fn=self.path_model
        fnindex=self.path_index
        fnparams=self.path_params

        # save?
        if force or not os.path.exists(fn) or not os.path.exists(fnindex):
            mdl = self.mdl = tp.LDAModel(k=self.ntopic)
            docd=self.id2index={}
            for page in self.iter_docs(lim=lim):
                docd[page.id] = mdl.add_doc(page.content_words)

            def getdesc():
                return f'Training model (ndocs={len(docd)}, log-likelihood = {mdl.ll_per_word:.4})'
            
            pbar=tqdm(list(range(0, self.niter, 1)),desc=getdesc(),position=0)
            
            for i in pbar:
                pbar.set_description(getdesc())
                mdl.train(1)
            mdl.save(fn)
            print('Saved:',fn)
            with open(fnindex,'wb') as of:
                of.write(orjson.dumps(docd,option=orjson.OPT_INDENT_2))
            mdl.summary()
            print('Saved:',fnindex)

            params=dict(
                ntopic=self.ntopic,
                niter=self.niter,
                clean=self.clean,
                min_doc_len=self.min_doc_len,
                frac=self.frac,
                lim=lim
            )
            with open(fnparams,'wb') as of:
                of.write(orjson.dumps(params,option=orjson.OPT_INDENT_2))

        else:
            print('Loading:',fn)
            self.mdl = tp.LDAModel.load(fn)
            with open(fnindex,'rb') as f:
                self.id2index=orjson.loads(f.read())
        self.index2id={v:k for k,v in self.id2index.items()}
    
    
    @cached_property
    def topic_term_dists(self): 
        return np.stack([self.mdl.get_topic_word_dist(k) for k in range(self.mdl.k)])

    @cached_property
    def doc_topic_dists(self):
        doc_topic_dists = np.stack([doc.get_topic_dist() for doc in self.mdl.docs])
        doc_topic_dists /= doc_topic_dists.sum(axis=1, keepdims=True)
        return doc_topic_dists


    @cached_property
    def doc_lengths(self):
        return np.array([len(doc.words) for doc in self.mdl.docs])

    @cached_property
    def vocab(self): return list(self.mdl.used_vocabs)
    @cached_property
    def term_frequency(self): return self.mdl.used_vocab_freq

    def save_pyldavis(self, output_dir=None):
        import pyLDAvis
        prepared_data = pyLDAvis.prepare(
            self.topic_term_dists, 
            self.doc_topic_dists, 
            self.doc_lengths, 
            self.vocab, 
            self.term_frequency,
            start_index=0, # tomotopy starts topic ids with 0, pyLDAvis with 1
            sort_topics=False # IMPORTANT: otherwise the topic_ids between pyLDAvis and tomotopy are not matching!
        )
        
        output_dir = self.path_ldavis if not output_dir else output_dir
        os.makedirs(output_dir, exist_ok=True)
        fn=os.path.join(output_dir, 'index.html')
        print('Saving:',fn)
        pyLDAvis.save_html(prepared_data, fn)
        return fn
    

    @cached_property
    def doc_topic_dists_df(self):
        page_ids,values = zip(*[(self.index2id[i],x) for i,x in enumerate(self.doc_topic_dists) if i in self.index2id])
        dftopicdist = pd.DataFrame(values)
        dftopicdist['page_id'] = page_ids
        dftopicdist['work_id']=[i.split('_')[0] for i in page_ids]
        dftopicdist['cluster']=[self.id2cluster.get(work_id,work_id) for work_id in dftopicdist.work_id]
        return dftopicdist.set_index(['cluster','work_id','page_id'])
    
    @cached_property
    def id2cluster(self):
        return dict(zip(self.corpus.meta.index, self.corpus.meta[CLUSTER_KEY]))

    @cached_property
    def meta(self):
        df = self.doc_topic_dists_df        
        df_avgs=df.groupby('work_id').mean(numeric_only=True)
        return self.corpus.meta.merge(df_avgs, on='work_id').set_index('work_id')
    






class BertTopicModel(BaseTopicModel):
    topicmodel_type='bertopic'
    embedding_model_name = 'emanjavacas/MacBERTh'

    @cached_property
    def embedding_model(self):
        from transformers.pipelines import pipeline
        embedding_model = pipeline(
            "feature-extraction", 
            model=self.embedding_model_name,
            # model_kwargs=dict(
                # max_length=512
            # )
        )
        return embedding_model
    
    def init_docs(self, by_sent=False, lim=None, force=False):
        if force or self._id_docs is None:
            for bk in ['id_docs','docs','ids','doc2id','id2doc']:
                if bk in self.__dict__:
                    self.__dict__.pop(bk)
            with logwatch('loading documents into memory') as lw:
                self._id_docs = list(self.iter_sents(lim=lim) if by_sent else self.iter_docs(lim=lim))
                lw.log(f'loaded {len(self._id_docs):,} documents into memory')
    
    @cached_property
    def id_docs(self):
        if self._id_docs is None: self.init_docs()
        return self._id_docs
    @cached_property
    def docs(self): return [y for x,y in self.id_docs]
    @cached_property
    def ids(self): return [x for x,y in self.id_docs]
    @cached_property
    def doc2id(self): return {y:x for x,y in self.id_docs}
    @cached_property
    def id2doc(self): return {x:y for x,y in self.id_docs}


    def model(self, output_dir=None,force=False, lim=None, save=True, embedding_model=None, by_sent=False, **kwargs):
        with logwatch('loading or generating model'):
            # get filename
            fdir=self.path if not output_dir else output_dir
            os.makedirs(fdir, exist_ok=True)
            fn=self.path_model
            if not force and os.path.exists(fn): return self.load(fn)

            with logwatch('importing BERTopic'):
                os.environ['TOKENIZERS_PARALLELISM']='false'
                from bertopic import BERTopic
                from bertopic.representation import KeyBERTInspired

            # get docs
            docs = self.init_docs(by_sent=by_sent, lim=lim)

            with logwatch('fitting model'):
                self._mdl = BERTopic(
                    embedding_model=embedding_model,
                    # embedding_model=self.embedding_model,
                    representation_model=KeyBERTInspired(), 
                    verbose=True, 
                    **kwargs
                )
                self._topics, self._probs = self._mdl.fit_transform(docs)
            
            if save: self.save()
            return self._mdl


    def save(self):
        if self._mdl is not None:
            fn=self.path_model
            with logwatch(f'saving model to disk: {truncfn(fn)}'):
                ensure_dir(fn)
                self.mdl.save(fn)

    def load(self, fn=None):
        if not fn: fn=self.path_model
        if os.path.exists(fn):
            with logwatch('importing BERTopic'):
                from bertopic import BERTopic
            with logwatch(f'loading model from disk: {truncfn(fn)}'):
                self._mdl = BERTopic.load(self.path_model)
                return self._mdl

    @cached_property
    def doc_df(self):
        docinfo = self.mdl.get_document_info(self.docs)
        docinfo.columns = [x.lower() for x in docinfo]
        docinfo['page_id']=[self.doc2id[doc] for doc in docinfo.document]
        return docinfo

    @cached_property
    def topic_df(self):
        tdf = self.mdl.get_topic_info()
        tdf.columns = [x.lower() for x in tdf]
        tdf['representative_docs_ids']=[[self.doc2id[doc] for doc in docs] for docs in tdf.representative_docs]
        return tdf

    @cached_property
    def page2topic(self):
        return dict(zip(self.doc_df.page_id,self.doc_df.name))


def PPATopicModel(model_type='bertopic',**kwargs):
    if model_type and model_type.lower().startswith('tomo'): return TomotopyTopicModel(**kwargs)
    return BertTopicModel(**kwargs)