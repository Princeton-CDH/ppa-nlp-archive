from .imports import *
from .cleanup import cleanup_pages

PPA_OBJ=None
PATH_CORPUS=PATH_PPA_CORPUS

def set_corpus_path(path=None,**kwargs):
    global PPA_OBJ,PATH_CORPUS
    PATH_CORPUS=os.path.abspath(os.path.expanduser(path)) if path else PATH_PPA_CORPUS
    PPA_OBJ = PPACorpus(PATH_CORPUS,**kwargs)

def PPA(path=None, **kwargs):
    if PPA_OBJ == None or path or kwargs: set_corpus_path(path, **kwargs)
    return PPA_OBJ


class PPACorpus:
    WORK_ID_FIELD = 'id'

    def __init__(self, path:str, clean=False, texts_dir='texts', metadata_fn='metadata.json', texts_preproc_dir='texts_preproc'):
        self.path = os.path.abspath(os.path.expanduser(path))
        self.do_clean=clean
        self.path_texts = os.path.join(self.path,texts_dir) if not os.path.isabs(texts_dir) else texts_dir
        self.path_texts_preproc = os.path.join(self.path,texts_preproc_dir) if not os.path.isabs(texts_preproc_dir) else texts_preproc_dir
        self.path_metadata = os.path.join(self.path,metadata_fn) if not os.path.isabs(metadata_fn) else metadata_fn

    def __iter__(self): yield from self.iter_texts()

    @cached_property
    def meta(self):
        return pd.read_json(self.path_metadata).fillna('').set_index(self.WORK_ID_FIELD)
    
    @cache
    def get_text(self, work_id):
        return PPAText(work_id, corpus=self)

    @cached_property
    def texts(self):
        return list(self.iter_texts())


    def iter_texts(self, work_ids=None):
        if work_ids is None: work_ids=self.meta.index
        pdesc='Iteration over texts in PPA'
        pbar = tqdm(total=len(work_ids), position=0, desc=pdesc)
        for work_id in work_ids:
            # pbar.set_description(f'{pdesc}: {work_id}')
            yield self.get_text(work_id)
            pbar.update()

    def clean_texts(self, work_ids=None, num_proc=None):
        work_ids = list(self.meta.index) if work_ids == None else work_ids
        num_proc=(mp.cpu_count()//2)-1 if not num_proc else num_proc
        num_proc=num_proc if num_proc>0 else 1
        pool = mp.Pool(num_proc)

        random.shuffle(work_ids)
        objs = work_ids
        res = list(tqdm(pool.imap_unordered(cleanup_mp,objs), total=len(objs), position=0, desc=f'Cleaning PPA pages (script from Wouter Haverals) [{num_proc}x]'))
        return res


class PPAText:
    FILE_ID_KEY='work_id'

    def __init__(self, id, corpus=None,clean=None):
        self.id=id
        self.corpus=corpus if corpus is not None else PPA()
        self.do_clean=self.corpus.do_clean if clean==None else clean

    def __iter__(self): yield from self.iter_pages()

    @cached_property
    def id(self): return self.meta.get('work_id')

    @cached_property
    def meta(self):
        return dict(self.corpus.meta.loc[self.id])

    @cached_property
    def path(self):
        return os.path.join(self.corpus.path_texts, self.meta[self.FILE_ID_KEY]+'.json')
    
    @cached_property
    def path_preproc(self):
        return os.path.join(self.corpus.path_texts_preproc, self.meta[self.FILE_ID_KEY]+'.json.gz')

    @cached_property
    def pages_df(self):
        return pd.DataFrame([p.meta for p in self.iter_pages()]).set_index('page_id')

    @cached_property
    def pages(self): 
        return list(self.iter_pages())
    @cached_property
    def pages_orig(self): 
        return list(self.iter_pages_orig())

    def clean(self,remove_headers=True,force=False):
        self.do_clean=True
        for k in ['pages','pages_df','txt']:
            if k in self.__dict__: del self.__dict__[k]

        if force or not os.path.exists(self.path_preproc):
            pages_ld = list(self.iter_pages_orig(as_dict=True))
            new_pages = cleanup_pages(pages_ld, remove_headers=remove_headers)

            os.makedirs(os.path.dirname(self.path_preproc),exist_ok=True)
            with open(self.path_preproc,'w') as of:
                json.dump(new_pages, of, indent=2)
        return self
    
    def iter_pages_orig(self, as_dict=False):
        if os.path.exists(self.path):
            with open(self.path,'r') as f: 
                for d in json.load(f):
                    yield PPAPage(self, **d) if not as_dict else d
    
    def iter_pages_preproc(self, as_dict=False):
        self.clean()
        if os.path.exists(self.path_preproc):
            with open(self.path_preproc,'r') as f: 
                for d in json.load(f):
                    yield PPAPage(self, **d) if not as_dict else d

    def iter_pages(self, clean=None):
        clean = self.do_clean if clean==None else clean
        yield from self.iter_pages_preproc() if clean else self.iter_pages_orig()

    @cached_property
    def txt(self, sep='\n\n\n\n'):
        return sep.join(page.txt for page in self.pages)
        


def cleanup_mp(work_id):
    t = PPA().get_text(work_id)
    t.clean()
    return t.path_preproc


class PPAPage:
    def __init__(self, text, **page_d):
        self.text = text
        self.d = page_d

    @cached_property
    def meta(self):
        return {'work_id':self.text.id, **self.d}

    @cached_property
    def txt(self):
        return self.meta.get('page_text')