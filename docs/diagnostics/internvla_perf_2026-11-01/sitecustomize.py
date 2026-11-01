import os, sys, time, json, functools
if os.environ.get('IVLA_PROFILE_CLIENT') == '1' and 'simpler_env.eval' in sys.orig_argv:
    from simpler_env.run import Runner
    _orig_init = Runner.__init__
    _orig_get = Runner._get_action
    _stats = {}
    _log = open('/tmp/internvla_diag/client_profile.jsonl','a',buffering=1)
    def _emit(kind, **data):
        _log.write(json.dumps(dict(ts=time.time(),pid=os.getpid(),kind=kind,**data))+'\n')
    def _wrap_obj(obj, attr, label):
        orig=getattr(obj,attr)
        @functools.wraps(orig)
        def call(*a,**kw):
            t=time.perf_counter(); out=orig(*a,**kw); dt=time.perf_counter()-t
            _stats.setdefault(label,[]).append(dt)
            if label=='env_step' and len(_stats[label])%10==0:
                _emit('steps',n=len(_stats[label]),means={k:sum(v)/len(v) for k,v in _stats.items() if v})
            return out
        setattr(obj,attr,call)
    @functools.wraps(_orig_init)
    def _init(self,*a,**kw):
        if _stats: _emit('previous_shard',means={k:sum(v)/len(v) for k,v in _stats.items() if v})
        _stats.clear(); t=time.perf_counter(); _orig_init(self,*a,**kw)
        _emit('runner_init',seconds=time.perf_counter()-t)
        _wrap_obj(self.env,'step','env_step'); _wrap_obj(self.env,'reset','env_reset')
    @functools.wraps(_orig_get)
    def _get(self,*a,**kw):
        t=time.perf_counter(); out=_orig_get(self,*a,**kw); _stats.setdefault('get_action',[]).append(time.perf_counter()-t); return out
    Runner.__init__=_init; Runner._get_action=_get
