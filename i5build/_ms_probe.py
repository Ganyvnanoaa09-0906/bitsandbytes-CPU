import time, urllib.request
UA={"User-Agent":"Mozilla/5.0"}
cands=[
 ("api-v1","https://modelscope.cn/api/v1/models/Qwen/Qwen-Image-2.1/repo?Revision=master&FilePath=text_encoder/model-00001-of-00004.safetensors"),
 ("resolve","https://modelscope.cn/models/Qwen/Qwen-Image-2.1/resolve/master/text_encoder/model-00001-of-00004.safetensors"),
]
for name,url in cands:
    try:
        h=dict(UA); h["Range"]="bytes=0-9999999"
        t0=time.time()
        with urllib.request.urlopen(urllib.request.Request(url,headers=h),timeout=60) as r:
            st=r.status; cl=r.headers.get("Content-Length"); cr=r.headers.get("Content-Range")
            n=0
            while n < (2<<20):
                c=r.read(1<<20)
                if not c: break
                n+=len(c)
            t1=time.time(); n=0
            while n < (10<<20):
                c=r.read(1<<20)
                if not c: break
                n+=len(c)
            dt=time.time()-t1
        print(f"{name:9s} status={st} len={cl} range={cr}  steady={n/dt/1e6:.2f} MB/s ({n/1e6:.1f} MB in {dt:.1f}s)")
    except Exception as e:
        print(f"{name:9s} ERR {type(e).__name__}: {str(e)[:80]}")
