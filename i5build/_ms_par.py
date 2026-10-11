import threading, time, urllib.request
UA={"User-Agent":"Mozilla/5.0"}
MS="https://modelscope.cn/api/v1/models/Qwen/Qwen-Image-2.1/repo?Revision=master&FilePath=text_encoder/model-00001-of-00004.safetensors"
HF="https://hf-mirror.com/Qwen/Qwen-Image-2.1/resolve/main/text_encoder/model-00001-of-00004.safetensors"
SIZE=4998056552
def one(url,start,end,out,i,warm=1<<20):
    h=dict(UA); h["Range"]=f"bytes={start}-{end}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url,headers=h),timeout=120) as r:
            st=r.status
            if st!=206:
                out[i]={"err":f"status {st}","b":0,"t":0.0}; return
            n=0
            while n<warm:
                c=r.read(1<<20)
                if not c: break
                n+=len(c)
            n=0; t0=time.time()
            while n<(6<<20):
                c=r.read(1<<20)
                if not c: break
                n+=len(c)
            out[i]={"err":None,"b":n,"t":time.time()-t0}
    except Exception as e:
        out[i]={"err":f"{type(e).__name__}:{str(e)[:50]}","b":0,"t":0.0}
def probe(name,url,n,span=8<<20):
    out=[None]*n; th=[]
    for i in range(n):
        end=SIZE-1-i*span; start=end-span+1
        th.append(threading.Thread(target=one,args=(url,start,end,out,i)))
    t0=time.time()
    for t in th: t.start()
    for t in th: t.join()
    wall=time.time()-t0
    ok=[r for r in out if r and not r["err"]]
    tot=sum(r["b"] for r in ok)
    print(f"{name:9s} n={n:2d}  wall={wall:5.1f}s  ok={len(ok):2d}/{n}  aggregate={tot/wall/1e6:5.2f} MB/s  best_single={max((r['b']/max(r['t'],1e-9)/1e6 for r in ok),default=0):5.2f}")
    errs=[r["err"] for r in out if r and r["err"]]
    if errs: print("           errs:", errs[:3])
print("6 MB sample per connection, 1 MB warm-up")
probe("hf-mirror",HF,8)
probe("modelscope",MS,8)
probe("modelscope",MS,16)
