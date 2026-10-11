import time, torch
torch.set_grad_enabled(False)
for (M,K,N) in ((64,64,64),(256,256,256),(1024,1024,1024),(2048,2048,2048)):
    a = torch.randint(-8,8,(M,K),dtype=torch.int8)
    b = torch.randint(-8,8,(K,N),dtype=torch.int8)
    t0=time.perf_counter()
    try:
        c = torch._int_mm(a,b)
        dt=time.perf_counter()-t0
        print("  %5dx%-5dx%-5d  %8.4f s   out %s %s" % (M,K,N,dt,tuple(c.shape),c.dtype), flush=True)
    except Exception as e:
        print("  %5dx%-5dx%-5d  FAILED %s: %s" % (M,K,N,type(e).__name__,str(e)[:60]), flush=True)
