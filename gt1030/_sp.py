import io
p=r"bitsandbytes/tools/sector_mirror.c"
L=io.open(p,encoding="utf-8").read().split("\n")
H=["static DWORD SalvageRead(HANDLE h, BYTE* buf, DWORD want, DWORDLONG off, FILE* badmap, DWORDLONG* badOut) {","    DWORD got=0,o,q,part,sp,br,sb; const DWORD mid=65536, small=4096; LARGE_INTEGER li;","    for (o=0;o<want;o+=mid) {","        part=(want-o<mid)?(want-o):mid; li.QuadPart=(LONGLONG)(off+o);","        if (!SetFilePointerEx(h,li,NULL,FILE_BEGIN)) break;","        br=0;","        if (ReadFile(h,buf+o,part,&br,NULL)&&br==part) { got+=part; continue; }","        for (q=0;q<part;q+=small) {","            sp=(part-q<small)?(part-q):small; li.QuadPart=(LONGLONG)(off+o+q);","            if (!SetFilePointerEx(h,li,NULL,FILE_BEGIN)) break;","            sb=0;","            if (ReadFile(h,buf+o+q,sp,&sb,NULL)&&sb==sp) got+=sp;","            else { memset(buf+o+q,0,sp); if (badmap) fprintf(badmap,\"%016llx %u\\n\",(unsigned long long)(off+o+q),sp); if (badOut) *badOut+=sp; }","        }","    }","    return got;","}",""]
for i,ln in enumerate(L):
    if ln.strip().startswith("int wmain("): L[i:i]=H; break
for i,ln in enumerate(L):
    if ln.strip()=="DWORDLONG total = diskSize;":
        L[i+1:i+1]=["    DWORDLONG badBytes = 0;","    FILE* fpBad = NULL;"]; break
for i,ln in enumerate(L):
    if "if (!ReadFile(hSrc, buf, bufSize, &br, NULL)) {" in ln:
        L[i:i+5]=["        if (!ReadFile(hSrc, buf, bufSize, &br, NULL)) {","            DWORD err = GetLastError();","            if (err == ERROR_HANDLE_EOF) break;","            { LARGE_INTEGER lr; lr.QuadPart=(LONGLONG)scanned; SetFilePointerEx(hSrc,lr,NULL,FILE_BEGIN); }","            br = SalvageRead(hSrc, buf, bufSize, scanned, fpBad, &badBytes);","            if (br == 0) { memset(buf,0,bufSize); br = bufSize; }","        }"]; break
io.open(p,"w",encoding="utf-8",newline="\n").write("\n".join(L))
print("patched", len(L))
