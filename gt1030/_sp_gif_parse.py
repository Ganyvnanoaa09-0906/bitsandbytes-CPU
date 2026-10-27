# -*- coding: utf-8 -*-
"""给 sector_carve 加一个真正的 GIF 结构解析器（按块走到 trailer 0x3B），替换"猜 0x3B"。

GIF 布局：
  "GIF87a"/"GIF89a"(6) + 逻辑屏描述(7) [+ 全局调色板 3*2^(N+1)]
  然后循环：
    0x21 扩展：label(1) + 子块序列（len 字节 + 数据）... 直到 len==0
    0x2C 图像：描述符(9) [+ 局部调色板] + LZW 最小码长(1) + 子块序列（len+数据）... 直到 len==0
    0x3B 结束（trailer）→ 文件到此为止
返回 1=完整（*outSize 有效），0=数据不够（需要继续读），-1=结构非法。
"""
import io

p = r'D:\work\bitsandbytes-CPU\bitsandbytes\tools\sector_carve.c'
s = io.open(p, encoding='utf-8').read()
L = s.split('\n')

helper = r'''
/* ---------------- GIF 结构解析：走到 trailer(0x3B) 得到精确文件长度 ----------------
   1 = 完整(*outSize 有效)，0 = 数据还不够（继续读），-1 = 结构非法。 */
static int gifFileSize(const unsigned char* b, size_t len, DWORDLONG* outSize) {
    size_t p = 0;
    size_t gt;
    int gct;
    for (;;) {
        if (p + 13 > len) return 0;
        if (memcmp(b + p, "GIF8", 4) != 0) return -1;
        gct = (b[p + 10] & 0x80) ? 1 : 0;
        p += 13;
        if (gct) {
            gt = (size_t)3 << ((b[p - 3] & 0x07) + 1);
            if (p + gt > len) return 0;
            p += gt;
        }
        break;
    }
    for (;;) {
        unsigned char blk;
        if (p + 1 > len) return 0;
        blk = b[p];
        if (blk == 0x3B) { *outSize = (DWORDLONG)p + 1; return 1; }
        if (blk == 0x21) {                 /* 扩展块 */
            p += 2;                        /* 0x21 + label */
            for (;;) {
                size_t n;
                if (p + 1 > len) return 0;
                n = b[p];
                p += 1;
                if (n == 0) break;
                if (p + n > len) return 0;
                p += n;
            }
            continue;
        }
        if (blk == 0x2C) {                 /* 图像描述符 */
            int lct;
            size_t lt;
            if (p + 10 > len) return 0;
            lct = (b[p + 9] & 0x80) ? 1 : 0;
            p += 10;
            if (lct) {
                lt = (size_t)3 << ((b[p - 1] & 0x07) + 1);
                if (p + lt > len) return 0;
                p += lt;
            }
            p += 1;                        /* LZW 最小码长 */
            if (p > len) return 0;
            for (;;) {
                size_t n;
                if (p + 1 > len) return 0;
                n = b[p];
                p += 1;
                if (n == 0) break;
                if (p + n > len) return 0;
                p += n;
            }
            continue;
        }
        return -1;
    }
}
'''
# 插在 extractFile 之前
ei = next(i for i, ln in enumerate(L) if 'static DWORDLONG extractFile(' in ln)
L[ei:ei] = helper.split('\n')

# 替换 GIF 分支（当前是我上一版的块）
gi = next(i for i, ln in enumerate(L)
          if ln.strip() in ('case FMT_GIF:', 'case FMT_GIF: {'))
if L[gi].strip() == 'case FMT_GIF:':
    end = gi + 3
    assert '0x3B' in L[gi + 2], 'GIF 主体不符: %r' % L[gi:gi + 4]
else:
    depth = 0
    end = None
    for j in range(gi, len(L)):
        depth += L[j].count('{') - L[j].count('}')
        if j > gi and depth == 0 and L[j].strip() == '}':
            end = j
            break
    assert end, 'GIF 块边界未找到'
new = '''        case FMT_GIF: {
            /* 按 GIF 结构解析（块遍历）拿到精确长度；比"猜 0x3B"可靠得多。 */
            DWORDLONG gsz = 0;
            int gr = gifFileSize(buf, len, &gsz);
            if (gr == 1) { endPos = fileStart + gsz; hitTail = TRUE; }
            else if (gr == -1) break;      /* 结构非法：交给外层 32MB 兜底 */
            break;
        }'''.split('\n')
L[gi:end + 1] = new

io.open(p, 'w', encoding='utf-8', newline='\n').write('\n'.join(L))
print('GIF 结构解析器已加入，行数 =', len(L))
