# -*- coding: utf-8 -*-
"""GIF 解析失败时用"最后成功块边界"收尾，避免吞掉后面的文件（D 变体的 FAIL）。"""
import io

p = r'D:\work\bitsandbytes-CPU\bitsandbytes\tools\sector_carve.c'
s = io.open(p, encoding='utf-8').read()

# 1) 函数签名加 outSafe
s = s.replace("static int gifFileSize(const unsigned char* b, size_t len, DWORDLONG* outSize) {\n    size_t p = 0;",
              "static int gifFileSize(const unsigned char* b, size_t len, DWORDLONG* outSize,\n"
              "                       DWORDLONG* outSafe) {\n    size_t p = 0;")

# 2) 每个块循环开头记录安全边界
s = s.replace("""    for (;;) {
        unsigned char blk;
        if (p + 1 > len) return 0;""",
"""    for (;;) {
        unsigned char blk;
        if (outSafe) *outSafe = (DWORDLONG)p;   /* 最近一个完整块边界 */
        if (p + 1 > len) return 0;""")

# 3) 调用处：失败时用安全边界
s = s.replace("""            DWORDLONG gsz = 0;
            int gr = gifFileSize(buf, len, &gsz);
            if (gr == 1) { endPos = fileStart + gsz; hitTail = TRUE; }
            else if (gr == -1) break;      /* 结构非法：交给外层 32MB 兜底 */
            break;""",
"""            DWORDLONG gsz = 0, gsafe = 0;
            int gr = gifFileSize(buf, len, &gsz, &gsafe);
            if (gr == 1) {
                endPos = fileStart + gsz;
                hitTail = TRUE;
            } else if (gr == -1) {
                /* 中段损坏：用最后成功走到的块边界收尾，绝不吞掉后续文件 */
                if (gsafe > 13) { endPos = fileStart + gsafe; hitTail = TRUE; }
                else break;
            }
            break;""")
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('GIF 失败兜底已加')
