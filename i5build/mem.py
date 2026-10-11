import ctypes, os
class MS(ctypes.Structure):
    _fields_=[("dwLength",ctypes.c_ulong),("dwMemoryLoad",ctypes.c_ulong),
              ("ullTotalPhys",ctypes.c_ulonglong),("ullAvailPhys",ctypes.c_ulonglong),
              ("ullTotalPageFile",ctypes.c_ulonglong),("ullAvailPageFile",ctypes.c_ulonglong),
              ("ullTotalVirtual",ctypes.c_ulonglong),("ullAvailVirtual",ctypes.c_ulonglong),
              ("ullAvailExtendedVirtual",ctypes.c_ulonglong)]
m=MS(); m.dwLength=ctypes.sizeof(MS)
ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
print("total %.2f GB  available %.2f GB  load %d%%" % (m.ullTotalPhys/2**30, m.ullAvailPhys/2**30, m.dwMemoryLoad))
