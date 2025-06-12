## faster wigner funcitons
import pywigxjpf as wig
from functools import lru_cache

# Initialize
@lru_cache(maxsize=None)
def init_wig_table(maxJ):
    wig.wig_table_init(2*maxJ,9)
    wig.wig_temp_init(2*maxJ)

init_wig_table(20)

from pywigxjpf import wig3jj
from pywigxjpf import wig6jj
from pywigxjpf import wig9jj


@lru_cache(maxsize=None)
def wigner_3j(j1,j2,j3,m1,m2,m3):
    return wig3jj(int(2*j1),int(2*j2),int(2*j3),int(2*m1),int(2*m2),int(2*m3))

@lru_cache(maxsize=None)    
def wigner_6j(j1,j2,j3,j4,j5,j6):
    return wig6jj(int(2*j1),int(2*j2),int(2*j3),int(2*j4),int(2*j5),int(2*j6))

@lru_cache(maxsize=None)    
def wigner_9j(j1,j2,j3,j4,j5,j6,j7,j8,j9):
    return wig9jj(int(2*j1),int(2*j2),int(2*j3),int(2*j4),int(2*j5),int(2*j6),int(2*j7),int(2*j8),int(2*j9))

if __name__ == "__main__":
    print('WIGXJPF python test program')
