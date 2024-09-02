

use = [f"{i}x{i+3}" for i in range(0,10,3)]
use2 = [f"{i}x{i+3}" for i in range(10)]

use.append('12x15')
use.append('15x18')
use.append('18x21')
use.append('21x24')

print(use)
print(use2)

