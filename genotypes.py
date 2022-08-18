from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [        # 基本体
    'none',
    'max_pool_3x1',
    'avg_pool_3x1',
    'skip_connect',
    'sep_conv_3x1',
    'sep_conv_5x1',
    'dil_conv_3x1',
    'dil_conv_5x1'
]

NASNet = Genotype(
  normal = [
    ('sep_conv_5x1', 1),
    ('sep_conv_3x1', 0),
    ('sep_conv_5x1', 0),
    ('sep_conv_3x1', 0),
    ('avg_pool_3x1', 1),
    ('skip_connect', 0),
    ('avg_pool_3x1', 0),
    ('avg_pool_3x1', 0),
    ('sep_conv_3x1', 1),
    ('skip_connect', 1),
  ],
  normal_concat = [2, 3, 4, 5, 6],
  reduce = [
    ('sep_conv_5x1', 1),
    ('sep_conv_7x1', 0),
    ('max_pool_3x1', 1),
    ('sep_conv_7x1', 0),
    ('avg_pool_3x1', 1),
    ('sep_conv_5x1', 0),
    ('skip_connect', 3),
    ('avg_pool_3x1', 2),
    ('sep_conv_3x1', 2),
    ('max_pool_3x1', 1),
  ],
  reduce_concat = [4, 5, 6],
)
    
AmoebaNet = Genotype(
  normal = [
    ('avg_pool_3x1', 0),
    ('max_pool_3x1', 1),
    ('sep_conv_3x1', 0),
    ('sep_conv_5x1', 2),
    ('sep_conv_3x1', 0),
    ('avg_pool_3x1', 3),
    ('sep_conv_3x1', 1),
    ('skip_connect', 1),
    ('skip_connect', 0),
    ('avg_pool_3x1', 1),
    ],
  normal_concat = [4, 5, 6],
  reduce = [
    ('avg_pool_3x1', 0),
    ('sep_conv_3x1', 1),
    ('max_pool_3x1', 0),
    ('sep_conv_7x1', 2),
    ('sep_conv_7x1', 0),
    ('avg_pool_3x1', 1),
    ('max_pool_3x1', 0),
    ('max_pool_3x1', 1),
    ('conv_7x1_1x1', 0),
    ('sep_conv_3x1', 5),
  ],
  reduce_concat = [3, 4, 6]
)

DARTS_V1 = Genotype(normal=[('sep_conv_3x1', 1), ('sep_conv_3x1', 0), ('skip_connect', 0), ('sep_conv_3x1', 1), ('skip_connect', 0), ('sep_conv_3x1', 1), ('sep_conv_3x1', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x1', 0), ('max_pool_3x1', 1), ('skip_connect', 2), ('max_pool_3x1', 0), ('max_pool_3x1', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x1', 0)], reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(normal=[('sep_conv_3x1', 0), ('sep_conv_3x1', 1), ('sep_conv_3x1', 0), ('sep_conv_3x1', 1), ('sep_conv_3x1', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x1', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x1', 0), ('max_pool_3x1', 1), ('skip_connect', 2), ('max_pool_3x1', 1), ('max_pool_3x1', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x1', 1)], reduce_concat=[2, 3, 4, 5])


# --------------snr = FREE----------------
# model = Genotype(normal=[('sep_conv_3x1', 1), ('sep_conv_5x1', 0), ('sep_conv_5x1', 1), ('sep_conv_5x1', 0), ('dil_conv_5x1', 1), ('dil_conv_3x1', 3), ('sep_conv_3x1', 4), ('sep_conv_5x1', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_3x1', 0), ('max_pool_3x1', 1), ('dil_conv_5x1', 1), ('max_pool_3x1', 0), ('skip_connect', 2), ('max_pool_3x1', 0), ('sep_conv_3x1', 3), ('sep_conv_5x1', 0)], reduce_concat=range(2, 6))
# ==========================================================================

# ============(50) first accuracy second spareness---------------------

# model = Genotype(normal=[('sep_conv_5x1', 0), ('dil_conv_5x1', 1), ('dil_conv_5x1', 1), ('dil_conv_5x1', 2), ('sep_conv_5x1', 1), ('sep_conv_5x1', 0), ('max_pool_3x1', 4), ('dil_conv_3x1', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_5x1', 0), ('dil_conv_3x1', 1), ('dil_conv_5x1', 1), ('max_pool_3x1', 0), ('dil_conv_3x1', 1), ('max_pool_3x1', 0), ('sep_conv_3x1', 3), ('max_pool_3x1', 1)], reduce_concat=range(2, 6))

# model = Genotype(normal=[('sep_conv_5x1', 0), ('dil_conv_3x1', 1), ('skip_connect', 1), ('dil_conv_5x1', 2), ('max_pool_3x1', 3), ('max_pool_3x1', 0), ('avg_pool_3x1', 0), ('sep_conv_3x1', 3)], normal_concat=range(2, 6), reduce=[('dil_conv_5x1', 1), ('max_pool_3x1', 0), ('sep_conv_5x1', 1), ('dil_conv_3x1', 2), ('avg_pool_3x1', 0), ('avg_pool_3x1', 1), ('sep_conv_3x1', 3), ('sep_conv_5x1', 0)], reduce_concat=range(2, 6))

model = Genotype(normal=[('dil_conv_5x1', 0), ('dil_conv_5x1', 1), ('sep_conv_5x1', 2), ('sep_conv_5x1', 1), ('dil_conv_5x1', 1), ('max_pool_3x1', 3), ('dil_conv_3x1', 4), ('sep_conv_5x1', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_3x1', 0), ('skip_connect', 1), ('sep_conv_5x1', 1), ('dil_conv_3x1', 2), ('skip_connect', 1), ('avg_pool_3x1', 0), ('max_pool_3x1', 1), ('sep_conv_5x1', 0)], reduce_concat=range(2, 6))

# model = Genotype(normal=[('dil_conv_5x1', 1), ('sep_conv_5x1', 0), ('sep_conv_5x1', 1), ('dil_conv_5x1', 2), ('max_pool_3x1', 3), ('sep_conv_5x1', 1), ('avg_pool_3x1', 4), ('dil_conv_3x1', 1)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('dil_conv_5x1', 2), ('dil_conv_3x1', 3), ('max_pool_3x1', 0), ('sep_conv_5x1', 0), ('sep_conv_3x1', 3)], reduce_concat=range(2, 6))

# model = Genotype(normal=[('sep_conv_5x1', 0), ('sep_conv_5x1', 1), ('sep_conv_5x1', 2), ('avg_pool_3x1', 1), ('dil_conv_5x1', 0), ('dil_conv_3x1', 3), ('dil_conv_3x1', 4), ('sep_conv_5x1', 0)], normal_concat=range(2, 6), reduce=[('max_pool_3x1', 1), ('dil_conv_3x1', 0), ('sep_conv_5x1', 1), ('sep_conv_5x1', 0), ('sep_conv_3x1', 2), ('avg_pool_3x1', 1), ('sep_conv_3x1', 3), ('avg_pool_3x1', 2)], reduce_concat=range(2, 6))

# model = Genotype(normal=[('dil_conv_5x1', 1), ('sep_conv_5x1', 0), ('max_pool_3x1', 1), ('dil_conv_5x1', 2), ('skip_connect', 3), ('dil_conv_5x1', 1), ('sep_conv_5x1', 0), ('avg_pool_3x1', 4)], normal_concat=range(2, 6), reduce=[('max_pool_3x1', 0), ('dil_conv_3x1', 1), ('sep_conv_5x1', 1), ('sep_conv_5x1', 2), ('max_pool_3x1', 1), ('avg_pool_3x1', 0), ('sep_conv_3x1', 3), ('sep_conv_3x1', 2)], reduce_concat=range(2, 6))

# model = Genotype(normal=[('sep_conv_5x1', 0), ('dil_conv_3x1', 1), ('sep_conv_3x1', 0), ('dil_conv_5x1', 1), ('max_pool_3x1', 1), ('sep_conv_3x1', 0), ('avg_pool_3x1', 4), ('max_pool_3x1', 2)], normal_concat=range(2, 6), reduce=[('dil_conv_3x1', 0), ('max_pool_3x1', 1), ('dil_conv_3x1', 1), ('max_pool_3x1', 0), ('dil_conv_5x1', 0), ('sep_conv_3x1', 3), ('sep_conv_3x1', 3), ('skip_connect', 2)], reduce_concat=range(2, 6))

# model = Genotype(normal=[('dil_conv_5x1', 1), ('sep_conv_5x1', 0), ('max_pool_3x1', 1), ('dil_conv_5x1', 2), ('max_pool_3x1', 3), ('dil_conv_5x1', 1), ('avg_pool_3x1', 4), ('sep_conv_5x1', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_3x1', 0), ('max_pool_3x1', 1), ('dil_conv_5x1', 1), ('sep_conv_3x1', 2), ('dil_conv_5x1', 0), ('max_pool_3x1', 1), ('sep_conv_5x1', 3), ('avg_pool_3x1', 2)], reduce_concat=range(2, 6))

# model = Genotype(normal=[('sep_conv_5x1', 0), ('dil_conv_5x1', 1), ('max_pool_3x1', 1), ('dil_conv_5x1', 0), ('max_pool_3x1', 3), ('dil_conv_5x1', 0), ('sep_conv_5x1', 1), ('sep_conv_3x1', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_5x1', 0), ('max_pool_3x1', 1), ('dil_conv_5x1', 2), ('sep_conv_5x1', 0), ('skip_connect', 1), ('dil_conv_5x1', 3), ('sep_conv_5x1', 0), ('sep_conv_3x1', 3)], reduce_concat=range(2, 6))

# model = Genotype(normal=[('dil_conv_5x1', 0), ('dil_conv_5x1', 1), ('sep_conv_5x1', 2), ('sep_conv_5x1', 0), ('max_pool_3x1', 3), ('max_pool_3x1', 0), ('dil_conv_3x1', 4), ('sep_conv_3x1', 3)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('max_pool_3x1', 1), ('dil_conv_5x1', 1), ('sep_conv_5x1', 2), ('max_pool_3x1', 0), ('max_pool_3x1', 1), ('sep_conv_3x1', 4), ('sep_conv_5x1', 3)], reduce_concat=range(2, 6))


DARTS = model

def main():
  # print(DARTS_V1)
  print(PRIMITIVES.index("none"))
  print(PRIMITIVES[5])

if __name__ == '__main__':
    main()