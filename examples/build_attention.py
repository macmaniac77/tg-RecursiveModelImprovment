from tgaoi.aoi import attention_graph, softmax_graph
from tgaoi.serialize import dumps

print(dumps(attention_graph()))
print("\n--- Recursive child AOI: SOFTMAX ---\n")
print(dumps(softmax_graph()))
