from qrretriever.attn_retriever import QRRetriever
retriever = QRRetriever(model_name_or_path="meta-llama/Llama-3.1-8B-Instruct")

query = "Which town in Nizhnyaya has the largest population?"
docs = [
    {"idx": "test0", "title": "Kushva", "paragraph_text": "Kushva is a largest town in Nizhnyaya. It has a population of 1,000."},
    {"idx": "test1", "title": "Levikha", "paragraph_text": "Kushva is a bustling town in Nizhnyaya. It has a population of 200,000."},
]
scores = retriever.score_docs(query, docs)

print(scores)
# expected output: {'test0': 0.63, 'test1': 1.17}