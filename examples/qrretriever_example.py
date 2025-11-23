from qrretriever.attn_retriever import QRRetriever
retriever = QRRetriever(model_name_or_path="meta-llama/Llama-3.2-3B-Instruct")

query = "Which town in Nizhnyaya has the largest population?"
docs = [
    {"idx": "test0", "title": "Kushva", "paragraph_text": "Kushva is the largest town in Nizhnyaya. It has a population of 1,000."},
    {"idx": "test1", "title": "Levikha", "paragraph_text": "Levikha is a bustling town in Nizhnyaya. It has a population of 200,000."},
]
scores = retriever.score_docs(query, docs)

print(scores)
# expected output:
# Llama-3.2-3B-Instruct: {'test0': 0.74169921875, 'test1': 1.1298828125}
# Llama-3.1-8B-Instruct: {'test0': 1.0048828125, 'test1': 1.0791015625}
