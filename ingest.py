import os
from rag_core import index_folder

PAPERS_FOLDER = "papers"

if not os.path.exists(PAPERS_FOLDER):
    raise FileNotFoundError("Mappen 'papers' finns inte.")

stored = index_folder(PAPERS_FOLDER, reset=True)
print(f"Done. Stored {stored} chunks.")