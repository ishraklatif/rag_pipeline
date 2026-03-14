# print("Testing new import locations...")

# try:
#     from langchain.retrievers import create_retrieval_chain
#     print("Found in: langchain.retrievers")
# except Exception as e:
#     print("Not in langchain.retrievers:", e)

# try:
#     from langchain.chains import create_retrieval_chain
#     print("Found in: langchain.chains")
# except Exception as e:
#     print("Not in langchain.chains:", e)

# print("\nTest complete.")
import importlib

print("Checking available LangChain modules and attributes...\n")

modules_to_try = [
    "langchain.chains",
    "langchain.chains.retrieval",
    "langchain.chains.retrieval_qa",
    "langchain.retrievers",
    "langchain_community.chains",
    "langchain_community.retrievers",
]

for mod in modules_to_try:
    try:
        m = importlib.import_module(mod)
        print(f"Found module: {mod}")
        if hasattr(m, "create_retrieval_chain"):
            print(f"   → contains create_retrieval_chain")
        else:
            print(f"   → no create_retrieval_chain in this module.")
    except Exception as e:
        print(f"{mod} not found: {e.__class__.__name__}: {e}")

print("\nTest complete.")
