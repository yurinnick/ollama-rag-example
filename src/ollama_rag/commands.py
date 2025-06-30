from ollama_rag.learner import Learner
from ollama_rag.continuos_learner import ContinousLearner


def learn(args):
    learner = Learner(
        ollama_host=args.ollama_host,
        llm_model=args.llm_model,
        embedding_model=args.embedding_model,
        vector_collection=args.vector_collection,
        vector_store_path=args.vector_store_path,
    )
    learner.learn(args.files)


def query(args):
    learner = Learner(
        ollama_host=args.ollama_host,
        llm_model=args.llm_model,
        embedding_model=args.embedding_model,
        vector_collection=args.vector_collection,
        vector_store_path=args.vector_store_path,
    )
    print(learner.query(args.query))


def watch(args):
    learner = Learner(
        ollama_host=args.ollama_host,
        llm_model=args.llm_model,
        embedding_model=args.embedding_model,
        vector_collection=args.vector_collection,
        vector_store_path=args.vector_store_path,
    )
    ContinousLearner(
        learner=learner, index_db_path=f"{args.vector_store_path}/index.db"
    ).run(args.path)
