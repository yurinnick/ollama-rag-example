import sys
import argparse
import logging

from ollama_rag.learner import Learner
from ollama_rag.continuos_learner import ContinousLearner


def main():
    root_parser = argparse.ArgumentParser(
        description="Learn and query infromation from files."
    )
    root_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )
    root_parser.add_argument(
        "--ollama-host",
        type=str,
        default="http://localhost:11434",
        help="Ollama base URL",
    )
    root_parser.add_argument(
        "--vector-collection",
        type=str,
        default="knowledge_db",
        help="Vector collection name",
    )
    root_parser.add_argument(
        "--vector-store-path",
        type=str,
        default=".knowledge_db",
        help="Path to the vector store",
    )
    root_parser.add_argument(
        "--embedding-model",
        type=str,
        default="nomic-embed-text",
        help="Embedding model",
    )
    root_parser.add_argument(
        "--llm-model",
        type=str,
        default="llama3.2",
        help="Embedding model",
    )
    subparsers = root_parser.add_subparsers(
        dest="subcommand",
        title="Subcommands",
    )

    learn_parser = subparsers.add_parser("learn")
    learn_parser.add_argument("files", nargs="+", help="Files to learn")

    def run_learn(args):
        learner = Learner(
            ollama_host=args.ollama_host,
            llm_model=args.llm_model,
            embedding_model=args.embedding_model,
            vector_collection=args.vector_collection,
            vector_store_path=args.vector_store_path,
        )
        learner.learn(args.files)

    learn_parser.set_defaults(func=run_learn)

    query_parser = subparsers.add_parser("query")
    query_parser.add_argument("query", type=str, help="Query")

    def run_query(args):
        learner = Learner(
            ollama_host=args.ollama_host,
            llm_model=args.llm_model,
            embedding_model=args.embedding_model,
            vector_collection=args.vector_collection,
            vector_store_path=args.vector_store_path,
        )
        print(learner.query(args.query))

    query_parser.set_defaults(func=run_query)

    watch_parser = subparsers.add_parser("watch")
    watch_parser.add_argument("path", type=str, help="Watch directory")

    def run_watch(args):
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

    watch_parser.set_defaults(func=run_watch)

    args = root_parser.parse_args()

    log_level = logging.INFO
    if args.verbose:
        log_level = logging.DEBUG
    logging.basicConfig(level=log_level)

    if hasattr(args, "func"):
        args.func(args)
    else:
        print("No subcommand specified. Please use one of the following:")
        root_parser.print_help()
        sys.exit(1)
