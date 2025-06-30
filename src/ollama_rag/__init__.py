import sys
import argparse
import logging
import ollama_rag.commands as commands

from phoenix.otel import register


def setup_tracing():
    register(
        project_name="local-llm-rag",
        auto_instrument=True,
    )


def setup_logging(verbose: bool):
    logging.getLogger("pdfminer").setLevel(logging.WARNING)
    logging.getLogger("unstructured").setLevel(logging.WARNING)

    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level)


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
        "--no-tracing",
        action="store_true",
        help="Disable tracing logging",
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
    learn_parser.set_defaults(func=commands.learn)

    query_parser = subparsers.add_parser("query")
    query_parser.add_argument("query", type=str, help="Query")
    query_parser.set_defaults(func=commands.query)

    watch_parser = subparsers.add_parser("watch")
    watch_parser.add_argument("path", type=str, help="Watch directory")
    watch_parser.set_defaults(func=commands.watch)

    args = root_parser.parse_args()

    setup_logging(args.verbose)
    if not args.no_tracing:
        setup_tracing()

    if hasattr(args, "func"):
        args.func(args)
    else:
        print("No subcommand specified. Please use one of the following:")
        root_parser.print_help()
        sys.exit(1)
