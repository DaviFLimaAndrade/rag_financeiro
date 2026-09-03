import sys

from rag_financeiro import config
from rag_financeiro.cache.importer import import_from_file

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python scripts/import_cache_table.py <arquivo.md>")
        sys.exit(1)

    entries = import_from_file(sys.argv[1])
    print(f"{len(entries)} entradas no cache. Salvo em {config.CACHE_PATH}")
