"""Função do Vercel: recebe uma proposta em JSON e devolve a avaliação do modelo (Fase 14).

Para testar no computador, rode `python api/simular.py` e abra http://localhost:8000.
"""

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# No Vercel o pacote não é instalado em modo editável: importa direto da pasta src/
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from credair.simulador import avaliar  # noqa: E402

# Parâmetros das versões do modelo, gerados com `python -m credair.simulador`
PARAMETROS = json.loads((RAIZ / "api" / "modelos.json").read_text(encoding="utf-8"))

# Resumo das versões para a página montar o seletor e o controle do limiar
VERSOES = {
    "padrao": PARAMETROS["padrao"],
    "versoes": {chave: {"nome": v["nome"], "limiar": v["limiar"], "auc_cv": v["auc_cv"]}
                for chave, v in PARAMETROS["versoes"].items()},
}


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            tamanho = int(self.headers.get("Content-Length", 0))
            pedido = json.loads(self.rfile.read(tamanho) or b"{}")
            if not isinstance(pedido, dict):
                raise ValueError("O pedido deve ser um objeto JSON")
            versao, limiar = pedido.pop("versao", None), pedido.pop("limiar", None)
            self._responder(200, avaliar(pedido, PARAMETROS, versao, limiar))
        except (ValueError, TypeError) as erro:
            self._responder(400, {"erro": str(erro)})

    def do_GET(self):
        if self.path.startswith("/api/simular"):
            return self._responder(200, VERSOES)
        # Só usado no teste local: serve a página de public/
        pagina = RAIZ / "public" / "index.html"
        if self.path not in ("/", "/index.html") or not pagina.exists():
            return self._responder(404, {"erro": "Não encontrado"})
        corpo = pagina.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(corpo)

    def _responder(self, status, dados):
        corpo = json.dumps(dados, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(corpo)


if __name__ == "__main__":
    # Porta opcional, para quando a 8000 já estiver em uso: python api/simular.py 8001
    porta = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"Simulador em http://localhost:{porta}")
    ThreadingHTTPServer(("localhost", porta), handler).serve_forever()
