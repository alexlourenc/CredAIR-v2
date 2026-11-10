import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from api.simular import PARAMETROS, handler
from credair.simulador import avaliar

PROPOSTA = {"idade": 30, "renda_mensal": 3000, "tempo_emprego_anos": 5, "score_credito": 700,
            "valor_solicitado": 12000, "prazo_meses": 24, "dividas_ativas": 0}


@pytest.fixture(scope="module")
def url():
    # Sobe a função do Vercel num servidor local, numa porta livre, só durante os testes
    servidor = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=servidor.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{servidor.server_port}/api/simular"
    servidor.shutdown()


def enviar(url, dados):
    pedido = urllib.request.Request(url, data=json.dumps(dados).encode(), method="POST")
    with urllib.request.urlopen(pedido) as resposta:
        return json.load(resposta)


def test_lista_as_versoes(url):
    with urllib.request.urlopen(url) as resposta:
        versoes = json.load(resposta)
    assert versoes["padrao"] == "final"
    assert set(versoes["versoes"]) == {"final", "sem_comprometimento", "so_score"}
    assert versoes["versoes"]["final"]["limiar"] == pytest.approx(0.31)


def test_avalia_com_versao_e_limiar(url):
    resposta = enviar(url, {**PROPOSTA, "versao": "sem_comprometimento", "limiar": 0.4})
    esperado = avaliar(PROPOSTA, PARAMETROS, "sem_comprometimento", 0.4)
    assert resposta["probabilidade"] == pytest.approx(esperado["probabilidade"])
    assert resposta["limiar"] == pytest.approx(0.4)
    assert resposta["versao"] == "sem_comprometimento"


def test_sem_versao_usa_o_modelo_final(url):
    assert enviar(url, PROPOSTA)["versao"] == "final"


@pytest.mark.parametrize("dados", [{**PROPOSTA, "prazo_meses": 0}, {**PROPOSTA, "versao": "estado"}, [1, 2]])
def test_rejeita_pedido_invalido(url, dados):
    with pytest.raises(urllib.error.HTTPError) as erro:
        enviar(url, dados)
    assert erro.value.code == 400
