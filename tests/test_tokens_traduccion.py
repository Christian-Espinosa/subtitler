"""21/11/2026: deepseek-flash reasoned 0-8,000 tokens per batch (once the whole cap, with an empty answer). Thinking is
disabled for DeepSeek and each batch asks only for what its lines can take."""
import io
import json

import llm_translate as lt


def test_el_tope_va_con_las_lineas_y_nunca_pasa_del_maximo():
    diez = [{"i": k, "text": "x", "max_chars": 60} for k in range(10)]
    assert lt.tope(diez) == 300 + 10 * (60 + 15)
    assert lt.tope([{"i": k, "max_chars": 500} for k in range(100)]) == lt.MAX_TOKENS


def test_deepseek_va_sin_razonar_y_con_el_tope_del_lote(monkeypatch):
    enviado = {}

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def urlopen(req, timeout=0):
        enviado.update(json.loads(req.data))
        return Resp(json.dumps({"choices": [{"message": {"content": json.dumps({"cues": [{"i": 1, "text": "hola"}]})}}], "usage": {}}).encode())

    monkeypatch.setattr(lt.urllib.request, "urlopen", urlopen)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    ds = [p for p in lt.PROVIDERS if p.name == "deepseek"]
    out = lt.translate_cues([(0.0, 2.0, "hello")], "en", "es", providers=ds)
    assert out == ["hola"]
    assert enviado["thinking"] == {"type": "disabled"}
    assert enviado["max_tokens"] == lt.tope([{"max_chars": lt.budget(0.0, 2.0, "es")}])
    nv = [p for p in lt.PROVIDERS if p.name == "nvidia"][0]
    assert "thinking" not in nv.extra          # only where it was measured
