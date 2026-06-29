"""
Benchmark: Prompt Caching cu Anthropic

Trimite aceeași întrebare de 2 ori cu un system prompt fix.
Prima dată: cache_creation_input_tokens (se creează cache-ul)
A doua dată: cache_read_input_tokens (se citește din cache)

Compară: tokeni, latență, cost.
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "skillab-py" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from skillab import get_llm

SYSTEM_PROMPT = """Ești un asistent expert în achiziții publice din România.
Răspunzi concis și precis la întrebări despre licitații, contracte, furnizori și legislație.
Folosești terminologia oficială din SEAP (Sistemul Electronic de Achiziții Publice).
Când citezi sume, specifici moneda și dacă include TVA.
Răspunsurile tale sunt structurate cu bullet points sau tabele când e cazul.

Reguli importante de care ții cont:
- Achizițiile directe se fac pentru valori sub 135.060 lei fără TVA pentru produse și servicii
- Achizițiile directe pentru lucrări se fac sub 450.200 lei fără TVA
- Procedura simplificată se aplică între 135.060 și 1.000.000 lei pentru produse și servicii
- Licitația deschisă se aplică pentru valori peste pragurile europene
- Criteriile de atribuire pot fi: prețul cel mai scăzut, costul cel mai scăzut, cel mai bun raport calitate-preț
- Termenele de publicare variază în funcție de tipul procedurii și valoarea estimată
- Documentația de atribuire trebuie să conțină: fișa de date, caiet de sarcini, formulare, contract model
- Evaluarea ofertelor se face în ordinea: admisibilitate, conformitate, evaluare tehnică, evaluare financiară
- Contestațiile se depun la CNSC sau instanța competentă în termenele prevăzute de lege
- Contractul se semnează după expirarea termenului de așteptare de 11 sau 6 zile

Legislație relevantă:
- Legea 98/2016 privind achizițiile publice
- Legea 99/2016 privind achizițiile sectoriale
- Legea 100/2016 privind concesiunile de lucrări și servicii
- Legea 101/2016 privind remediile și căile de atac
- HG 395/2016 norme metodologice de aplicare a Legii 98/2016
- OUG 107/2017 modificări ale legislației achizițiilor publice
- Ordinul ANAP 1.017/2019 privind documentele constatatoare
- Instrucțiunile ANAP privind utilizarea SEAP/SICAP

Tipuri de proceduri de atribuire:
1. Achiziția directă - cel mai simplu tip, fără publicare obligatorie în SEAP pentru valori foarte mici
2. Procedura simplificată - publicare în SEAP, termene reduse
3. Licitația deschisă - cea mai transparentă, publicare în SEAP și eventual JOUE
4. Licitația restrânsă - în două etape: preselectare și apoi depunere oferte
5. Negocierea competitivă - cu publicare prealabilă, permite negociere
6. Dialogul competitiv - pentru proiecte complexe unde soluția nu e predefinită
7. Parteneriatul pentru inovare - pentru soluții inovatoare inexistente pe piață
8. Negocierea fără publicare - excepțional, doar în cazuri justificate de lege
""" * 3  # Repetăm ca să depășim minimul de 1024 tokeni necesar pentru caching


QUERY = "Care sunt pașii pentru o achiziție directă sub 135.060 lei?"


def run_benchmark():
    llm = get_llm(provider="anthropic")
    print(f"Model: {llm.model}")
    print(f"System prompt: {len(SYSTEM_PROMPT)} caractere")
    print("=" * 70)

    results = []

    for i in range(2):
        label = "Call 1 (cache creation)" if i == 0 else "Call 2 (cache read)"
        print(f"\n{label}...")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": QUERY},
        ]

        start = time.time()
        answer = llm.generate_sync(messages)
        elapsed = time.time() - start

        usage = llm.last_usage
        results.append({
            "label": label,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_creation": getattr(usage, "cache_creation_input_tokens", 0) or 0,
            "cache_read": getattr(usage, "cache_read_input_tokens", 0) or 0,
            "latency": elapsed,
        })

        print(f"  Answer: {answer[:100]}...")

    # Print comparison table
    print("\n" + "=" * 70)
    print(f"{'Metric':<30} {'Call 1 (create)':<20} {'Call 2 (read)':<20}")
    print("-" * 70)

    r1, r2 = results
    rows = [
        ("Input tokens", r1["input_tokens"], r2["input_tokens"]),
        ("Output tokens", r1["output_tokens"], r2["output_tokens"]),
        ("Cache creation tokens", r1["cache_creation"], r2["cache_creation"]),
        ("Cache read tokens", r1["cache_read"], r2["cache_read"]),
        ("Latency (s)", f"{r1['latency']:.2f}", f"{r2['latency']:.2f}"),
    ]

    for name, v1, v2 in rows:
        print(f"{name:<30} {str(v1):<20} {str(v2):<20}")

    # Savings
    if r1["cache_creation"] > 0 and r2["cache_read"] > 0:
        print("\n" + "=" * 70)
        print("ECONOMII:")
        saved_tokens = r2["cache_read"]
        print(f"  Tokeni citiți din cache: {saved_tokens}")
        print(f"  Cache read cost: 0.1x vs input normal")
        if r2["latency"] < r1["latency"]:
            pct = (1 - r2["latency"] / r1["latency"]) * 100
            print(f"  Latență redusă cu: {pct:.0f}%")
    else:
        print("\nCache-ul nu s-a activat. System prompt-ul poate fi prea scurt (minim ~1024 tokeni).")


if __name__ == "__main__":
    run_benchmark()
