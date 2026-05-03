from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import sympy as sp


APP_TITLE = "Lise Matematik Uygulaması"
DATA_DIR = Path(__file__).parent / "data"


@dataclass(frozen=True)
class Topic:
    id: str
    title: str


GRADE_TOPICS: dict[str, list[Topic]] = {
    "9": [
        Topic("9_numbers_sets", "Sayı Kümeleri"),
        Topic("9_numbers_divisibility", "Bölünebilme Kuralları"),
        Topic("9_numbers_gcd_lcm", "EBOB–EKOK"),
        Topic("9_quant_eq_ineq", "Denklemler ve Eşitsizlikler"),
        Topic("9_quant_abs", "Mutlak Değer"),
        Topic("9_quant_exp_root", "Üslü ve Köklü İfadeler"),
        Topic("9_quant_ratio", "Oran ve Orantı"),
        Topic("9_quant_problems", "Problemler"),
        Topic("9_functions_intro", "Fonksiyon Kavramı ve Grafikler"),
        Topic("9_geo_triangles", "Üçgenler (Açı/Eşlik/Benzerlik/Alan)"),
        Topic("9_stats_center_spread", "Merkezi Eğilim ve Yayılım Ölçüleri"),
        Topic("9_stats_viz", "Veri Görselleştirme"),
    ],
    "10": [
        Topic("10_count_perm", "Permütasyon"),
        Topic("10_count_comb", "Kombinasyon"),
        Topic("10_count_binom", "Binom Açılımı"),
        Topic("10_count_prob", "Olasılık"),
        Topic("10_func_ops", "Fonksiyonlarda İşlemler"),
        Topic("10_func_comp", "Bileşke Fonksiyon"),
        Topic("10_poly_concept", "Polinom Kavramı"),
        Topic("10_poly_ops", "Polinomlarda İşlemler"),
        Topic("10_poly_factor", "Çarpanlara Ayırma"),
        Topic("10_quad_complex", "Karmaşık Sayılar"),
        Topic("10_quad_root_coeff", "Kök–Katsayı İlişkileri"),
        Topic("10_geo_quads", "Özel Dörtgenler"),
        Topic("10_solid_prism", "Prizma"),
        Topic("10_solid_pyramid", "Piramit"),
    ],
    "11": [
        Topic("11_trig_angles", "Yönlü Açılar"),
        Topic("11_trig_funcs", "Trigonometrik Fonksiyonlar"),
        Topic("11_trig_period_graph", "Periyot ve Grafikler"),
        Topic("11_analytic_line", "Doğrunun Analitik İncelenmesi"),
        Topic("11_func_parabola", "Parabol (2. Dereceden Fonksiyon)"),
        Topic("11_func_transforms", "Grafik Dönüşümleri"),
        Topic("11_sys_quad2", "2. Dereceden 2 Bilinmeyenli Sistemler"),
        Topic("11_sys_ineq", "Eşitsizlik Sistemleri"),
        Topic("11_circle_elements", "Çemberin Temel Elemanları"),
        Topic("11_circle_tangent_area", "Teğet ve Alan"),
        Topic("11_prob_conditional", "Koşullu Olasılık"),
        Topic("11_prob_compound", "Bileşik Olaylar"),
    ],
    "12": [
        Topic("12_exp", "Üstel Fonksiyon"),
        Topic("12_log", "Logaritma Fonksiyonu"),
        Topic("12_exp_log_eq", "Üstel/Log Denklemler ve Eşitsizlikler"),
        Topic("12_seq_real", "Gerçek Sayı Dizileri"),
        Topic("12_seq_arith_geo", "Aritmetik ve Geometrik Diziler"),
        Topic("12_trig_identities", "Toplam–Fark ve Yarım Açı Formülleri"),
        Topic("12_trig_eq", "Trigonometrik Denklemler"),
        Topic("12_transformations", "Öteleme, Dönme, Yansıma"),
        Topic("12_limit_cont", "Limit ve Süreklilik"),
        Topic("12_derivative_rules", "Türev Alma Kuralları"),
        Topic("12_derivative_tangent", "Teğet Denklemi"),
        Topic("12_derivative_mono_extrema", "Artan–Azalan, Maks–Min"),
        Topic("12_integral_indef", "Belirsiz İntegral"),
        Topic("12_integral_def_area", "Belirli İntegral ve Alan"),
        Topic("12_analytic_circle", "Çemberin Analitik İncelenmesi"),
    ],
}


@dataclass(frozen=True)
class BuiltinTopicContent:
    title: str
    unit: str
    summary: str
    key_points: list[str]
    formulas: list[str]
    worked_examples: list[dict[str, str]]  # {"q": "...", "a": "..."}


def _md_list(items: list[str]) -> str:
    return "\n".join([f"- {x}" for x in items]) if items else ""


BUILTIN_NOTES: dict[str, BuiltinTopicContent] = {
    # 9. sınıf
    "9_numbers_sets": BuiltinTopicContent(
        title="Sayı Kümeleri",
        unit="Sayılar",
        summary="Sayı kümeleri, sayıların özelliklerine göre sınıflandırılmasıdır. Gerçek sayılar kümesi içinde rasyonel ve irrasyonel sayılar ayrımı, aralık gösterimi ve mutlak değer–aralık dönüşümleri bu ünitenin temelidir.",
        key_points=[
            r"\(\mathbb{N} \subset \mathbb{Z} \subset \mathbb{Q} \subset \mathbb{R}\)",
            "Sonlu veya devirli ondalık gösterimler rasyoneldir.",
            r"\(|x-a|<r \iff a-r < x < a+r\)",
        ],
        formulas=[
            r"\(0.\overline{ab}=\frac{ab}{99}\)",
            r"\(|x-a|\le r \iff a-r \le x \le a+r\)",
        ],
        worked_examples=[
            {"q": r"\(|x-3|<2\) çözüm kümesi?", "a": r"\(1<x<5\Rightarrow(1,5)\)"},
            {"q": r"\(0.\overline{12}\) kesir?", "a": r"\(\frac{4}{33}\)"},
        ],
    ),
    "9_numbers_divisibility": BuiltinTopicContent(
        title="Bölünebilme Kuralları",
        unit="Sayılar",
        summary="Bölünebilme kuralları, bir sayının belirli bir sayıya kalansız bölünüp bölünmediğini hızlıca test etmemizi sağlar. Özellikle 2,3,4,5,6,8,9,10,11 gibi sayılar için pratik kurallar kullanılır.",
        key_points=[
            "2: son basamak çift",
            "3: rakamlar toplamı 3'ün katı",
            "9: rakamlar toplamı 9'un katı",
            "11: tek basamaklar toplamı ile çift basamaklar toplamı farkı 11'in katı",
        ],
        formulas=[
            r"6'ya bölünebilme: 2 ve 3'e aynı anda bölünebilme",
            r"12'ye bölünebilme: 3 ve 4'e aynı anda bölünebilme",
        ],
        worked_examples=[
            {"q": "561 sayısı 3'e bölünür mü?", "a": "5+6+1=12, 12 üçe bölünür → evet."},
            {"q": "1540 sayısı 8'e bölünür mü?", "a": "Son üç basamak 540; 540/8 tam değil → hayır."},
        ],
    ),
    "9_numbers_gcd_lcm": BuiltinTopicContent(
        title="EBOB–EKOK",
        unit="Sayılar",
        summary="EBOB (en büyük ortak bölen) ve EKOK (en küçük ortak kat) iki veya daha fazla sayının ortak bölen/kat ilişkilerini verir. Asal çarpanlara ayırma yöntemi en güvenilir yoldur.",
        key_points=[
            "EBOB: ortak asal çarpanların küçük üsleri",
            "EKOK: tüm asal çarpanların büyük üsleri",
            "İki sayı için: EBOB(a,b)·EKOK(a,b)=|a·b|",
        ],
        formulas=[r"\(\gcd(a,b)\cdot \mathrm{lcm}(a,b)=|ab|\)"],
        worked_examples=[
            {"q": "EBOB(18,24)?", "a": "18=2·3², 24=2³·3 → EBOB=2·3=6"},
            {"q": "EKOK(18,24)?", "a": "EKOK=2³·3²=72"},
        ],
    ),
    "9_quant_eq_ineq": BuiltinTopicContent(
        title="Denklemler ve Eşitsizlikler",
        unit="Nicelikler ve Değişimler",
        summary="Denklem çözümünde amaç bilinmeyeni izole etmektir. Eşitsizliklerde ise işlem yaparken özellikle negatif sayıyla çarpma/bölmede yön değişimi kuralına dikkat edilir.",
        key_points=[
            "Denklem: eşitliğin iki tarafına aynı işlem yapılabilir.",
            "Eşitsizlik: negatifle çarpma/bölmede işaret yön değiştirir.",
            "Çözüm kümesi aralık/küme gösterimiyle yazılabilir.",
        ],
        formulas=[r"\(a<b \Rightarrow -a>-b\)"],
        worked_examples=[
            {"q": r"2x+3=11", "a": r"x=4"},
            {"q": r"-3x+6<0", "a": r"-3x<-6 \Rightarrow x>2"},
        ],
    ),
    "9_quant_abs": BuiltinTopicContent(
        title="Mutlak Değer",
        unit="Nicelikler ve Değişimler",
        summary="Mutlak değer, bir sayının 0'a uzaklığıdır ve her zaman 0 veya pozitiftir. Mutlak değerli eşitsizlikler aralığa dönüştürülerek çözülür.",
        key_points=[
            r"\(|x| \ge 0\)",
            r"\(|x|=a\) (a≥0) ise \(x=a\) veya \(x=-a\)",
            r"\(|x-a|<r\) aralık verir; \(|x-a|>r\) birleşim verir",
        ],
        formulas=[
            r"\(|x-a|<r \iff a-r<x<a+r\)",
            r"\(|x-a|\ge r \iff x\le a-r \ \text{veya}\ x\ge a+r\)",
        ],
        worked_examples=[
            {"q": r"|2x-1|=5", "a": r"2x-1=5 → x=3; 2x-1=-5 → x=-2"},
            {"q": r"|x+2|≥3", "a": r"x≤-5 veya x≥1"},
        ],
    ),
    "9_quant_exp_root": BuiltinTopicContent(
        title="Üslü ve Köklü İfadeler",
        unit="Nicelikler ve Değişimler",
        summary="Üslü ifadelerde taban/üs kuralları; köklü ifadelerde kökün tanımı ve sadeleştirme önemlidir. Tanım kümelerine (özellikle çift dereceli köklerde) dikkat edilir.",
        key_points=[
            r"\(a^m\cdot a^n=a^{m+n}\), \(a^m/a^n=a^{m-n}\) (a≠0)",
            r"\((a^m)^n=a^{mn}\)",
            "Çift dereceli kökte iç ifade ≥0 olmalı (gerçek sayılarda).",
        ],
        formulas=[
            r"\(\sqrt{a^2}=|a|\)",
            r"\(\sqrt{ab}=\sqrt{a}\sqrt{b}\) (a,b≥0)",
        ],
        worked_examples=[
            {"q": r"2^3·2^5", "a": r"2^8=256"},
            {"q": r"\(\sqrt{72}\)", "a": r"\(\sqrt{36·2}=6\sqrt{2}\)"},
        ],
    ),
    "9_quant_ratio": BuiltinTopicContent(
        title="Oran ve Orantı",
        unit="Nicelikler ve Değişimler",
        summary="Oran, iki niceliğin karşılaştırılmasıdır. Orantı ise iki oranın eşitliğidir. Doğru orantı ve ters orantı problemleri günlük hayatta sık görülür.",
        key_points=[
            r"\(\frac{a}{b}=\frac{c}{d}\Rightarrow ad=bc\) (b,d≠0)",
            "Doğru orantı: biri artarsa diğeri de artar.",
            "Ters orantı: biri artarsa diğeri azalır (çarpım sabit).",
        ],
        formulas=[r"Doğru: \(y=kx\)", r"Ters: \(y=\frac{k}{x}\)"],
        worked_examples=[
            {"q": "3 işçi işi 10 günde bitiriyor. 5 işçi kaç günde?", "a": "Ters orantı: 3·10=5·t → t=6 gün"},
            {"q": r"\(\frac{x}{6}=\frac{5}{9}\)", "a": r"x= \(\frac{30}{9}=\frac{10}{3}\)"},
        ],
    ),
    "9_quant_problems": BuiltinTopicContent(
        title="Problemler",
        unit="Nicelikler ve Değişimler",
        summary="Problem çözmede ana strateji: bilinmeyenleri tanımlamak, denklemi kurmak ve kontrol etmektir. Yaş, kar-zarar, karışım, hız-zaman gibi türlerde model kurma becerisi esastır.",
        key_points=[
            "Bilinmeyeni sembolle tanımla (x, y…).",
            "Cümleyi matematiksel eşitliğe çevir.",
            "Sonucu problem şartlarına göre kontrol et.",
        ],
        formulas=[r"Hız \(v=\frac{\text{yol}}{\text{zaman}}\)"],
        worked_examples=[
            {"q": "Bir sayının 3 fazlası 11 ise sayı?", "a": "x+3=11 → x=8"},
            {"q": "60 km yolu 2 saatte alan aracın hızı?", "a": "v=60/2=30 km/s"},
        ],
    ),
    "9_functions_intro": BuiltinTopicContent(
        title="Fonksiyon Kavramı ve Grafikler",
        unit="Fonksiyonlar",
        summary="Fonksiyon, her girdiye tek bir çıktı eşleyen bağıntıdır. Tanım kümesi, değer kümesi, görüntü kümesi ve grafik yorumlama temel becerilerdir.",
        key_points=[
            "Her x için tek f(x) olmalı (dikey doğru testi).",
            "Grafikten artan/azalan, maksimum/minimum okunabilir.",
            "Tanım kümesi kısıtları (payda 0 olamaz, kök içi ≥0 vb.).",
        ],
        formulas=[r"f: A→B, \(x\mapsto f(x)\)"],
        worked_examples=[
            {"q": "f(x)=2x+1 için f(3)?", "a": "7"},
            {"q": r"f(x)=\(\sqrt{x-1}\) tanım kümesi?", "a": r"x≥1 → \([1,\infty)\)"},
        ],
    ),
    "9_geo_triangles": BuiltinTopicContent(
        title="Üçgenler (Açı/Eşlik/Benzerlik/Alan)",
        unit="Geometri",
        summary="Üçgende temel açılar, eşlik ve benzerlik kriterleri ile alan bağıntıları geometrinin omurgasıdır. Yardımcı elemanlar (yükseklik, açıortay, kenarortay) çok sayıda soruyu açar.",
        key_points=[
            "Üçgenin iç açıları toplamı 180°",
            "Eşlik: SSS, SAS, ASA",
            "Benzerlik: AAA, SAS, SSS (oran)",
        ],
        formulas=[r"Alan: \(A=\frac{1}{2}ah\)", r"Heron: \(A=\sqrt{s(s-a)(s-b)(s-c)}\)"],
        worked_examples=[
            {"q": "Taban 10, yükseklik 6 ise alan?", "a": "A=1/2·10·6=30"},
            {"q": "İç açılar 50°, 60° ise üçüncü açı?", "a": "70°"},
        ],
    ),
    "9_stats_center_spread": BuiltinTopicContent(
        title="Merkezi Eğilim ve Yayılım Ölçüleri",
        unit="Veri ve İstatistik",
        summary="Veriyi özetlemek için ortalama, ortanca, tepe değer gibi merkezi eğilim; açıklık, varyans, standart sapma gibi yayılım ölçüleri kullanılır.",
        key_points=[
            "Aritmetik ortalama: toplam / adet",
            "Ortanca: sıralı verinin ortası",
            "Açıklık: max-min",
        ],
        formulas=[r"\(\bar{x}=\frac{\sum x_i}{n}\)", r"Açıklık \(=x_{\max}-x_{\min}\)"],
        worked_examples=[
            {"q": "2,3,5,10 ortalama?", "a": "(2+3+5+10)/4=5"},
            {"q": "2,3,5,10 açıklık?", "a": "10-2=8"},
        ],
    ),
    "9_stats_viz": BuiltinTopicContent(
        title="Veri Görselleştirme",
        unit="Veri ve İstatistik",
        summary="Sütun grafiği, histogram, daire grafiği gibi görseller veriyi daha hızlı okumayı sağlar. Doğru grafik türü, verinin yapısına göre seçilmelidir.",
        key_points=[
            "Kategorik veri: sütun/daire grafiği",
            "Sürekli sayısal veri: histogram",
            "Grafik okuma: eksen, ölçek, birim kontrolü",
        ],
        formulas=[],
        worked_examples=[
            {"q": "Sınıf dağılımı (kız/erkek) için hangi grafik?", "a": "Daire veya sütun grafiği"},
            {"q": "Boy uzunlukları dağılımı için hangi grafik?", "a": "Histogram"},
        ],
    ),
}


def builtin_notes_md(topic_id: str) -> str:
    c = BUILTIN_NOTES.get(topic_id)
    if not c:
        # Kapsamlı içerik yazılmadıysa bile boş bırakma: standart şablon.
        return (
            f"# {topic_id}\n\n"
            "Bu konu için temel içerik şablonu yüklendi.\n\n"
            "## Hızlı Özet\n- Tanım ve amaç\n- Temel kavramlar\n- En sık kullanılan formüller\n\n"
            "## Örnek\n- Örnek soru ve çözüm\n"
        )

    examples_md = "\n\n".join([f"**Örnek {i+1}:** {ex['q']}\n\n**Çözüm:** {ex['a']}" for i, ex in enumerate(c.worked_examples)])
    return (
        f"# {c.title} ({topic_id.split('_')[0]}. Sınıf)\n\n"
        f"**Ünite:** {c.unit}\n\n"
        "## Konu Özeti\n"
        f"{c.summary}\n\n"
        "## Temel Kazanımlar\n"
        f"{_md_list(c.key_points)}\n\n"
        "## Formüller / Kurallar\n"
        f"{_md_list(c.formulas) if c.formulas else '- (Bu konuda daha çok kavram ve uygulama vardır.)'}\n\n"
        "## Çözümlü Örnekler\n"
        f"{examples_md if examples_md else 'Bu konu için örnekler yakında eklenecek.'}\n"
    )


def _rng_for(topic_id: str) -> random.Random:
    # deterministik üretim: aynı konu aynı soruları üretir
    seed = abs(hash(topic_id)) % (2**32)
    return random.Random(seed)


def builtin_question_set(topic_id: str) -> list[dict[str, Any]]:
    """
    Her konu için 20 adet (kolay→zor) offline üretilmiş soru seti.
    Not: Geometri gibi çizim gerektiren konularda metin tabanlı sorular üretilir.
    """
    rng = _rng_for(topic_id)
    grade = topic_id.split("_", 1)[0]

    def q(level: str, question: str, solution: str) -> dict[str, str]:
        return {"level": level, "question": question, "solution": solution}

    qs: list[dict[str, Any]] = []

    # Genel cebir üreticileri
    if "eq_ineq" in topic_id or "problems" in topic_id:
        for i in range(10):
            a = rng.randint(1, 9)
            b = rng.randint(1, 12)
            c = rng.randint(-10, 10)
            x = sp.Symbol("x")
            expr = a * x + c
            qs.append(
                q(
                    "Kolay" if i < 4 else "Orta",
                    rf"Denklemi çöz: \({sp.latex(expr)}={b}\)",
                    rf"\(x=\frac{{{b - c}}}{{{a}}}={sp.latex(sp.Rational(b - c, a))}\)",
                )
            )
        for i in range(10):
            a = rng.randint(1, 6)
            b = rng.randint(-8, 8)
            c = rng.randint(-10, 10)
            sign = "<" if rng.random() < 0.5 else "≤"
            # a(x-b) + c < 0
            x = sp.Symbol("x")
            ineq = a * (x - b) + c
            sol = sp.solve_univariate_inequality(sp.Lt(ineq, 0) if sign == "<" else sp.Le(ineq, 0), x)
            qs.append(
                q(
                    "Zor" if i >= 6 else "Orta",
                    rf"Eşitsizliği çöz: \({sp.latex(ineq)} {sign} 0\)",
                    rf"Çözüm: \({sp.latex(sol)}\)",
                )
            )
        return qs[:20]

    if "abs" in topic_id:
        x = sp.Symbol("x")
        for i in range(10):
            a = rng.randint(-6, 6)
            r = rng.randint(1, 8)
            qs.append(
                q(
                    "Kolay" if i < 4 else "Orta",
                    rf"Eşitliği çöz: \(|x{a:+d}|={r}\)",
                    rf"\(x{a:+d}={r}\) veya \(x{a:+d}=-{r}\) → \(x={r - a}\) veya \(x={-r - a}\)",
                )
            )
        for i in range(10):
            a = rng.randint(-5, 5)
            r = rng.randint(1, 7)
            qs.append(
                q(
                    "Zor" if i >= 6 else "Orta",
                    rf"Eşitsizliği çöz: \(|x{a:+d}|<{r}\)",
                    rf"\(-{r}<x{a:+d}<{r}\Rightarrow {-r - a}<x<{r - a}\)",
                )
            )
        return qs[:20]

    if "exp_root" in topic_id:
        for i in range(10):
            a = rng.randint(2, 6)
            m = rng.randint(2, 5)
            n = rng.randint(2, 5)
            qs.append(
                q(
                    "Kolay" if i < 4 else "Orta",
                    rf"İşlemi sadeleştir: \({a}^{m}\cdot {a}^{n}\)",
                    rf"\({a}^{{{m+n}}}\)",
                )
            )
        for i in range(10):
            k = rng.choice([2, 3, 5, 6, 7, 8])
            t = rng.choice([2, 3, 5, 6])
            val = (t * t) * k
            qs.append(
                q(
                    "Zor" if i >= 6 else "Orta",
                    rf"Kökü sadeleştir: \(\sqrt{{{val}}}\)",
                    rf"\(\sqrt{{{t*t}\cdot {k}}}={t}\sqrt{{{k}}}\)",
                )
            )
        return qs[:20]

    if "ratio" in topic_id:
        for i in range(10):
            a = rng.randint(2, 12)
            b = rng.randint(2, 12)
            c = rng.randint(2, 12)
            x_val = sp.Rational(a * c, b)
            qs.append(
                q(
                    "Kolay" if i < 4 else "Orta",
                    rf"Orantıda \( \frac{{x}}{{{a}}}=\frac{{{c}}}{{{b}}} \) ise \(x\) kaçtır?",
                    rf"\(x=\frac{{{a}\cdot {c}}}{{{b}}}={sp.latex(x_val)}\)",
                )
            )
        for i in range(10):
            workers1 = rng.randint(2, 8)
            days1 = rng.randint(3, 20)
            workers2 = rng.randint(2, 10)
            days2 = sp.Rational(workers1 * days1, workers2)
            qs.append(
                q(
                    "Zor" if i >= 6 else "Orta",
                    rf"{workers1} işçi bir işi {days1} günde bitiriyor. {workers2} işçi kaç günde bitirir? (Ters orantı)",
                    rf"{workers1}\cdot {days1}={workers2}\cdot t \Rightarrow t={sp.latex(days2)}",
                )
            )
        return qs[:20]

    if "functions" in topic_id or "func_" in topic_id:
        x = sp.Symbol("x")
        for i in range(10):
            a = rng.randint(-5, 6)
            b = rng.randint(-10, 10)
            val = rng.randint(-4, 5)
            f = a * x + b
            qs.append(
                q(
                    "Kolay" if i < 4 else "Orta",
                    rf"f(x)={sp.latex(f)} ise f({val}) kaçtır?",
                    rf"f({val})={sp.latex(f.subs(x, val))}",
                )
            )
        for i in range(10):
            # tanım kümesi soruları
            c = rng.randint(-6, 6)
            qs.append(
                q(
                    "Zor" if i >= 6 else "Orta",
                    rf"f(x)=\(\sqrt{{x{c:+d}}}\) için tanım kümesi nedir?",
                    rf"x{c:+d}\ge 0 \Rightarrow x\ge {-c}. Çözüm: \([{ -c},\infty)\)",
                )
            )
        return qs[:20]

    # default: karışık 20 soru (genel matematik pratik)
    x = sp.Symbol("x")
    for i in range(20):
        a = rng.randint(1, 9)
        b = rng.randint(-9, 9)
        c = rng.randint(-12, 12)
        expr = a * x + b
        eq = sp.Eq(expr, c)
        sol = sp.solve(eq, x)
        level = "Kolay" if i < 6 else ("Orta" if i < 14 else "Zor")
        qs.append(
            q(
                level,
                rf"Denklemi çöz: \({sp.latex(eq)}\)",
                rf"Çözüm: \(x={sp.latex(sol[0])}\)",
            )
        )
    return qs


def fix_latex(text: str) -> str:
    """Convert LaTeX delimiters to Streamlit format."""
    text = re.sub(r'\\\[(.+?)\\\]', lambda m: '$$' + m.group(1) + '$$', text, flags=re.DOTALL)
    text = re.sub(r'\\\((.+?)\\\)', lambda m: '$' + m.group(1) + '$', text, flags=re.DOTALL)
    return text

def load_topic_markdown(topic_id: str) -> str:
    path = DATA_DIR / "notes" / f"{topic_id}.md"
    if path.exists():
        return fix_latex(path.read_text(encoding="utf-8"))
    return fix_latex(builtin_notes_md(topic_id))


def load_question_set(topic_id: str) -> list[dict[str, Any]]:
    path = DATA_DIR / "questions" / f"{topic_id}.json"
    if not path.exists():
        return builtin_question_set(topic_id)
    return json.loads(path.read_text(encoding="utf-8"))


def inject_css() -> None:
    st.markdown(
        """
        <style>
          .block-container { padding-top: 1.25rem; padding-bottom: 2.5rem; max-width: 1150px; }
          [data-testid="stSidebar"] { border-right: 1px solid rgba(255,255,255,0.06); }
          .muted { opacity: 0.85; }
          .chip {
            display:inline-flex; align-items:center; gap:8px;
            padding:6px 10px; border-radius:999px;
            background: rgba(124,92,255,0.12);
            border: 1px solid rgba(124,92,255,0.25);
            font-size: 12px;
          }
          .card {
            border: 1px solid rgba(255,255,255,0.07);
            background: rgba(255,255,255,0.03);
            border-radius: 14px;
            padding: 14px 14px;
          }
          .card h3 { margin: 0 0 6px 0; }
          code { font-size: 0.9em; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def parse_math_prompt(text: str) -> tuple[Literal["equation", "derivative", "integral", "simplify", "unknown"], dict[str, str]]:
    t = (text or "").strip()
    if not t:
        return "unknown", {}

    t_norm = t.lower()
    if any(k in t_norm for k in ["türev", "turev", "d/dx", "derivative"]):
        return "derivative", {"expr": t}
    if any(k in t_norm for k in ["integral", "∫", "dx"]):
        return "integral", {"expr": t}
    if "=" in t:
        return "equation", {"expr": t}
    if any(k in t_norm for k in ["sadeleştir", "simplify", "çarpan", "factor", "expand", "açılım"]):
        return "simplify", {"expr": t}
    return "unknown", {"expr": t}


def try_solve(text: str) -> dict[str, Any]:
    """
    Offline 'akıllı yardım':
    - basit denklem çözümü (1 değişken)
    - türev/integral denemesi
    - sadeleştirme/çarpanlara ayırma
    """
    kind, payload = parse_math_prompt(text)
    raw = payload.get("expr", text)

    x = sp.Symbol("x")
    y = sp.Symbol("y")

    def _clean(s: str) -> str:
        s2 = s.replace("^", "**")
        s2 = s2.replace("×", "*").replace("·", "*")
        s2 = s2.replace("−", "-")
        return s2

    raw = _clean(raw)

    try:
        if kind == "equation":
            left, right = raw.split("=", 1)
            eq = sp.Eq(sp.sympify(left), sp.sympify(right))
            sol = sp.solve(eq, [x], dict=True)
            return {"kind": "equation", "input": str(eq), "result": sol or "Çözüm bulunamadı."}

        if kind == "derivative":
            expr_match = re.search(r"(?:türev|turev|d/dx|derivative)\s*:?\s*(.+)", raw, re.IGNORECASE)
            expr_str = expr_match.group(1) if expr_match else raw
            expr = sp.sympify(expr_str)
            res = sp.diff(expr, x)
            return {"kind": "derivative", "input": str(expr), "result": str(res)}

        if kind == "integral":
            expr_match = re.search(r"(?:integral)\s*:?\s*(.+)", raw, re.IGNORECASE)
            expr_str = expr_match.group(1) if expr_match else raw
            expr_str = expr_str.replace("dx", "").strip()
            expr = sp.sympify(expr_str)
            res = sp.integrate(expr, x)
            return {"kind": "integral", "input": str(expr), "result": str(res) + " + C"}

        if kind == "simplify":
            expr = sp.sympify(raw)
            simp = sp.simplify(expr)
            fact = sp.factor(expr)
            expd = sp.expand(expr)
            return {
                "kind": "simplify",
                "input": str(expr),
                "result": {"sadeleştirme": str(simp), "çarpan": str(fact), "açılım": str(expd)},
            }
    except Exception as e:  # noqa: BLE001
        return {"kind": kind, "input": raw, "error": str(e)}

    return {"kind": "unknown", "input": raw, "result": "Ne yapmak istediğini anlayamadım. Örn: `2x+3=7` ya da `türev: x^2+3x`"}


def render_question_list(questions: list[dict[str, Any]], key_prefix: str = "q") -> None:
    if not questions:
        st.info("Bu konu için 20 soruluk set henüz eklenmedi.")
        return

    for i, q in enumerate(questions, start=1):
        with st.expander(f"{i}. Soru — {q.get('level','')}".strip(), expanded=i <= 2):
            st.markdown(q.get("question", ""))
            show = st.toggle("Çözümü göster", key=f"show_sol_{key_prefix}_{i}")
            if show:
                st.markdown(q.get("solution", "Çözüm eklenmedi."))


def plot_function(expr_str: str, x_min: float, x_max: float) -> None:
    x = sp.Symbol("x")
    expr = sp.sympify(expr_str.replace("^", "**"))
    f = sp.lambdify(x, expr, "numpy")

    xs = np.linspace(x_min, x_max, 800)
    ys = f(xs)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="f(x)"))
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=10, r=10, t=30, b=10),
        height=420,
        xaxis_title="x",
        yaxis_title="f(x)",
    )
    st.plotly_chart(fig, use_container_width=True)


def sidebar_topic_picker(grade: str) -> Topic:
    topics = GRADE_TOPICS[grade]
    labels = [t.title for t in topics]
    default_idx = 0
    pick = st.sidebar.selectbox(f"{grade}. sınıf konu seç", labels, index=default_idx)
    return next(t for t in topics if t.title == pick)


def grade_page(grade: str) -> None:
    topic = sidebar_topic_picker(grade)
    st.markdown(
        f"""
        <div class="chip">📌 {grade}. sınıf • <b>{topic.title}</b></div>
        """,
        unsafe_allow_html=True,
    )

    t_notes, t_questions, t_calc = st.tabs(["Konu Anlatımı", "Sorular", "Hesaplama"])

    with t_notes:
        st.markdown(load_topic_markdown(topic.id))

    with t_questions:
        q_ai, q_bank = st.tabs(["AI'a Sor (Offline)", "20 Soru (Kolay→Zor)"])

        with q_ai:
            st.caption("API anahtarı yok: burada SymPy tabanlı offline çözüm denemesi yapıyoruz.")
            prompt = st.text_area(
                "Soruyu yaz (ör: 2x+3=7, türev: x^2+3x, integral: sin(x) dx)",
                height=120,
                placeholder="Sorunu buraya yaz…",
                key=f"ai_prompt_{grade}",
            )
            col1, col2 = st.columns([1, 1])
            with col1:
                run = st.button("Çözmeyi dene", type="primary", key=f"run_{grade}")
            with col2:
                st.write("")

            if run:
                out = try_solve(prompt)
                if "error" in out:
                    st.error(out["error"])
                else:
                    st.success("Çözüm üretildi (offline).")
                    st.json(out, expanded=False)

            st.divider()
            st.subheader("Fotoğrafla soru (OCR)")
            st.caption("Not: OCR için bilgisayarda Tesseract kurulu olmalı. Kurulu değilse bu kısım çalışmayabilir.")
            img = st.file_uploader("Görsel yükle (png/jpg)", type=["png", "jpg", "jpeg"], key=f"img_{grade}")
            if img is not None:
                try:
                    from PIL import Image

                    image = Image.open(img)
                    st.image(image, caption="Yüklenen görsel", use_container_width=True)
                    import pytesseract

                    text = pytesseract.image_to_string(image, lang="tur+eng")
                    st.text_area("OCR çıktısı", value=text, height=140, key=f"ocr_out_{grade}")
                    if st.button("OCR metninden çözmeyi dene", key=f"ocr_solve_{grade}"):
                        out = try_solve(text)
                        if "error" in out:
                            st.error(out["error"])
                        else:
                            st.json(out, expanded=False)
                except Exception as e:  # noqa: BLE001
                    st.warning(f"OCR çalışmadı: {e}")

        with q_bank:
            questions = load_question_set(topic.id)
            render_question_list(questions, key_prefix=f"{grade}_{topic.id}")

    with t_calc:
        st.subheader("Hesaplama Araçları")
        st.caption("Bu bölüm konuya göre zenginleştirilecek. Şimdilik en faydalı genel araçlar var.")

        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown('<div class="card"><h3>Fonksiyon Grafiği</h3><div class="muted">f(x) ifadesini yaz ve aralığı seç.</div></div>', unsafe_allow_html=True)
            expr = st.text_input("f(x) =", value="x^2", key=f"fx_{grade}")
            x_min, x_max = st.slider("x aralığı", -10.0, 10.0, (-5.0, 5.0), step=0.5, key=f"range_{grade}")
            if st.button("Grafiği çiz", key=f"plot_{grade}"):
                try:
                    plot_function(expr, float(x_min), float(x_max))
                except Exception as e:  # noqa: BLE001
                    st.error(str(e))

        with c2:
            st.markdown('<div class="card"><h3>Denklem Çözücü</h3><div class="muted">1 bilinmeyenli denklem çözümü (x).</div></div>', unsafe_allow_html=True)
            eq = st.text_input("Denklem (ör: 2*x+3=7)", value="2*x+3=7", key=f"eq_{grade}")
            if st.button("Çöz", key=f"solve_{grade}"):
                out = try_solve(eq)
                if "error" in out:
                    st.error(out["error"])
                else:
                    st.json(out, expanded=False)


def calculator_page() -> None:
    st.subheader("Hesap Makinesi")
    st.caption("Bilimsel hesaplamalar için SymPy kullanır (offline).")

    expr = st.text_input("İfade", value="(2+3)*5 - 4^2")
    if st.button("Hesapla", type="primary"):
        try:
            val = sp.N(sp.sympify(expr.replace("^", "**")))
            st.success(f"Sonuç: {val}")
        except Exception as e:  # noqa: BLE001
            st.error(str(e))

    st.divider()
    st.subheader("Birim: Radyan/Derece (Trigonometri)")
    mode = st.radio("Mod", ["Radyan", "Derece"], horizontal=True)
    angle = st.number_input("Açı", value=30.0)
    func = st.selectbox("Fonksiyon", ["sin", "cos", "tan"])
    if st.button("Hesapla (Trig)"):
        try:
            x = float(angle)
            if mode == "Derece":
                x = np.deg2rad(x)
            if func == "sin":
                r = np.sin(x)
            elif func == "cos":
                r = np.cos(x)
            else:
                r = np.tan(x)
            st.success(f"{func}({angle} {mode.lower()}) = {r}")
        except Exception as e:  # noqa: BLE001
            st.error(str(e))


def assistant_page() -> None:
    st.subheader("Akıllı Yardım (Offline)")
    st.caption("Bu bölüm, konu notlarını hızlı arama + SymPy çözüm denemesi sağlar. API anahtarı yoktur.")

    query = st.text_input("Ne arıyorsun? (örn: 'mutlak değer eşitsizlik')", placeholder="Konu, formül, örnek…")

    if query:
        matches: list[tuple[str, str]] = []

        # Dosya sisteminde ara
        notes_dir = DATA_DIR / "notes"
        if notes_dir.exists():
            for md in notes_dir.glob("*.md"):
                txt = md.read_text(encoding="utf-8", errors="ignore")
                if query.lower() in txt.lower():
                    matches.append((md.stem, md.name))

        # Builtin notlarda ara (data/ klasörü olmasa da çalışır)
        for topic_id, note in BUILTIN_NOTES.items():
            already = any(m[0] == topic_id for m in matches)
            if not already:
                txt = builtin_notes_md(topic_id)
                if query.lower() in txt.lower():
                    matches.append((topic_id, note.title))

        if matches:
            st.success(f"{len(matches)} konuda eşleşme bulundu.")
            for topic_id, label in matches[:20]:
                st.markdown(f"**{label}**")
                st.markdown(load_topic_markdown(topic_id)[:800] + "…")
                st.divider()
        else:
            st.info("Hiçbir konuda eşleşme bulunamadı.")

    st.divider()
    st.subheader("Çözüm dene (SymPy)")
    prompt = st.text_area("Sorunu yaz", height=120, placeholder="Örn: 3x-5=10 veya türev: x^3-2x")
    if st.button("Çöz", type="primary"):
        out = try_solve(prompt)
        if "error" in out:
            st.error(out["error"])
        else:
            st.json(out, expanded=False)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="∑", layout="wide")
    inject_css()

    st.title(APP_TITLE)
    st.caption("Türkiye Yüzyılı Maarif Modeli konu başlıklarına göre • API anahtarsız (offline) çalışma")

    tabs = st.tabs(["9. Sınıf", "10. Sınıf", "11. Sınıf", "12. Sınıf", "Akıllı Yardım", "Hesap Makinesi"])

    with tabs[0]:
        grade_page("9")
    with tabs[1]:
        grade_page("10")
    with tabs[2]:
        grade_page("11")
    with tabs[3]:
        grade_page("12")
    with tabs[4]:
        assistant_page()
    with tabs[5]:
        calculator_page()


if __name__ == "__main__":
    main()
