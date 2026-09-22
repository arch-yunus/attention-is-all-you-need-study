"""
İnteraktif Görselleştirici ve Raporlama Modülü (Interactive Visualizer).

Bu modül, Transformer ve modern LLM bileşenlerini görselleştirmek için
modern, tek dosyalık interaktif HTML panelleri (Dashboards) ve SVG grafikleri üretir:
1. Dikkat Haritaları (Multi-Head vs Multi-Query vs Grouped-Query Attention)
2. Sinüzoidal ve RoPE Pozisyonel Kodlama Frekansları
3. KV-Cache Bellek Büyüme Eğrileri
4. Tokenizasyon ve Alt Kelime (BPE) Dağılımları
"""

import json
from typing import List, Dict, Any, Optional
import torch


def generate_html_dashboard(
    title: str = "Transformer & LLM Modern Mimarileri - İnteraktif Panel",
    memory_data: Optional[Dict[str, Any]] = None,
    output_path: str = "dashboard.html",
) -> str:
    """
    Kapsamlı ve duyarlı bir HTML/CSS/JS görselleştirme panosu üretir.
    """
    if memory_data is None:
        # Örnek bağlam uzunlukları için KV-cache verileri (MB cinsinden)
        seq_lens = [512, 1024, 2048, 4096, 8192, 16384, 32768]
        # 32 katman, 32 baş, d_model=4096, FP16
        # Bytes = 2 * 2 * n_kv_heads * d_k * n_layers * seq_len
        d_k = 128
        n_layers = 32
        mha_mb = [(2 * 2 * 32 * d_k * n_layers * s) / (1024 * 1024) for s in seq_lens]
        gqa_mb = [(2 * 2 * 8 * d_k * n_layers * s) / (1024 * 1024) for s in seq_lens]
        mqa_mb = [(2 * 2 * 1 * d_k * n_layers * s) / (1024 * 1024) for s in seq_lens]
        memory_data = {
            "seq_lens": seq_lens,
            "mha": [round(x, 1) for x in mha_mb],
            "gqa": [round(x, 1) for x in gqa_mb],
            "mqa": [round(x, 1) for x in mqa_mb],
        }

    html_content = f"""<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-tertiary: #21262d;
            --text-main: #f0f6fc;
            --text-muted: #8b949e;
            --accent-blue: #58a6ff;
            --accent-purple: #bc8cff;
            --accent-green: #3fb950;
            --accent-orange: #d29922;
            --accent-red: #f85149;
            --border: #30363d;
        }}
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
        }}
        body {{
            background-color: var(--bg-primary);
            color: var(--text-main);
            padding: 2rem;
            line-height: 1.6;
        }}
        header {{
            text-align: center;
            margin-bottom: 2.5rem;
            padding-bottom: 1.5rem;
            border-bottom: 1px solid var(--border);
        }}
        h1 {{
            font-size: 2.2rem;
            color: var(--accent-blue);
            margin-bottom: 0.5rem;
        }}
        .subtitle {{
            color: var(--text-muted);
            font-size: 1.1rem;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}
        .card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }}
        .card h2 {{
            font-size: 1.3rem;
            color: var(--accent-purple);
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .formula-box {{
            background: var(--bg-tertiary);
            padding: 0.8rem;
            border-radius: 6px;
            font-family: monospace;
            color: #7ee787;
            margin: 0.8rem 0;
            overflow-x: auto;
            border-left: 3px solid var(--accent-green);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }}
        th, td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        th {{
            background: var(--bg-tertiary);
            color: var(--accent-blue);
        }}
        .badge {{
            display: inline-block;
            padding: 0.2rem 0.6rem;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 600;
        }}
        .badge-green {{ background: rgba(63, 185, 80, 0.2); color: #3fb950; }}
        .badge-purple {{ background: rgba(188, 140, 255, 0.2); color: #bc8cff; }}
        .badge-orange {{ background: rgba(210, 153, 34, 0.2); color: #d29922; }}
        .badge-blue {{ background: rgba(88, 166, 255, 0.2); color: #58a6ff; }}
        .chart-container {{
            position: relative;
            height: 300px;
            width: 100%;
        }}
    </style>
</head>
<body>
    <header>
        <h1>⚡ {title}</h1>
        <p class="subtitle">2017 Referans Transformer'dan 2026 Modern Decoder-Only LLM Mimarilerine Mimari ve Matematiksel Analiz</p>
    </header>

    <div class="grid">
        <!-- Kart 1: KV-Cache Bellek Tasarrufu Grafiği -->
        <div class="card">
            <h2>📊 KV-Cache Bellek Ölçekleme Karşılaştırması</h2>
            <p style="color: var(--text-muted); font-size: 0.9rem;">
                70B Sınıfı Modelde (32 Katman, 32 Baş, FP16) Sekans Uzunluğuna Göre Bellek Tüketimi (MB):
            </p>
            <div class="chart-container">
                <canvas id="kvCacheChart"></canvas>
            </div>
        </div>

        <!-- Kart 2: Mimari Karşılaştırma Matrisi -->
        <div class="card">
            <h2>🏛️ Modern Dikkat Varyantları</h2>
            <table>
                <thead>
                    <tr>
                        <th>Mimari</th>
                        <th>KV Baş Sayısı</th>
                        <th>Bellek Tasarrufu</th>
                        <th>Kullanılan Modeller</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>MHA</strong> (Orijinal)</td>
                        <td>H (Örn. 32)</td>
                        <td><span class="badge badge-orange">0% (Referans)</span></td>
                        <td>Vaswani 2017, GPT-3</td>
                    </tr>
                    <tr>
                        <td><strong>GQA</strong> (Gruplanmış)</td>
                        <td>G (Örn. 8)</td>
                        <td><span class="badge badge-green">%75 Tasarruf</span></td>
                        <td>LLaMA-3, Mistral 7B</td>
                    </tr>
                    <tr>
                        <td><strong>MQA</strong> (Tekil Baş)</td>
                        <td>1</td>
                        <td><span class="badge badge-green">%96.8 Tasarruf</span></td>
                        <td>PaLM, Falcon 40B</td>
                    </tr>
                    <tr>
                        <td><strong>ALiBi</strong></td>
                        <td>Doğrusal Eğim</td>
                        <td><span class="badge badge-purple">Ekstrapolasyon</span></td>
                        <td>BLOOM, MPT-7B</td>
                    </tr>
                    <tr>
                        <td><strong>SWA</strong> (Kayan Pencere)</td>
                        <td>W (Pencere)</td>
                        <td><span class="badge badge-blue">O(N * W) Maliyet</span></td>
                        <td>Mistral 7B</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <!-- Kart 3: Modern Katmanlar & Formüller -->
        <div class="card">
            <h2>📐 Modern Normalizasyon ve Aktivasyonlar</h2>
            <div>
                <strong>RMSNorm (Zhang & Sennrich 2019):</strong>
                <div class="formula-box">
                    RMS(x) = sqrt( (1/d) * sum(x_i^2) + eps )<br>
                    y = (x / RMS(x)) * gamma
                </div>
            </div>
            <div>
                <strong>SwiGLU (Noam Shazeer 2020):</strong>
                <div class="formula-box">
                    SwiGLU(x) = (SiLU(x * W_gate) * (x * W_up)) * W_down
                </div>
            </div>
            <div>
                <strong>RoPE - Döner Pozisyonel Kodlama (Su et al., 2021):</strong>
                <div class="formula-box">
                    R_theta,m * x = [ x1*cos(m*theta) - x2*sin(m*theta), x1*sin(m*theta) + x2*cos(m*theta) ]
                </div>
            </div>
        </div>

        <!-- Kart 4: Yeni Nesil Örnekleme Algoritmaları -->
        <div class="card">
            <h2>🎯 İleri Seviye Çıkarım & Min-P Örnekleme</h2>
            <p style="color: var(--text-muted); font-size: 0.9rem; margin-bottom: 0.5rem;">
                Top-P ve Sıcaklık parametrelerinin halüsinasyon ve daralma sorunlarını çözen modern filtreleme:
            </p>
            <div class="formula-box">
                Min-P Kriteri: P(token) >= P_max * min_p (Örn: min_p = 0.05)
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Yöntem</th>
                        <th>Dinamik Eşik</th>
                        <th>Tekrarı Önleme</th>
                        <th>Kalite Dengesi</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Greedy</strong></td>
                        <td>Sabit (Argmax)</td>
                        <td>Düşük</td>
                        <td>Deterministik</td>
                    </tr>
                    <tr>
                        <td><strong>Top-P (Nucleus)</strong></td>
                        <td>Kümülatif Toplam</td>
                        <td>Orta</td>
                        <td>İyi (Düşük Sıcaklıkta Kısıtlı)</td>
                    </tr>
                    <tr>
                        <td><strong>Min-P Sampling</strong></td>
                        <td>Zirveye Orantılı</td>
                        <td>Yüksek</td>
                        <td>Mükemmel (Doğal Çeşitlilik)</td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>

    <script>
        const ctx = document.getElementById('kvCacheChart').getContext('2d');
        const memoryData = {json.dumps(memory_data)};

        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: memoryData.seq_lens.map(s => s + ' Tok'),
                datasets: [
                    {{
                        label: 'MHA (32 KV Heads)',
                        data: memoryData.mha,
                        borderColor: '#f85149',
                        backgroundColor: 'rgba(248, 81, 73, 0.1)',
                        borderWidth: 2,
                        tension: 0.3
                    }},
                    {{
                        label: 'GQA (8 KV Heads - LLaMA-3)',
                        data: memoryData.gqa,
                        borderColor: '#3fb950',
                        backgroundColor: 'rgba(63, 185, 80, 0.1)',
                        borderWidth: 2,
                        tension: 0.3
                    }},
                    {{
                        label: 'MQA (1 KV Head - PaLM)',
                        data: memoryData.mqa,
                        borderColor: '#58a6ff',
                        backgroundColor: 'rgba(88, 166, 255, 0.1)',
                        borderWidth: 2,
                        tension: 0.3
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        labels: {{ color: '#f0f6fc' }}
                    }}
                }},
                scales: {{
                    x: {{
                        grid: {{ color: '#30363d' }},
                        ticks: {{ color: '#8b949e' }}
                    }},
                    y: {{
                        title: {{ display: true, text: 'Bellek (MB)', color: '#8b949e' }},
                        grid: {{ color: '#30363d' }},
                        ticks: {{ color: '#8b949e' }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_path
