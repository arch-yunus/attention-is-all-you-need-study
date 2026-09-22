"""
İnteraktif Transformer ve LLM Dashboard Üretici Scripti.

Kullanım:
    python generate_dashboard.py
"""

import sys
import os

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.visualizer import generate_html_dashboard


def main():
    print("[*] Transformer & LLM Interaktif Dashboard olusturuluyor...")
    output_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    generated_file = generate_html_dashboard(output_path=output_path)
    print(f"[+] Dashboard basariyla olusturuldu: {os.path.abspath(generated_file)}")
    print("[+] Tarayicinizda (browser) acarak grafikleri ve formulleri inceleyebilirsiniz.")


if __name__ == "__main__":
    main()
