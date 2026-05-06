import io
import base64
from html import escape

import matplotlib.pyplot as plt

def plot_bar_chart(df, x, y, title):
    fig, ax = plt.subplots(figsize=(8, 5))
    df.plot(kind="bar", x=x, y=y, ax=ax, legend=False)
    ax.set_title(title)
    ax.set_ylabel(y)
    ax.set_xlabel(x)
    fig.tight_layout()
    return fig


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=140)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def generate_html_report(df, fig, title, summary=None):
    img_base64 = fig_to_base64(fig)
    plt.close(fig)
    safe_title = escape(title)
    safe_summary = escape(summary.strip()) if summary else ""
    summary_html = f"<section><h2>Tóm tắt</h2><p>{safe_summary}</p></section>" if safe_summary else ""

    html = f"""
    <!doctype html>
    <html lang="vi">
    <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>{safe_title}</title>
        <style>
            body {{
                margin: 0;
                color: #1f2937;
                background: #f8fafc;
                font-family: Arial, sans-serif;
                line-height: 1.5;
            }}
            main {{
                max-width: 1040px;
                margin: 0 auto;
                padding: 32px 20px 48px;
            }}
            h1 {{
                margin: 0 0 20px;
                color: #111827;
                font-size: 30px;
            }}
            h2 {{
                margin: 0 0 8px;
                color: #111827;
                font-size: 18px;
            }}
            section {{
                margin-bottom: 24px;
            }}
            .table-wrap {{
                overflow-x: auto;
                background: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 14px;
            }}
            th, td {{
                padding: 10px 12px;
                border-bottom: 1px solid #e5e7eb;
                text-align: left;
            }}
            th {{
                background: #f3f4f6;
                color: #374151;
                font-weight: 700;
            }}
            tr:last-child td {{
                border-bottom: 0;
            }}
            img {{
                display: block;
                max-width: 100%;
                height: auto;
                margin-top: 24px;
                background: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
            }}
        </style>
    </head>
    <body>
        <main>
            <h1>{safe_title}</h1>
            {summary_html}
            <section>
                <h2>Dữ liệu</h2>
                <div class="table-wrap">
                    {df.to_html(index=False, escape=True)}
                </div>
                <img src="data:image/png;base64,{img_base64}" alt="Biểu đồ {safe_title}" />
            </section>
        </main>
    </body>
    </html>
    """
    return html
