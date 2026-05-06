from agent.agent import OlistAgent

def main():
    agent = OlistAgent()
    df = agent.analyze_favorite_products(year=2018, top_n=10)
    report = agent.generate_report(df, title="Top sản phẩm yêu thích nhất năm 2018", y_col="avg_review_score")
    with open("report_favorite_products.html", "w", encoding="utf-8") as f:
        f.write(report)
    print("Report generated: report_favorite_products.html")

if __name__ == "__main__":
    main()
