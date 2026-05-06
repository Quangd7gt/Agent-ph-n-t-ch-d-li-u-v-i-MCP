from agent.agent import OlistAgent
def main():
    agent = OlistAgent()
    df = agent.analyze_top_products(year=2018, month=5, top_n=10)
    report = agent.generate_report(df, title="Top sản phẩm bán chạy tháng 5/2018")
    with open("report_top_products.html", "w", encoding="utf-8") as f:
        f.write(report)
    print("Report generated: report_top_products.html")

if __name__ == "__main__":
    main()
