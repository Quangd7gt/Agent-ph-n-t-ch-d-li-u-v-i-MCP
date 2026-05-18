/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        page: "#050505",
        sidebar: "#0b0b0b",
        panel: "#1f1f1f",
        panelHover: "#2a2a2a",
        line: "#303030",
        muted: "#a5a5a5",
        accent: "#10a37f"
      },
      borderRadius: {
        ui: "8px"
      }
    }
  },
  plugins: []
};
