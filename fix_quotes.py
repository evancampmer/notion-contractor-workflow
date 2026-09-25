with open("supplier_screening.py", "r", encoding="utf-8") as f:
    content = f.read()

content = (
    content
    .replace("“", '"')
    .replace("”", '"')
    .replace("‘", "'")
    .replace("’", "'")
)

with open("supplier_screening.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Done — curly quotes removed.")