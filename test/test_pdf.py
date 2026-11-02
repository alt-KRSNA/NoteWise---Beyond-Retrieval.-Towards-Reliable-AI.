import fitz 

doc = fitz.open("AI_Sample.pdf")

print("Pages: ", len(doc))